from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, TargetEncoder


class EnsembleFreightModel:
    """
    Production-grade weighted ensemble combining:
      1. CatBoostRegressor (gradient boosting with native categorical route handling)
      2. HistGradientBoostingRegressor (regularized with out-of-fold target encoding)
      3. RidgeCV (L2 regularized linear model for baseline stability)

    Supports both Direct Rate prediction and Rate-Per-Mile (RPM) target formulation.
    """

    def __init__(
        self,
        cat_features: List[str],
        weights: Tuple[float, float, float] = (0.80, 0.16, 0.04),
        target_mode: str = "direct",
        floor_rate: float = 10.0,
        seed: int = 42,
    ):
        self.cat_features = cat_features
        self.w_cb, self.w_hgb, self.w_ridge = weights
        self.target_mode = target_mode
        self.floor_rate = floor_rate
        self.seed = seed

        self.cb_model: CatBoostRegressor | None = None
        self.hgb_model: HistGradientBoostingRegressor | None = None
        self.ridge_model: RidgeCV | None = None
        self.target_encoder: TargetEncoder | None = None
        self.scaler: StandardScaler | None = None

        self.feature_cols: List[str] | None = None
        self.num_cols: List[str] | None = None

    def _transform_target(self, y: pd.Series, distances: pd.Series | None) -> np.ndarray:
        if self.target_mode == "rpm":
            if distances is None:
                raise ValueError("distances required when target_mode is 'rpm'")
            return (y / distances.clip(lower=1.0)).values
        return y.values

    def _inverse_transform_target(self, p: np.ndarray, distances: pd.Series | None) -> np.ndarray:
        if self.target_mode == "rpm":
            if distances is None:
                raise ValueError("distances required when target_mode is 'rpm'")
            return p * distances.values
        return p

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        distances_train: pd.Series | None = None,
        distances_val: pd.Series | None = None,
        cb_iterations: int = 1200,
        cb_learning_rate: float = 0.04,
        cb_depth: int = 6,
        early_stopping_rounds: int = 100,
        verbose: int = 200,
    ) -> EnsembleFreightModel:
        """Trains the full ensemble on input features."""
        self.feature_cols = X_train.columns.tolist()
        self.num_cols = [c for c in self.feature_cols if c not in self.cat_features]

        # Target transformations
        y_tr = self._transform_target(y_train, distances_train)
        y_va = self._transform_target(y_val, distances_val) if (X_val is not None and y_val is not None) else None

        # 1. CatBoost
        print(f"[{self.__class__.__name__}] Training CatBoost (iterations={cb_iterations}, lr={cb_learning_rate})...")
        self.cb_model = CatBoostRegressor(
            iterations=cb_iterations,
            learning_rate=cb_learning_rate,
            depth=cb_depth,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.seed,
            thread_count=-1,
            verbose=verbose,
        )

        if X_val is not None and y_va is not None:
            self.cb_model.fit(
                X_train,
                y_tr,
                cat_features=self.cat_features,
                eval_set=(X_val, y_va),
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        else:
            self.cb_model.fit(
                X_train,
                y_tr,
                cat_features=self.cat_features,
                verbose=verbose,
            )

        # 2. HistGradientBoosting with KFold Target Encoding
        print(f"[{self.__class__.__name__}] Training HistGradientBoosting with Target Encoding...")
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        self.target_encoder = TargetEncoder(smooth="auto", cv=cv)
        tr_cat_enc = pd.DataFrame(
            self.target_encoder.fit_transform(X_train[self.cat_features], y_tr),
            columns=[f"{c}_te" for c in self.cat_features],
            index=X_train.index,
        )
        X_tr_hgb = pd.concat([X_train[self.num_cols], tr_cat_enc], axis=1).fillna(0)

        self.hgb_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=8,
            l2_regularization=2.0,
            random_state=self.seed,
        )
        self.hgb_model.fit(X_tr_hgb, y_tr)

        # 3. Ridge Regression on Scaled Features
        print(f"[{self.__class__.__name__}] Training RidgeCV...")
        self.scaler = StandardScaler()
        X_tr_ridge = self.scaler.fit_transform(X_tr_hgb)
        self.ridge_model = RidgeCV(alphas=np.logspace(-2, 4, 20))
        self.ridge_model.fit(X_tr_ridge, y_tr)

        print(f"[{self.__class__.__name__}] Ensemble training complete.")
        return self

    def predict(self, X_input: pd.DataFrame, distances: pd.Series | None = None) -> np.ndarray:
        """Generates ensemble weighted rate predictions."""
        if self.cb_model is None or self.hgb_model is None or self.ridge_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = X_input.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)

        # CatBoost predictions
        p_cb = self.cb_model.predict(X[self.feature_cols])

        # HistGB predictions
        cat_enc = pd.DataFrame(
            self.target_encoder.transform(X[self.cat_features]),
            columns=[f"{c}_te" for c in self.cat_features],
            index=X.index,
        )
        X_hgb = pd.concat([X[self.num_cols], cat_enc], axis=1).fillna(0)
        p_hgb = self.hgb_model.predict(X_hgb)

        # Ridge predictions
        X_ridge = self.scaler.transform(X_hgb)
        p_ridge = self.ridge_model.predict(X_ridge)

        # Weighted blend
        p_blend = self.w_cb * p_cb + self.w_hgb * p_hgb + self.w_ridge * p_ridge

        # Inverse transform if RPM mode
        final_rates = self._inverse_transform_target(p_blend, distances)
        return np.maximum(final_rates, self.floor_rate)

    def get_feature_importances(self) -> pd.Series:
        """Returns CatBoost feature importances."""
        if self.cb_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        importances = self.cb_model.get_feature_importance()
        return pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
