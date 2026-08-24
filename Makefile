.PHONY: setup test train evaluate clean

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:
	python main.py --config configs/default_config.yaml

evaluate:
	python scripts/validate_benchmark.py --predictions validation_predictions.csv --december-predictions data/december_benchmark.csv

clean:
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__ tests/__pycache__ .pytest_cache catboost_info
