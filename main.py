"""
Entrypoint for Freight Rate Pricing Intelligence & Forecasting Engine.
"""

import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Freight Rate Pricing Intelligence Pipeline")
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()