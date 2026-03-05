"""
InvestLink Data Science Pipeline — Master entry point.

Orchestrates all stages end-to-end or selectively:

  Stage 1 — scrape    : web scraping (indicators + prices) → DB
  Stage 2 — dataset   : feature engineering + labeling → parquet
  Stage 3 — train     : GradientBoosting vs XGBoost → best_model.joblib
  Stage 4 — evaluate  : cross-validated metrics + plots
  Stage 5 — predict   : predictions on current data → DB stock_predictions

Usage:
    # Full pipeline (all stages)
    python pipeline.py --all

    # Only retrain + predict (assumes data is already in DB)
    python pipeline.py --train --predict

    # Scrape specific tickers then generate predictions
    python pipeline.py --scrape --predict --tickers VALE3 PETR4 ITUB4

    # Evaluate existing model
    python pipeline.py --evaluate

    # Full run with browser visible for debugging
    python pipeline.py --all --no-headless

Run from the data_science_pipeline root directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

DATASET_PATH = "data/training_dataset.parquet"
MODELS_DIR   = "models"


def stage_scrape(tickers: list[str] | None, headless: bool, force: bool) -> None:
    from web_scraping.run_scraping import run_indicators, run_prices
    from database.connector import get_session
    from database.queries import get_all_tickers

    if not tickers:
        session = get_session()
        tickers = get_all_tickers(session)
        session.close()

    logger.info(f"═══ Stage 1: Scraping {len(tickers)} tickers ═══")
    t0 = time.time()
    run_indicators(tickers, headless=headless)
    run_prices(tickers, force=force)
    logger.info(f"Stage 1 done in {time.time() - t0:.0f}s")


def stage_dataset(use_relative: bool) -> None:
    from data_processing.build_training_dataset import build_training_dataset

    logger.info("═══ Stage 2: Building training dataset ═══")
    t0 = time.time()
    df = build_training_dataset(use_relative=use_relative, output_path=DATASET_PATH)
    logger.info(
        f"Stage 2 done in {time.time() - t0:.0f}s — "
        f"{len(df)} rows, labels: {df['label'].value_counts().to_dict()}"
    )


def stage_train(cv_folds: int) -> str:
    from models.trainer import train

    logger.info("═══ Stage 3: Model training ═══")
    t0 = time.time()
    version = train(dataset_path=DATASET_PATH, cv_folds=cv_folds, output_dir=MODELS_DIR)
    logger.info(f"Stage 3 done in {time.time() - t0:.0f}s — version: {version}")
    return version


def stage_evaluate(cv_folds: int) -> None:
    from models.evaluator import evaluate_saved_model

    logger.info("═══ Stage 4: Model evaluation ═══")
    t0 = time.time()
    results = evaluate_saved_model(
        dataset_path=DATASET_PATH,
        models_dir=MODELS_DIR,
        cv_folds=cv_folds,
        save_plots=True,
    )
    logger.info(
        f"Stage 4 done in {time.time() - t0:.0f}s — "
        f"F1 weighted (OOF): {results['f1_weighted']}"
    )


def stage_predict(dry_run: bool) -> None:
    from models.predictor import predict, print_top_predictions

    logger.info("═══ Stage 5: Generating predictions ═══")
    t0 = time.time()
    df = predict(models_dir=MODELS_DIR, dry_run=dry_run)
    print_top_predictions(df)
    logger.info(f"Stage 5 done in {time.time() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InvestLink data science pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Stage flags
    parser.add_argument("--all",      action="store_true", help="Run all stages")
    parser.add_argument("--scrape",   action="store_true", help="Stage 1: scraping")
    parser.add_argument("--dataset",  action="store_true", help="Stage 2: build training dataset")
    parser.add_argument("--train",    action="store_true", help="Stage 3: model training")
    parser.add_argument("--evaluate", action="store_true", help="Stage 4: evaluation + plots")
    parser.add_argument("--predict",  action="store_true", help="Stage 5: predict + save to DB")

    # Options
    parser.add_argument("--tickers",    nargs="+", metavar="TICKER", help="Tickers to scrape (default: all from DB)")
    parser.add_argument("--no-headless", dest="headless", action="store_false", default=True, help="Show browser window")
    parser.add_argument("--force",      action="store_true",  help="Force re-scrape prices even if stored")
    parser.add_argument("--no-relative", dest="relative", action="store_false", default=True, help="Use absolute returns (no Ibovespa)")
    parser.add_argument("--cv",         type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument("--dry-run",    action="store_true",  help="Predict but do not save to DB")

    args = parser.parse_args()

    run_all  = args.all
    run_any  = any([args.scrape, args.dataset, args.train, args.evaluate, args.predict])

    if not run_all and not run_any:
        parser.print_help()
        sys.exit(0)

    t_start = time.time()
    logger.info("InvestLink pipeline started")

    if run_all or args.scrape:
        stage_scrape(args.tickers, headless=args.headless, force=args.force)

    if run_all or args.dataset:
        stage_dataset(use_relative=args.relative)

    if run_all or args.train:
        stage_train(cv_folds=args.cv)

    if run_all or args.evaluate:
        stage_evaluate(cv_folds=args.cv)

    if run_all or args.predict:
        stage_predict(dry_run=args.dry_run)

    logger.info(f"Pipeline complete in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
