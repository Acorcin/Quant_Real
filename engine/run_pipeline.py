#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrates the entire backtest and walkforward test pipeline:
1. Ingests all Databento tick data into Postgres (process_databento.py).
2. Generates historical feature vectors for the training period (generate_historical_features.py).
3. Trains the meta-model and runs the out-of-sample walkforward test (walkforward_test.py).
"""

import sys
import subprocess
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run_command(command: list[str]) -> bool:
    """Run a command as a subprocess, streaming output to logs."""
    logger.info("Running: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Stream stdout/stderr in real time
    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
    rc = process.wait()
    if rc != 0:
        logger.error("Command failed with exit code %d: %s", rc, " ".join(command))
        return False
    logger.info("Command completed successfully: %s", " ".join(command))
    return True


def main():
    start_time = time.time()
    logger.info("=========================================")
    logger.info("STARTING BACKTEST & prediction PIPELINE")
    logger.info("=========================================")

    # Step 1: Databento Tick Ingestion
    logger.info("--- Step 1: Ingesting Databento tick data ---")
    if not run_command([sys.executable, "process_databento.py"]):
        sys.exit(1)

    # Step 2: Feature Generation (Up to OOS cutoff)
    logger.info("--- Step 2: Generating historical features ---")
    if not run_command([sys.executable, "generate_historical_features.py", "--end", "2025-04-04"]):
        sys.exit(1)

    # Step 3: Meta-Model Training & Walkforward Test
    logger.info("--- Step 3: Model training and walkforward backtest ---")
    if not run_command([
        sys.executable,
        "walkforward_test.py",
        "--force-train",
        "--output",
        "walkforward_results.json"
    ]):
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("=========================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Total elapsed time: %d minutes %d seconds", int(elapsed // 60), int(elapsed % 60))
    logger.info("=========================================")


if __name__ == "__main__":
    main()
