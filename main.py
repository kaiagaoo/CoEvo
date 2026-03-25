#!/usr/bin/env python3
"""AICE Experiment Entry Point.

Runs all conditions × domains × seeds, then generates figures.

Usage:
    python main.py                              # Run full experiment
    python main.py --domain retail --seed 0     # Single run
    python main.py --plots-only                 # Generate plots from all results
    python main.py --plots-only --domain retail # Generate plots for retail only
"""

import argparse
import logging
import sys

from config import ALL_DOMAINS, N_SEEDS

logger = logging.getLogger("aice")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="AICE Experiment Runner")
    parser.add_argument("--domain", type=str, default=None,
                        choices=ALL_DOMAINS,
                        help="Run a single domain (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (default: all)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate plots from existing results")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.plots_only:
        from analysis.plots import generate_all_figures
        domains = [args.domain] if args.domain else None
        generate_all_figures(args.output_dir, domains=domains)
        return

    # Import here to avoid loading heavy dependencies when only plotting
    from data.load_data import load_cseo_bench
    from simulation.runner import run_simulation

    domains = [args.domain] if args.domain else ALL_DOMAINS
    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    total_runs = len(domains) * len(seeds)
    logger.info(f"Starting AICE experiment: {len(domains)} domains × "
                f"{len(seeds)} seeds = {total_runs} runs")

    run_count = 0
    for seed in seeds:
        logger.info(f"\nLoading dataset for seed={seed}...")
        dataset = load_cseo_bench(seed=seed)

        for domain in domains:
            if domain not in dataset:
                logger.warning(f"Domain '{domain}' not found in dataset, skipping")
                continue

            queries = dataset[domain]
            run_count += 1
            logger.info(f"\n{'#'*60}")
            logger.info(f"RUN {run_count}/{total_runs}: {domain} | seed={seed}")
            logger.info(f"{'#'*60}")

            try:
                run_simulation(
                    domain=domain,
                    seed=seed,
                    queries=queries,
                    output_dir=args.output_dir,
                )
            except Exception:
                logger.exception(f"Run failed: adaptive_imitation_{domain}_seed{seed}")
                continue

    # Generate figures
    logger.info("\nGenerating figures...")
    from analysis.plots import generate_all_figures
    generate_all_figures(args.output_dir, domains=domains)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
