#!/usr/bin/env python3
"""AICE Experiment Entry Point.

Runs all conditions × domains × seeds, then generates figures.

Usage:
    python main.py                          # Run full experiment
    python main.py --condition adaptive_imitation --domain retail --seed 0  # Single run
    python main.py --plots-only             # Generate plots from existing results
"""

import argparse
import logging
import sys

from config import ALL_DOMAINS, CONDITIONS, N_SEEDS

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
    parser.add_argument("--condition", type=str, default=None,
                        choices=CONDITIONS,
                        help="Run a single condition (default: all)")
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
        generate_all_figures(args.output_dir)
        return

    # Import here to avoid loading heavy dependencies when only plotting
    from data.load_data import load_cseo_bench
    from simulation.runner import run_simulation

    conditions = [args.condition] if args.condition else CONDITIONS
    domains = [args.domain] if args.domain else ALL_DOMAINS
    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    total_runs = len(conditions) * len(domains) * len(seeds)
    logger.info(f"Starting AICE experiment: {len(conditions)} conditions × "
                f"{len(domains)} domains × {len(seeds)} seeds = {total_runs} runs")

    # Recommended execution order: no_optimization first, then fixed_geo, then adaptive_imitation
    ordered_conditions = sorted(
        conditions,
        key=lambda c: {"no_optimization": 0, "fixed_geo": 1, "adaptive_imitation": 2}[c],
    )

    run_count = 0
    for seed in seeds:
        # Load dataset once per seed
        logger.info(f"\nLoading dataset for seed={seed}...")
        dataset = load_cseo_bench(seed=seed)

        for domain in domains:
            if domain not in dataset:
                logger.warning(f"Domain '{domain}' not found in dataset, skipping")
                continue

            queries = dataset[domain]

            for condition in ordered_conditions:
                run_count += 1
                logger.info(f"\n{'#'*60}")
                logger.info(f"RUN {run_count}/{total_runs}: "
                            f"{condition} | {domain} | seed={seed}")
                logger.info(f"{'#'*60}")

                try:
                    run_simulation(
                        condition=condition,
                        domain=domain,
                        seed=seed,
                        queries=queries,
                        output_dir=args.output_dir,
                    )
                except Exception:
                    logger.exception(f"Run failed: {condition}_{domain}_seed{seed}")
                    continue

    # Generate figures
    logger.info("\nGenerating figures...")
    from analysis.plots import generate_all_figures
    generate_all_figures(args.output_dir)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
