import copy
import json
import logging
import os

from config import (
    CONDITIONS,
    EVALUATION_ROUNDS,
    N_ROUNDS,
)
from engine.ranker import rank_documents_batch_with_stability
from evaluation.metrics import (
    compute_evaluation_round_metrics,
    compute_every_round_metrics,
)
from features.discriminator import fit_discriminator
from features.extractor import extract_features_batch
from imitation.rewriter import rewrite_documents_batch

logger = logging.getLogger(__name__)


def run_simulation(
    condition: str,
    domain: str,
    seed: int,
    queries: list,
    output_dir: str = "results",
) -> dict:
    """Run a single simulation (one condition × one domain × one seed).

    Args:
        condition: one of ["adaptive_imitation", "fixed_geo", "no_optimization"]
        domain: domain name
        seed: random seed
        queries: list of query dicts (deep-copied internally)
        output_dir: base output directory

    Returns:
        dict of round_num -> metrics dict
    """
    run_dir = os.path.join(output_dir, f"{condition}_{domain}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Deep copy queries so we don't mutate the original
    queries = copy.deepcopy(queries)

    all_results = {}
    aspect_checklists = None  # For QA domains, generated at round 0

    # Check for resume capability
    resume_round = _find_resume_round(run_dir)
    if resume_round > 0:
        logger.info(f"Resuming from round {resume_round}")
        # Load state
        queries = _load_queries_snapshot(run_dir, resume_round - 1)
        all_results = _load_results(run_dir)
        aspect_checklists = _load_aspect_checklists(run_dir)

    for round_num in range(resume_round, N_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num} | {condition} | {domain} | seed={seed}")
        logger.info(f"{'='*60}")

        discriminator_result = None
        feature_data = None

        # --- Phase 2: Feature extraction and rewriting (rounds > 0) ---
        if round_num > 0:
            if condition == "no_optimization":
                pass  # Documents never change

            elif condition == "fixed_geo":
                # Rewrite with fixed prompt, no feature analysis
                queries = rewrite_documents_batch(
                    queries=queries,
                    condition="fixed_geo",
                    round_num=round_num,
                    seed=seed,
                    domain=domain,
                )

            elif condition == "adaptive_imitation":
                # Phase 2A-B: Extract features
                logger.info("Phase 2A-B: Extracting features...")
                feature_data = extract_features_batch(queries, domain)

                # Phase 2C: Fit discriminator
                logger.info("Phase 2C: Fitting discriminator...")
                # Use previous round's rankings
                prev_ranks = all_results.get(round_num - 1, {}).get("avg_ranks", {})
                discriminator_result = fit_discriminator(
                    feature_data, prev_ranks, queries
                )

                # Phase 2D: Rewrite
                logger.info("Phase 2D: Rewriting documents...")
                queries = rewrite_documents_batch(
                    queries=queries,
                    condition="adaptive_imitation",
                    discriminator_result=discriminator_result,
                    round_num=round_num,
                    seed=seed,
                    domain=domain,
                )

        # --- Phase 1: Forced ranking ---
        logger.info("Phase 1: Ranking documents...")
        avg_ranks, per_query_rand_rankings = rank_documents_batch_with_stability(
            queries=queries,
            domain=domain,
            round_num=round_num,
            seed=seed,
        )

        # --- Features for round 0 or if not yet computed ---
        if feature_data is None and condition == "adaptive_imitation":
            logger.info("Computing features for metrics...")
            feature_data = extract_features_batch(queries, domain)
            discriminator_result = fit_discriminator(feature_data, avg_ranks, queries)

        # --- Every-round metrics ---
        logger.info("Computing every-round metrics...")
        metrics = compute_every_round_metrics(
            queries=queries,
            avg_ranks=avg_ranks,
            per_query_rand_rankings=per_query_rand_rankings,
            discriminator_result=discriminator_result,
            domain=domain,
            condition=condition,
        )
        metrics["avg_ranks"] = avg_ranks

        # --- Evaluation-round metrics ---
        if round_num in EVALUATION_ROUNDS:
            logger.info("Computing evaluation-round metrics...")
            eval_metrics, natural_responses, aspect_checklists = (
                compute_evaluation_round_metrics(
                    queries=queries,
                    avg_ranks=avg_ranks,
                    feature_data=feature_data,
                    discriminator_result=discriminator_result,
                    domain=domain,
                    condition=condition,
                    round_num=round_num,
                    aspect_checklists=aspect_checklists,
                )
            )
            metrics.update(eval_metrics)

        all_results[round_num] = metrics

        # Save round results and document snapshot
        _save_round(run_dir, round_num, metrics, queries, aspect_checklists)

        logger.info(f"Round {round_num} complete. Key metrics:")
        for k, v in metrics.items():
            if k not in ("avg_ranks", "feature_coefficients") and v is not None:
                logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    logger.info(f"\nSimulation complete: {condition}_{domain}_seed{seed}")
    return all_results


def _save_round(run_dir, round_num, metrics, queries, aspect_checklists):
    """Save round results and document snapshot to disk."""
    # Save metrics (excluding non-serializable objects)
    metrics_to_save = {}
    for k, v in metrics.items():
        if k == "avg_ranks":
            # Convert int keys to strings for JSON
            metrics_to_save[k] = {
                str(qid): {str(did): rank for did, rank in doc_ranks.items()}
                for qid, doc_ranks in v.items()
            }
        elif isinstance(v, dict):
            metrics_to_save[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            metrics_to_save[k] = v

    metrics_path = os.path.join(run_dir, f"round_{round_num:03d}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)

    # Save document snapshot
    snapshot = []
    for q in queries:
        snapshot.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "text": d["text"],
                    "optimization_probability": d.get("optimization_probability", 0),
                }
                for d in q["documents"]
            ],
        })
    snapshot_path = os.path.join(run_dir, f"round_{round_num:03d}_docs.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Save aspect checklists if available
    if aspect_checklists:
        checklists_path = os.path.join(run_dir, "aspect_checklists.json")
        with open(checklists_path, "w") as f:
            json.dump(
                {str(k): v for k, v in aspect_checklists.items()},
                f, indent=2,
            )


def _find_resume_round(run_dir: str) -> int:
    """Find the last completed round to resume from."""
    if not os.path.exists(run_dir):
        return 0
    max_round = -1
    for fname in os.listdir(run_dir):
        if fname.startswith("round_") and fname.endswith("_metrics.json"):
            try:
                r = int(fname.split("_")[1])
                max_round = max(max_round, r)
            except (ValueError, IndexError):
                pass
    return max_round + 1 if max_round >= 0 else 0


def _load_queries_snapshot(run_dir: str, round_num: int) -> list:
    """Load document snapshot from a previous round."""
    path = os.path.join(run_dir, f"round_{round_num:03d}_docs.json")
    with open(path) as f:
        return json.load(f)


def _load_results(run_dir: str) -> dict:
    """Load all saved round results."""
    results = {}
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("round_") and fname.endswith("_metrics.json"):
            try:
                r = int(fname.split("_")[1])
                path = os.path.join(run_dir, fname)
                with open(path) as f:
                    data = json.load(f)
                # Convert string keys back to int where needed
                if "avg_ranks" in data:
                    data["avg_ranks"] = {
                        int(qid): {int(did): rank for did, rank in doc_ranks.items()}
                        for qid, doc_ranks in data["avg_ranks"].items()
                    }
                results[r] = data
            except (ValueError, IndexError, json.JSONDecodeError):
                pass
    return results


def _load_aspect_checklists(run_dir: str) -> dict | None:
    """Load aspect checklists if they exist."""
    path = os.path.join(run_dir, "aspect_checklists.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return None
