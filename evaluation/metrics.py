import logging

import numpy as np
from scipy.stats import spearmanr

from engine.ranker import (
    compute_kendall_tau_from_rankings,
    get_domain_type,
)
from evaluation.quality import (
    check_aspect_coverage_batch,
    compute_justification_distinctiveness,
    compute_recommendation_diversity,
    compute_source_diversity,
    evaluate_constraint_satisfaction_batch,
    evaluate_quality_batch,
    generate_aspect_checklists_batch,
    generate_natural_responses_batch,
)
from features.extractor import FEATURE_NAMES, compute_content_diversity, compute_embeddings

logger = logging.getLogger(__name__)


def compute_every_round_metrics(
    queries: list,
    avg_ranks: dict,
    per_query_rand_rankings: dict,
    discriminator_result: dict | None,
    domain: str,
    condition: str,
) -> dict:
    """Compute metrics that are collected every round.

    Returns dict of metric_name -> value.
    """
    metrics = {}

    # 1. Content diversity
    metrics["content_diversity"] = compute_content_diversity(queries)

    # 2. Classifier AUC
    if discriminator_result and condition == "adaptive_imitation":
        metrics["classifier_auc"] = discriminator_result["classifier_auc"]
    else:
        metrics["classifier_auc"] = None

    # 3. Ranking stability (Kendall's tau)
    metrics["ranking_stability"] = compute_kendall_tau_from_rankings(
        per_query_rand_rankings
    )

    # 4. Feature coefficients
    if discriminator_result and condition == "adaptive_imitation":
        metrics["feature_coefficients"] = discriminator_result["all_coefficients"]
    else:
        metrics["feature_coefficients"] = None

    return metrics


def compute_evaluation_round_metrics(
    queries: list,
    avg_ranks: dict,
    feature_data: dict | None,
    discriminator_result: dict | None,
    domain: str,
    condition: str,
    round_num: int,
    aspect_checklists: dict | None = None,
) -> tuple:
    """Compute metrics that are only collected at evaluation rounds.

    Returns (metrics_dict, natural_responses, aspect_checklists).
    """
    metrics = {}
    domain_type = get_domain_type(domain)

    # Generate natural responses
    natural_responses = generate_natural_responses_batch(queries, domain, round_num)

    # Quality scores (Goodhart correlation)
    quality_scores = evaluate_quality_batch(queries, domain, round_num)
    metrics["mean_quality_score"] = (
        float(np.mean(list(quality_scores.values())))
        if quality_scores else 0.0
    )

    # Goodhart correlation: feature similarity to top-K vs quality score
    if feature_data and discriminator_result and discriminator_result["top_feature_targets"]:
        x_vals = []
        y_vals = []
        top_targets = discriminator_result["top_feature_targets"]
        top_names = discriminator_result["top_feature_names"]

        for q in queries:
            qid = q["query_id"]
            for doc in q["documents"]:
                did = doc["doc_id"]
                key = (qid, did)
                if key in feature_data and key in quality_scores:
                    fv = feature_data[key]
                    # Compute similarity to top-K mean (simple Euclidean distance, inverted)
                    diffs = [
                        (fv.get(fn, 0) - top_targets.get(fn, 0)) ** 2
                        for fn in top_names
                    ]
                    dist = np.sqrt(sum(diffs))
                    x_vals.append(-dist)  # negate so higher = more similar
                    y_vals.append(quality_scores[key])

        if len(x_vals) > 5:
            rho, pval = spearmanr(x_vals, y_vals)
            metrics["goodhart_correlation"] = float(rho) if not np.isnan(rho) else 0.0
        else:
            metrics["goodhart_correlation"] = None
    else:
        metrics["goodhart_correlation"] = None

    # Domain-specific metrics
    if domain_type == "qa":
        # Completeness (aspect coverage)
        if aspect_checklists is None and round_num == 0:
            aspect_checklists = generate_aspect_checklists_batch(queries, domain)

        if aspect_checklists:
            metrics["completeness"] = check_aspect_coverage_batch(
                queries, natural_responses, aspect_checklists, domain, round_num
            )
        else:
            metrics["completeness"] = None

        # Source diversity
        metrics["source_diversity"] = compute_source_diversity(
            queries, natural_responses
        )

    elif domain_type == "recommendation":
        # Recommendation diversity (Jaccard)
        metrics["recommendation_diversity"] = compute_recommendation_diversity(
            queries, natural_responses
        )

        # Justification distinctiveness
        metrics["justification_distinctiveness"] = compute_justification_distinctiveness(
            queries, natural_responses
        )

        # Constraint satisfaction
        metrics["constraint_satisfaction"] = evaluate_constraint_satisfaction_batch(
            queries, natural_responses, domain, round_num
        )

    return metrics, natural_responses, aspect_checklists
