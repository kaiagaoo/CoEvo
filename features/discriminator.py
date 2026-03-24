import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import TOP_K_FRACTION, TOP_N_FEATURES
from features.extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)


def fit_discriminator(
    feature_data: dict,
    avg_ranks: dict,
    queries: list,
) -> dict:
    """Fit logistic regression to discriminate top-ranked from bottom-ranked documents.

    Args:
        feature_data: dict of (query_id, doc_id) -> feature_vector
        avg_ranks: dict of query_id -> {doc_id: average_rank}
        queries: list of query dicts

    Returns:
        dict with keys:
            top_feature_names: list of str
            top_feature_targets: dict {feature_name: target_value}
            per_doc_features: dict of (query_id, doc_id) -> {feature_name: value}
            classifier_auc: float
            all_coefficients: dict {feature_name: coefficient}
            labels: dict of (query_id, doc_id) -> 0 or 1
            scaler: fitted StandardScaler
    """
    # Build feature matrix and rank array
    keys = []
    feature_matrix = []
    rank_values = []

    for q in queries:
        qid = q["query_id"]
        if qid not in avg_ranks:
            continue
        for doc in q["documents"]:
            did = doc["doc_id"]
            key = (qid, did)
            if key in feature_data and did in avg_ranks[qid]:
                keys.append(key)
                fv = feature_data[key]
                feature_matrix.append([fv[fname] for fname in FEATURE_NAMES])
                rank_values.append(avg_ranks[qid][did])

    if not keys:
        logger.error("No documents with both features and ranks")
        return _empty_result()

    X = np.array(feature_matrix)
    ranks = np.array(rank_values)

    # Label: top K fraction by rank (lower rank = better)
    threshold = np.percentile(ranks, TOP_K_FRACTION * 100)
    labels = (ranks <= threshold).astype(int)

    n_positive = labels.sum()
    n_total = len(labels)
    logger.info(f"Discriminator: {n_positive}/{n_total} documents labeled as top-K "
                f"(threshold rank <= {threshold:.2f})")

    if n_positive == 0 or n_positive == n_total:
        logger.warning("All documents have same label, cannot fit classifier")
        return _empty_result()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_scaled, labels)

    # AUC on training data
    probs = clf.predict_proba(X_scaled)[:, 1]
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5

    # Coefficients
    coefficients = clf.coef_[0]
    coeff_dict = {fname: float(coefficients[i]) for i, fname in enumerate(FEATURE_NAMES)}

    # Top features by absolute coefficient
    abs_coeffs = np.abs(coefficients)
    top_indices = np.argsort(abs_coeffs)[::-1][:TOP_N_FEATURES]
    top_feature_names = [FEATURE_NAMES[i] for i in top_indices]

    # Compute target values: mean feature values among top-K documents
    top_mask = labels == 1
    top_X = X[top_mask]
    top_feature_targets = {}
    for idx in top_indices:
        fname = FEATURE_NAMES[idx]
        top_feature_targets[fname] = float(top_X[:, idx].mean())

    # Build labels dict
    labels_dict = {keys[i]: int(labels[i]) for i in range(len(keys))}

    logger.info(f"Discriminator AUC: {auc:.3f}")
    logger.info(f"Top features: {top_feature_names}")

    return {
        "top_feature_names": top_feature_names,
        "top_feature_targets": top_feature_targets,
        "per_doc_features": feature_data,
        "classifier_auc": auc,
        "all_coefficients": coeff_dict,
        "labels": labels_dict,
        "scaler": scaler,
    }


def _empty_result():
    return {
        "top_feature_names": [],
        "top_feature_targets": {},
        "per_doc_features": {},
        "classifier_auc": 0.5,
        "all_coefficients": {fname: 0.0 for fname in FEATURE_NAMES},
        "labels": {},
        "scaler": None,
    }
