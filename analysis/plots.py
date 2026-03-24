import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ALL_DOMAINS, CONDITIONS, EVALUATION_ROUNDS, N_ROUNDS, N_SEEDS
from features.extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Consistent color scheme
COLORS = {
    "adaptive_imitation": "#d62728",  # red
    "fixed_geo": "#1f77b4",  # blue
    "no_optimization": "#7f7f7f",  # gray
}
LABELS = {
    "adaptive_imitation": "Adaptive Imitation",
    "fixed_geo": "Fixed GEO",
    "no_optimization": "No Optimization",
}


def load_all_results(results_dir: str = "results") -> dict:
    """Load all experiment results.

    Returns nested dict: results[condition][domain][seed][round] = metrics
    """
    all_data = {}
    for condition in CONDITIONS:
        all_data[condition] = {}
        for domain in ALL_DOMAINS:
            all_data[condition][domain] = {}
            for seed in range(N_SEEDS):
                run_dir = os.path.join(
                    results_dir, f"{condition}_{domain}_seed{seed}"
                )
                if not os.path.exists(run_dir):
                    continue

                all_data[condition][domain][seed] = {}
                for fname in sorted(os.listdir(run_dir)):
                    if fname.startswith("round_") and fname.endswith("_metrics.json"):
                        try:
                            r = int(fname.split("_")[1])
                            with open(os.path.join(run_dir, fname)) as f:
                                all_data[condition][domain][seed][r] = json.load(f)
                        except (ValueError, json.JSONDecodeError):
                            pass

    return all_data


def _get_metric_over_rounds(
    all_data: dict,
    condition: str,
    domain: str,
    metric_name: str,
    rounds: list | None = None,
) -> tuple:
    """Extract metric values over rounds, averaged across seeds.

    Returns (rounds_array, mean_array, std_array).
    """
    if rounds is None:
        rounds = list(range(N_ROUNDS + 1))

    seed_data = all_data.get(condition, {}).get(domain, {})
    if not seed_data:
        return np.array(rounds), np.zeros(len(rounds)), np.zeros(len(rounds))

    values_per_round = {r: [] for r in rounds}
    for seed, round_data in seed_data.items():
        for r in rounds:
            if r in round_data and metric_name in round_data[r]:
                val = round_data[r][metric_name]
                if val is not None:
                    values_per_round[r].append(val)

    means = []
    stds = []
    valid_rounds = []
    for r in rounds:
        if values_per_round[r]:
            valid_rounds.append(r)
            means.append(np.mean(values_per_round[r]))
            stds.append(np.std(values_per_round[r]))

    return np.array(valid_rounds), np.array(means), np.array(stds)


def figure1_goodhart_collapse(
    all_data: dict,
    domain: str = "retail",
    save_path: str = "results/figure1_goodhart_collapse.pdf",
):
    """Figure 1: 4-panel main result showing Goodhart collapse."""
    from engine.ranker import get_domain_type

    domain_type = get_domain_type(domain)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Goodhart Collapse — {domain.replace('_', ' ').title()}", fontsize=14)

    panels = [
        ("content_diversity", "Content Diversity\n(Intra-corpus Similarity)", None),
        ("ranking_stability", "Ranking Stability\n(Kendall's τ)", None),
        ("classifier_auc", "Classifier AUC", None),
    ]

    if domain_type == "qa":
        panels.append(("completeness", "Completeness\n(Aspect Coverage)", EVALUATION_ROUNDS))
    else:
        panels.append(("constraint_satisfaction", "Constraint Satisfaction", EVALUATION_ROUNDS))

    for idx, (metric, ylabel, rounds) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]

        for condition in CONDITIONS:
            r, m, s = _get_metric_over_rounds(all_data, condition, domain, metric, rounds)
            if len(r) == 0:
                continue
            color = COLORS[condition]
            label = LABELS[condition]
            ax.plot(r, m, color=color, label=label, linewidth=2)
            ax.fill_between(r, m - s, m + s, color=color, alpha=0.15)

        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 1 to {save_path}")


def figure2_feature_waterfall(
    all_data: dict,
    domain: str = "retail",
    save_path: str = "results/figure2_feature_waterfall.pdf",
):
    """Figure 2: Feature coefficient heatmap over rounds."""
    condition = "adaptive_imitation"
    seed_data = all_data.get(condition, {}).get(domain, {})

    # Collect coefficient matrices across seeds
    all_matrices = []
    for seed, round_data in seed_data.items():
        matrix = np.zeros((len(FEATURE_NAMES), N_ROUNDS + 1))
        for r in range(N_ROUNDS + 1):
            if r in round_data and "feature_coefficients" in round_data[r]:
                coeffs = round_data[r]["feature_coefficients"]
                if coeffs:
                    for fi, fname in enumerate(FEATURE_NAMES):
                        matrix[fi, r] = abs(coeffs.get(fname, 0))
        all_matrices.append(matrix)

    if not all_matrices:
        logger.warning("No data for Figure 2")
        return

    avg_matrix = np.mean(all_matrices, axis=0)

    # Sort features by the round they first drop below threshold
    threshold = 0.1
    sort_keys = []
    for fi in range(len(FEATURE_NAMES)):
        row = avg_matrix[fi]
        drop_round = N_ROUNDS + 1
        for r in range(N_ROUNDS + 1):
            if row[r] < threshold and r > 0:
                drop_round = r
                break
        sort_keys.append(drop_round)

    sorted_indices = np.argsort(sort_keys)
    sorted_matrix = avg_matrix[sorted_indices]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        sorted_matrix,
        xticklabels=list(range(N_ROUNDS + 1)),
        yticklabels=sorted_names,
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature Coefficient Waterfall — {domain.replace('_', ' ').title()}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 2 to {save_path}")


def figure3_cross_domain(
    all_data: dict,
    save_path: str = "results/figure3_cross_domain.pdf",
):
    """Figure 3: Content diversity over rounds for all domains."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Content Diversity Across Domains (Adaptive Imitation)", fontsize=14)

    condition = "adaptive_imitation"

    for idx, domain in enumerate(ALL_DOMAINS):
        ax = axes[idx // 3][idx % 3]
        r, m, s = _get_metric_over_rounds(all_data, condition, domain, "content_diversity")

        if len(r) > 0:
            color = COLORS[condition]
            ax.plot(r, m, color=color, linewidth=2)
            ax.fill_between(r, m - s, m + s, color=color, alpha=0.15)

        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("Round")
        ax.set_ylabel("Content Diversity")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 3 to {save_path}")


def figure4_ranking_instability(
    all_data: dict,
    domain: str = "retail",
    save_path: str = "results/figure4_ranking_instability.pdf",
):
    """Figure 4: Ranking instability deep dive for a single query."""
    # This figure requires per-randomization ranking data which we don't store
    # in the standard metrics. Create a placeholder visualization using
    # ranking_stability metric trend instead.

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in CONDITIONS:
        r, m, s = _get_metric_over_rounds(
            all_data, condition, domain, "ranking_stability"
        )
        if len(r) > 0:
            color = COLORS[condition]
            label = LABELS[condition]
            ax.plot(r, m, color=color, label=label, linewidth=2)
            ax.fill_between(r, m - s, m + s, color=color, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel("Ranking Stability (Kendall's τ)")
    ax.set_title(f"Ranking Instability — {domain.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 4 to {save_path}")


def table1_summary(
    all_data: dict,
    save_path: str = "results/table1_summary.csv",
):
    """Table 1: Summary statistics across domains."""
    from engine.ranker import get_domain_type

    rows = []
    for domain in ALL_DOMAINS:
        domain_type = get_domain_type(domain)

        for condition in ["adaptive_imitation", "fixed_geo"]:
            # Goodhart threshold: first round where AUC < 0.6
            _, auc_means, _ = _get_metric_over_rounds(
                all_data, condition, domain, "classifier_auc"
            )
            goodhart_round = None
            for i, val in enumerate(auc_means):
                if val < 0.6:
                    goodhart_round = i
                    break

            # Content diversity at round 30
            _, div_means, _ = _get_metric_over_rounds(
                all_data, condition, domain, "content_diversity"
            )
            div_30 = div_means[-1] if len(div_means) > 0 else None

            # Ranking stability at round 30
            _, stab_means, _ = _get_metric_over_rounds(
                all_data, condition, domain, "ranking_stability"
            )
            stab_30 = stab_means[-1] if len(stab_means) > 0 else None

            # Quality degradation
            quality_metric = "completeness" if domain_type == "qa" else "constraint_satisfaction"
            _, qual_means, _ = _get_metric_over_rounds(
                all_data, condition, domain, quality_metric, EVALUATION_ROUNDS
            )
            if len(qual_means) >= 2:
                quality_delta = qual_means[-1] - qual_means[0]
            else:
                quality_delta = None

            rows.append({
                "domain": domain,
                "condition": condition,
                "goodhart_threshold_round": goodhart_round,
                "quality_degradation": quality_delta,
                "content_diversity_round30": div_30,
                "ranking_stability_round30": stab_30,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved Table 1 to {save_path}")
    return df


def generate_all_figures(results_dir: str = "results"):
    """Generate all figures and tables from experiment results."""
    all_data = load_all_results(results_dir)

    figure1_goodhart_collapse(all_data, domain="retail",
                               save_path=os.path.join(results_dir, "figure1_goodhart_collapse.pdf"))
    figure2_feature_waterfall(all_data, domain="retail",
                               save_path=os.path.join(results_dir, "figure2_feature_waterfall.pdf"))
    figure3_cross_domain(all_data,
                          save_path=os.path.join(results_dir, "figure3_cross_domain.pdf"))
    figure4_ranking_instability(all_data, domain="retail",
                                 save_path=os.path.join(results_dir, "figure4_ranking_instability.pdf"))
    table1_summary(all_data,
                    save_path=os.path.join(results_dir, "table1_summary.csv"))

    # Generate appendix figures for other domains
    for domain in ALL_DOMAINS:
        if domain == "retail":
            continue
        figure1_goodhart_collapse(
            all_data, domain=domain,
            save_path=os.path.join(results_dir, f"appendix_figure1_{domain}.pdf"),
        )
        figure2_feature_waterfall(
            all_data, domain=domain,
            save_path=os.path.join(results_dir, f"appendix_figure2_{domain}.pdf"),
        )

    logger.info("All figures generated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_figures()
