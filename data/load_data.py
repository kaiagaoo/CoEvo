import copy
import logging
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from config import BETA_A, BETA_B, N_QUERIES_PER_DOMAIN

logger = logging.getLogger(__name__)

# Map our canonical domain names to the actual HuggingFace split names
SPLIT_MAP = {
    "retail": "retail",
    "video_games": "videogames",
    "books": "books",
    "web": "web",
    "news": "news",
    "debate": "debate",
}


def load_cseo_bench(seed: int = 42) -> dict:
    """Load C-SEO Bench dataset from HuggingFace.

    The dataset has splits: retail, videogames, books, news, web, debate.
    Each row has columns: query_id, query, document.
    Rows sharing the same query_id belong to the same query.

    Returns a dict keyed by domain name, each value is a list of query dicts:
    [
        {
            "query_id": int,
            "query": str,
            "documents": [
                {"doc_id": int, "text": str, "original_text": str,
                 "optimization_probability": float}
            ]
        }
    ]
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    ds = load_dataset("parameterlab/c-seo-bench")

    logger.info(f"Dataset splits: {list(ds.keys())}")
    for split_name in ds:
        logger.info(f"  {split_name}: columns={ds[split_name].column_names}, "
                     f"num_rows={len(ds[split_name])}")

    result = {}
    for domain_name, split_name in SPLIT_MAP.items():
        if split_name not in ds:
            logger.warning(f"Split '{split_name}' not found for domain '{domain_name}', skipping")
            continue

        split_data = ds[split_name]
        logger.info(f"Loading domain '{domain_name}' from split '{split_name}'")

        # Group rows by query_id
        query_groups = defaultdict(lambda: {"query": None, "documents": []})
        for row in split_data:
            qid = row["query_id"]
            query_groups[qid]["query"] = row["query"]
            query_groups[qid]["documents"].append(row["document"])

        # Convert to list sorted by original query_id
        queries = []
        for qid in sorted(query_groups.keys()):
            g = query_groups[qid]
            queries.append({
                "query": g["query"],
                "documents": [{"text": doc_text} for doc_text in g["documents"]],
            })

        logger.info(f"  Parsed {len(queries)} queries, "
                     f"avg {np.mean([len(q['documents']) for q in queries]):.1f} docs/query")

        # Sample N_QUERIES_PER_DOMAIN queries
        if len(queries) > N_QUERIES_PER_DOMAIN:
            sampled_indices = py_rng.sample(range(len(queries)), N_QUERIES_PER_DOMAIN)
            queries = [queries[i] for i in sorted(sampled_indices)]
        elif len(queries) < N_QUERIES_PER_DOMAIN:
            logger.warning(f"  Domain '{domain_name}' has only {len(queries)} queries, "
                           f"less than {N_QUERIES_PER_DOMAIN}")

        # Assign sequential query_ids, doc_ids, and optimization probabilities
        for qidx, q in enumerate(queries):
            q["query_id"] = qidx
            for didx, doc in enumerate(q["documents"]):
                doc["doc_id"] = didx
                doc["original_text"] = copy.deepcopy(doc["text"])
                doc["optimization_probability"] = float(rng.beta(BETA_A, BETA_B))

        result[domain_name] = queries
        logger.info(f"  Final: {len(queries)} queries, "
                     f"avg {np.mean([len(q['documents']) for q in queries]):.1f} docs/query")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_cseo_bench(seed=42)
    for domain, queries in data.items():
        print(f"\nDomain: {domain}")
        print(f"  Queries: {len(queries)}")
        if queries:
            q0 = queries[0]
            print(f"  Sample query: {q0['query'][:100]}...")
            print(f"  Docs per query: {len(q0['documents'])}")
            if q0["documents"]:
                d0 = q0["documents"][0]
                print(f"  Sample doc text: {d0['text'][:100]}...")
                print(f"  Optimization prob: {d0['optimization_probability']:.3f}")
