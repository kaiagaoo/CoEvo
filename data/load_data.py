import copy
import logging
import random

import numpy as np
from datasets import load_dataset

from config import BETA_A, BETA_B, N_QUERIES_PER_DOMAIN

logger = logging.getLogger(__name__)


def load_cseo_bench(seed: int = 42) -> dict:
    """Load C-SEO Bench dataset from HuggingFace.

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

    # Inspect available splits/partitions
    logger.info(f"Dataset splits: {list(ds.keys())}")
    for split_name in ds:
        logger.info(f"  {split_name}: columns={ds[split_name].column_names}, "
                     f"num_rows={len(ds[split_name])}")

    # Map partition names to our domain names
    # The dataset may use different naming conventions; we try common patterns
    domain_map = _build_domain_map(ds)
    logger.info(f"Domain mapping: {domain_map}")

    result = {}
    for domain_name, split_name in domain_map.items():
        split_data = ds[split_name]
        columns = split_data.column_names
        logger.info(f"Loading domain '{domain_name}' from split '{split_name}', "
                     f"columns: {columns}")

        # Parse queries and documents from the split
        queries = _parse_split(split_data, columns)
        logger.info(f"  Parsed {len(queries)} queries for domain '{domain_name}'")

        # Sample N_QUERIES_PER_DOMAIN queries
        if len(queries) > N_QUERIES_PER_DOMAIN:
            sampled_indices = py_rng.sample(range(len(queries)), N_QUERIES_PER_DOMAIN)
            queries = [queries[i] for i in sorted(sampled_indices)]
        elif len(queries) < N_QUERIES_PER_DOMAIN:
            logger.warning(f"  Domain '{domain_name}' has only {len(queries)} queries, "
                           f"less than {N_QUERIES_PER_DOMAIN}")

        # Assign query_ids and doc_ids, optimization probabilities
        for qidx, q in enumerate(queries):
            q["query_id"] = qidx
            for didx, doc in enumerate(q["documents"]):
                doc["doc_id"] = didx
                doc["original_text"] = copy.deepcopy(doc["text"])
                doc["optimization_probability"] = float(
                    rng.beta(BETA_A, BETA_B)
                )

        result[domain_name] = queries
        logger.info(f"  Final: {len(queries)} queries, "
                     f"avg {np.mean([len(q['documents']) for q in queries]):.1f} docs/query")

    return result


def _build_domain_map(ds) -> dict:
    """Map our expected domain names to actual dataset split names."""
    expected_domains = ["retail", "video_games", "books", "web", "news", "debate"]
    splits = list(ds.keys())

    domain_map = {}
    for domain in expected_domains:
        # Try exact match first
        if domain in splits:
            domain_map[domain] = domain
            continue
        # Try case-insensitive match
        for split in splits:
            if split.lower().replace("-", "_").replace(" ", "_") == domain:
                domain_map[domain] = split
                break
        else:
            # Try substring match
            for split in splits:
                if domain in split.lower():
                    domain_map[domain] = split
                    break

    # If we couldn't map all domains, try to infer from available splits
    if len(domain_map) < len(expected_domains):
        unmapped = [d for d in expected_domains if d not in domain_map]
        unused_splits = [s for s in splits if s not in domain_map.values()]
        logger.warning(f"Could not map domains: {unmapped}. "
                       f"Available unused splits: {unused_splits}")
        # If there's a single split (e.g., 'train'), we need to check for
        # a 'domain' column inside it
        if len(splits) == 1 or "train" in splits:
            main_split = "train" if "train" in splits else splits[0]
            data = ds[main_split]
            cols = data.column_names
            # Look for a domain/category column
            domain_col = None
            for c in cols:
                if c.lower() in ("domain", "category", "task", "subset", "partition"):
                    domain_col = c
                    break
            if domain_col:
                unique_vals = set(data[domain_col])
                logger.info(f"Found domain column '{domain_col}' with values: {unique_vals}")
                # We'll handle this in a special way - return the main split
                # and let _parse_split handle domain filtering
                for domain in expected_domains:
                    domain_map[domain] = f"{main_split}:{domain_col}:{domain}"

    return domain_map


def _parse_split(split_data, columns) -> list:
    """Parse a dataset split into our internal format."""
    queries = []

    # Check if this is a filtered reference (main_split:col:value)
    # This case is handled by the caller passing the right data

    # Try to identify query and document columns
    query_col = _find_column(columns, ["query", "question", "input", "prompt"])
    doc_cols = _find_document_columns(columns)

    if query_col is None:
        logger.error(f"Cannot find query column among: {columns}")
        return queries

    for i in range(len(split_data)):
        row = split_data[i]
        query_text = row[query_col]

        docs = []
        if isinstance(doc_cols, str):
            # Single column containing a list of documents
            raw_docs = row[doc_cols]
            if isinstance(raw_docs, list):
                for d in raw_docs:
                    if isinstance(d, dict):
                        text = d.get("text", d.get("content", d.get("body", str(d))))
                    else:
                        text = str(d)
                    docs.append({"text": text})
            elif isinstance(raw_docs, str):
                docs.append({"text": raw_docs})
        elif isinstance(doc_cols, list):
            # Multiple columns, each containing one document
            for col in doc_cols:
                val = row[col]
                if val is not None and str(val).strip():
                    docs.append({"text": str(val)})

        if docs:
            queries.append({
                "query": query_text,
                "documents": docs,
            })

    return queries


def _find_column(columns, candidates):
    """Find the first matching column name."""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Substring match
    for cand in candidates:
        for c in columns:
            if cand in c.lower():
                return c
    return None


def _find_document_columns(columns):
    """Find document column(s).

    Could be a single list column or multiple individual document columns.
    """
    # Try single list column
    for name in ["documents", "docs", "passages", "candidates", "responses"]:
        for c in columns:
            if c.lower() == name:
                return c

    # Try multiple doc columns (doc_0, doc_1, ... or document_1, document_2, ...)
    import re
    doc_pattern = re.compile(r"(doc|document|passage|response|candidate)s?[_\-]?\d+", re.I)
    matched = [c for c in columns if doc_pattern.match(c)]
    if matched:
        return sorted(matched)

    # Try columns that contain "doc" or "text" or "content"
    for name in ["doc", "text", "content", "body", "answer"]:
        matches = [c for c in columns if name in c.lower() and c.lower() not in (
            "query", "question", "input")]
        if matches:
            return matches if len(matches) > 1 else matches[0]

    return None


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
