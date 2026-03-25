import logging
import re
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau

from api_client import submit_batch
from config import (
    ENGINE_MODEL,
    MIN_VALID_RANDOMIZATIONS,
    N_RANDOMIZATIONS,
)

logger = logging.getLogger(__name__)


def get_domain_type(domain: str) -> str:
    """Return 'recommendation' or 'qa' based on domain name."""
    if domain in ("retail", "video_games", "books"):
        return "recommendation"
    return "qa"


def build_ranking_prompt(query: str, documents: list, domain: str) -> str:
    """Build a forced-ranking prompt for the given query and documents.

    Documents should already be in the desired (randomized) order.
    Each document is a dict with at least a "text" key.
    """
    domain_type = get_domain_type(domain)
    n = len(documents)

    doc_block = "\n\n".join(
        f"[{i + 1}] {doc['text']}" for i, doc in enumerate(documents)
    )

    if domain_type == "recommendation":
        prompt = (
            f"You are a product recommendation engine.\n\n"
            f"Here are product descriptions. Read ALL of them carefully before ranking.\n\n"
            f"{doc_block}\n\n"
            f"User query: {query}\n\n"
            f"Rank ALL products from most recommended to least recommended.\n"
            f"You MUST include every number from 1 to {n} exactly once.\n"
            f"Format your response EXACTLY as:\n"
        )
    else:
        prompt = (
            f"You are an information retrieval engine.\n\n"
            f"Here are documents. Read ALL of them carefully before ranking.\n\n"
            f"{doc_block}\n\n"
            f"User query: {query}\n\n"
            f"Rank ALL documents from most relevant and helpful to least relevant.\n"
            f"You MUST include every number from 1 to {n} exactly once.\n"
            f"Format your response EXACTLY as:\n"
        )

    for rank in range(1, n + 1):
        prompt += f"{rank}. [number] - one sentence reason\n"

    return prompt


def parse_ranking(response_text: str, n_docs: int) -> list | None:
    """Parse LLM ranking response into ordered list of document indices (0-based).

    Returns list where position i contains the 0-based index of the document
    ranked at position i+1, or None if parsing fails.
    """
    # Primary pattern: "1. [3] - reason"
    primary_pattern = re.compile(r"^\d+\.\s*\[(\d+)\]")
    matches = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = primary_pattern.match(line)
        if m:
            matches.append(int(m.group(1)))

    if len(matches) == n_docs:
        # Convert to 0-based
        return [x - 1 for x in matches]

    # Fallback: find all bracketed numbers in order of appearance
    fallback_pattern = re.compile(r"\[(\d+)\]")
    matches = []
    seen = set()
    for m in fallback_pattern.finditer(response_text):
        val = int(m.group(1))
        if val not in seen and 1 <= val <= n_docs:
            seen.add(val)
            matches.append(val)

    if len(matches) == n_docs:
        return [x - 1 for x in matches]

    logger.warning(f"Failed to parse ranking. Got {len(matches)} valid entries, "
                   f"expected {n_docs}. Response:\n{response_text[:500]}")
    return None


def rank_documents_batch(
    queries: list,
    domain: str,
    round_num: int,
    seed: int = 42,
) -> dict:
    """Rank all documents for all queries using OpenAI Batch API.

    Args:
        queries: list of query dicts with "query_id", "query", "documents"
        domain: domain name
        round_num: current round number
        seed: random seed

    Returns:
        dict mapping query_id -> {doc_id: average_rank}
    """
    rng = np.random.RandomState(seed + round_num * 1000)

    # Build all batch requests
    batch_requests = []
    # Store mapping: custom_id -> (query_id, randomization_idx, permutation)
    request_meta = {}

    for q in queries:
        qid = q["query_id"]
        docs = q["documents"]
        n_docs = len(docs)

        for r in range(N_RANDOMIZATIONS):
            perm = rng.permutation(n_docs).tolist()
            shuffled_docs = [docs[i] for i in perm]
            prompt = build_ranking_prompt(q["query"], shuffled_docs, domain)

            custom_id = f"{domain}_q{qid}_r{r}_round{round_num}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": ENGINE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 1024,
                    "temperature": 0.0,
                },
            })
            request_meta[custom_id] = {
                "query_id": qid,
                "rand_idx": r,
                "permutation": perm,
                "n_docs": n_docs,
            }

    # Submit batch
    results = submit_batch(batch_requests, f"rank_{domain}_round{round_num}")

    # Parse results and compute average ranks
    query_ranks = defaultdict(lambda: defaultdict(list))

    for custom_id, response_body in results.items():
        meta = request_meta.get(custom_id)
        if meta is None:
            continue

        qid = meta["query_id"]
        perm = meta["permutation"]
        n_docs = meta["n_docs"]

        # Extract response text
        try:
            response_text = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning(f"No valid response for {custom_id}")
            continue

        # Parse ranking (returns shuffled-order indices)
        ranking = parse_ranking(response_text, n_docs)
        if ranking is None:
            continue

        # Map back to original doc_ids
        # ranking[rank_pos] = shuffled_index of doc at that rank
        # perm[shuffled_index] = original doc_id
        for rank_pos, shuffled_idx in enumerate(ranking):
            original_doc_id = perm[shuffled_idx]
            query_ranks[qid][original_doc_id].append(rank_pos + 1)  # 1-based rank

    # Average ranks and validate
    avg_ranks = {}
    for qid, doc_ranks in query_ranks.items():
        avg_ranks[qid] = {}
        for doc_id, ranks in doc_ranks.items():
            if len(ranks) >= MIN_VALID_RANDOMIZATIONS:
                avg_ranks[qid][doc_id] = float(np.mean(ranks))
            else:
                logger.warning(f"Query {qid}, doc {doc_id}: only {len(ranks)} valid "
                               f"randomizations (need {MIN_VALID_RANDOMIZATIONS})")
                avg_ranks[qid][doc_id] = float(np.mean(ranks))

    return avg_ranks


def compute_ranking_stability(queries: list, domain: str, round_num: int,
                               seed: int = 42) -> float:
    """Compute ranking stability as average Kendall's tau across randomization pairs.

    This requires re-doing the ranking with individual randomization results stored.
    In practice, this is computed from data already collected during rank_documents_batch.
    """
    # This is computed from the raw per-randomization rankings stored during batching
    # For now, return a placeholder - the actual computation happens in the runner
    # using per-randomization data
    return 0.0


def compute_kendall_tau_from_rankings(per_query_per_rand_rankings: dict) -> float:
    """Compute average Kendall's tau from per-query per-randomization rankings.

    Args:
        per_query_per_rand_rankings: dict of query_id -> list of ranking dicts
            where each ranking dict maps doc_id -> rank

    Returns:
        Average Kendall's tau across all query pairs of randomizations.
    """
    taus = []
    for qid, rand_rankings in per_query_per_rand_rankings.items():
        if len(rand_rankings) < 2:
            continue
        # Get common doc_ids
        common_docs = set(rand_rankings[0].keys())
        for rr in rand_rankings[1:]:
            common_docs &= set(rr.keys())
        if len(common_docs) < 2:
            continue
        doc_list = sorted(common_docs)

        for i in range(len(rand_rankings)):
            for j in range(i + 1, len(rand_rankings)):
                r1 = [rand_rankings[i][d] for d in doc_list]
                r2 = [rand_rankings[j][d] for d in doc_list]
                tau, _ = kendalltau(r1, r2)
                if not np.isnan(tau):
                    taus.append(tau)

    return float(np.mean(taus)) if taus else 0.0


def rank_documents_batch_with_stability(
    queries: list,
    domain: str,
    round_num: int,
    seed: int = 42,
) -> tuple:
    """Like rank_documents_batch but also returns per-randomization rankings for stability.

    Returns:
        (avg_ranks, per_query_per_rand_rankings)
    """
    rng = np.random.RandomState(seed + round_num * 1000)

    batch_requests = []
    request_meta = {}

    for q in queries:
        qid = q["query_id"]
        docs = q["documents"]
        n_docs = len(docs)

        for r in range(N_RANDOMIZATIONS):
            perm = rng.permutation(n_docs).tolist()
            shuffled_docs = [docs[i] for i in perm]
            prompt = build_ranking_prompt(q["query"], shuffled_docs, domain)

            custom_id = f"{domain}_q{qid}_r{r}_round{round_num}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": ENGINE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 1024,
                    "temperature": 0.0,
                },
            })
            request_meta[custom_id] = {
                "query_id": qid,
                "rand_idx": r,
                "permutation": perm,
                "n_docs": n_docs,
            }

    results = submit_batch(batch_requests, f"rank_{domain}_round{round_num}")

    # Parse results
    # per_query_per_rand: query_id -> rand_idx -> {doc_id: rank}
    per_query_per_rand = defaultdict(dict)
    query_ranks = defaultdict(lambda: defaultdict(list))

    for custom_id, response_body in results.items():
        meta = request_meta.get(custom_id)
        if meta is None:
            continue

        qid = meta["query_id"]
        r_idx = meta["rand_idx"]
        perm = meta["permutation"]
        n_docs = meta["n_docs"]

        try:
            response_text = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            continue

        ranking = parse_ranking(response_text, n_docs)
        if ranking is None:
            continue

        rand_ranking = {}
        for rank_pos, shuffled_idx in enumerate(ranking):
            original_doc_id = perm[shuffled_idx]
            rank_val = rank_pos + 1
            query_ranks[qid][original_doc_id].append(rank_val)
            rand_ranking[original_doc_id] = rank_val

        per_query_per_rand[qid][r_idx] = rand_ranking

    # Average ranks
    avg_ranks = {}
    for qid, doc_ranks in query_ranks.items():
        avg_ranks[qid] = {}
        for doc_id, ranks in doc_ranks.items():
            avg_ranks[qid][doc_id] = float(np.mean(ranks))

    # Convert per_query_per_rand to list format for stability computation
    per_query_rand_list = {}
    for qid, rand_dict in per_query_per_rand.items():
        per_query_rand_list[qid] = [
            rand_dict[r] for r in sorted(rand_dict.keys())
        ]

    return avg_ranks, per_query_rand_list


