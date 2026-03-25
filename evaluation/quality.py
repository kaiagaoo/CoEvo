import logging
import re

import numpy as np

from api_client import submit_batch
from config import (
    ENGINE_MODEL,
    JUDGE_MODEL,
)
from engine.ranker import get_domain_type

logger = logging.getLogger(__name__)


def generate_natural_responses_batch(
    queries: list,
    domain: str,
    round_num: int,
) -> dict:
    """Generate free-form natural responses for all queries.

    Returns dict mapping query_id -> response_text.
    """
    domain_type = get_domain_type(domain)
    batch_requests = []

    for q in queries:
        qid = q["query_id"]
        docs = q["documents"]
        doc_block = "\n\n".join(
            f"[{i + 1}] {doc['text']}" for i, doc in enumerate(docs)
        )

        if domain_type == "recommendation":
            prompt = (
                f"Here are product descriptions.\n\n"
                f"{doc_block}\n\n"
                f"User query: {q['query']}\n\n"
                f"Recommend the best products and explain why each is a good choice.\n"
                f"Cite documents by their number [1], [2], etc."
            )
        else:
            prompt = (
                f"Here are documents.\n\n"
                f"{doc_block}\n\n"
                f"User query: {q['query']}\n\n"
                f"Answer the question thoroughly using the documents.\n"
                f"Cite documents by their number [1], [2], etc."
            )

        custom_id = f"natural_{domain}_q{qid}_round{round_num}"
        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": ENGINE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2048,
                "temperature": 0.0,
            },
        })

    results = submit_batch(batch_requests, f"natural_{domain}_round{round_num}")

    responses = {}
    for q in queries:
        qid = q["query_id"]
        custom_id = f"natural_{domain}_q{qid}_round{round_num}"
        if custom_id in results:
            try:
                responses[qid] = results[custom_id]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass

    logger.info(f"Generated {len(responses)} natural responses")
    return responses


def generate_aspect_checklists_batch(
    queries: list,
    domain: str,
) -> dict:
    """Generate aspect checklists for QA queries (run once at round 0).

    Returns dict mapping query_id -> list of aspect strings.
    """
    batch_requests = []
    for q in queries:
        qid = q["query_id"]
        prompt = (
            f"For the query '{q['query']}', list 5-8 key aspects that a complete "
            f"answer should cover.\nFormat: one aspect per line, numbered."
        )
        custom_id = f"aspects_{domain}_q{qid}"
        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": ENGINE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 512,
                "temperature": 0.0,
            },
        })

    results = submit_batch(batch_requests, f"aspects_{domain}")

    checklists = {}
    for q in queries:
        qid = q["query_id"]
        custom_id = f"aspects_{domain}_q{qid}"
        if custom_id in results:
            try:
                text = results[custom_id]["choices"][0]["message"]["content"]
                aspects = [
                    line.strip()
                    for line in text.strip().split("\n")
                    if line.strip() and re.match(r"^\d+", line.strip())
                ]
                checklists[qid] = aspects
            except (KeyError, IndexError):
                pass

    logger.info(f"Generated {len(checklists)} aspect checklists")
    return checklists


def evaluate_quality_batch(
    queries: list,
    domain: str,
    round_num: int,
) -> dict:
    """Score each document's quality using GPT-4o judge.

    Returns dict mapping (query_id, doc_id) -> average quality score (1-5).
    """
    batch_requests = []
    for q in queries:
        qid = q["query_id"]
        for doc in q["documents"]:
            did = doc["doc_id"]
            prompt = (
                f"Rate how well this document answers the query '{q['query']}'.\n"
                f"Score from 1-5 on: (a) factual accuracy, (b) completeness, (c) usefulness.\n"
                f"Respond with ONLY three numbers separated by commas, e.g.: 4,3,5\n\n"
                f"Document:\n{doc['text']}"
            )
            custom_id = f"quality_{domain}_q{qid}_d{did}_round{round_num}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                },
            })

    results = submit_batch(batch_requests, f"quality_{domain}_round{round_num}")

    scores = {}
    for q in queries:
        qid = q["query_id"]
        for doc in q["documents"]:
            did = doc["doc_id"]
            custom_id = f"quality_{domain}_q{qid}_d{did}_round{round_num}"
            if custom_id in results:
                try:
                    text = results[custom_id]["choices"][0]["message"]["content"]
                    nums = re.findall(r"(\d+)", text)
                    if len(nums) >= 3:
                        vals = [int(x) for x in nums[:3]]
                        scores[(qid, did)] = float(np.mean(vals))
                except (KeyError, IndexError, ValueError):
                    pass

    logger.info(f"Evaluated {len(scores)} document quality scores")
    return scores


def check_aspect_coverage_batch(
    queries: list,
    natural_responses: dict,
    aspect_checklists: dict,
    domain: str,
    round_num: int,
) -> float:
    """Check what fraction of aspects are covered in natural responses.

    Returns mean coverage fraction across queries.
    """
    batch_requests = []
    valid_qids = []

    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses or qid not in aspect_checklists:
            continue

        response = natural_responses[qid]
        aspects = aspect_checklists[qid]
        aspects_text = "\n".join(aspects)

        prompt = (
            f"Given this response and this checklist, which aspects are covered?\n"
            f"Respond with the numbers of covered aspects, separated by commas.\n\n"
            f"Response:\n{response}\n\n"
            f"Checklist:\n{aspects_text}"
        )

        custom_id = f"coverage_{domain}_q{qid}_round{round_num}"
        batch_requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": ENGINE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 64,
                "temperature": 0.0,
            },
        })
        valid_qids.append(qid)

    if not batch_requests:
        return 0.0

    results = submit_batch(batch_requests, f"coverage_{domain}_round{round_num}")

    coverages = []
    for qid in valid_qids:
        custom_id = f"coverage_{domain}_q{qid}_round{round_num}"
        if custom_id not in results:
            continue
        try:
            text = results[custom_id]["choices"][0]["message"]["content"]
            covered_nums = re.findall(r"\d+", text)
            n_aspects = len(aspect_checklists[qid])
            if n_aspects > 0:
                coverages.append(len(set(covered_nums)) / n_aspects)
        except (KeyError, IndexError):
            pass

    return float(np.mean(coverages)) if coverages else 0.0


def evaluate_constraint_satisfaction_batch(
    queries: list,
    natural_responses: dict,
    domain: str,
    round_num: int,
) -> float:
    """For recommendation domains: check if recommended products match query constraints.

    Returns mean satisfaction fraction.
    """
    batch_requests = []
    query_rec_counts = {}

    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses:
            continue

        response = natural_responses[qid]
        # Extract recommended doc numbers from response
        rec_nums = re.findall(r"\[(\d+)\]", response)
        rec_indices = list(set(int(x) - 1 for x in rec_nums if 0 < int(x) <= len(q["documents"])))

        if not rec_indices:
            continue

        query_rec_counts[qid] = len(rec_indices)
        for idx in rec_indices:
            doc = q["documents"][idx]
            original_text = doc.get("original_text", doc["text"])
            prompt = (
                f"Query: {q['query']}\n"
                f"Original product description: {original_text}\n"
                f"Does this product match the query's requirements? "
                f"Answer YES or NO with one sentence reason."
            )
            custom_id = f"constraint_{domain}_q{qid}_d{idx}_round{round_num}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
            })

    if not batch_requests:
        return 0.0

    results = submit_batch(
        batch_requests, f"constraint_{domain}_round{round_num}"
    )

    satisfactions = []
    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses or qid not in query_rec_counts:
            continue

        response = natural_responses[qid]
        rec_nums = re.findall(r"\[(\d+)\]", response)
        rec_indices = list(set(int(x) - 1 for x in rec_nums if 0 < int(x) <= len(q["documents"])))

        n_satisfied = 0
        for idx in rec_indices:
            custom_id = f"constraint_{domain}_q{qid}_d{idx}_round{round_num}"
            if custom_id in results:
                try:
                    text = results[custom_id]["choices"][0]["message"]["content"]
                    if text.strip().upper().startswith("YES"):
                        n_satisfied += 1
                except (KeyError, IndexError):
                    pass

        if rec_indices:
            satisfactions.append(n_satisfied / len(rec_indices))

    return float(np.mean(satisfactions)) if satisfactions else 0.0


def compute_source_diversity(queries: list, natural_responses: dict) -> float:
    """Count distinct document numbers cited in natural responses."""
    diversities = []
    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses:
            continue
        response = natural_responses[qid]
        cited = set(re.findall(r"\[(\d+)\]", response))
        diversities.append(len(cited))
    return float(np.mean(diversities)) if diversities else 0.0


def compute_recommendation_diversity(queries: list, natural_responses: dict) -> float:
    """Compute mean pairwise Jaccard similarity of recommended products across queries."""
    rec_sets = []
    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses:
            continue
        response = natural_responses[qid]
        recs = set(re.findall(r"\[(\d+)\]", response))
        if recs:
            rec_sets.append(recs)

    if len(rec_sets) < 2:
        return 0.0

    jaccards = []
    for i in range(len(rec_sets)):
        for j in range(i + 1, len(rec_sets)):
            union = rec_sets[i] | rec_sets[j]
            if union:
                jaccards.append(len(rec_sets[i] & rec_sets[j]) / len(union))

    return float(np.mean(jaccards)) if jaccards else 0.0


def compute_justification_distinctiveness(
    queries: list, natural_responses: dict
) -> float:
    """Compute mean pairwise cosine similarity of justification texts within responses."""
    from features.extractor import compute_embeddings

    similarities = []
    for q in queries:
        qid = q["query_id"]
        if qid not in natural_responses:
            continue

        response = natural_responses[qid]
        # Split response into per-product sections
        sections = re.split(r"\[\d+\]", response)
        sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]

        if len(sections) < 2:
            continue

        embeddings = compute_embeddings(sections)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T

        n = len(sections)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(float(sim_matrix[i, j]))

    return float(np.mean(similarities)) if similarities else 0.0


