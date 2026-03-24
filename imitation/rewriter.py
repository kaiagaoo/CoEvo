import json
import logging
import os
import tempfile
import time

import numpy as np
from openai import OpenAI

from config import (
    BATCH_POLL_INTERVAL,
    OPENAI_API_KEY,
    REWRITER_MODEL,
    REWRITE_LENGTH_TOLERANCE,
)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

FEATURE_INSTRUCTIONS = {
    "citation_density": "Add references like 'according to [source]' or '[study] found that'",
    "statistic_density": "Include more specific numbers, percentages, or data points from the existing content",
    "quote_density": "Add direct quotations attributed to experts or sources",
    "heading_density": "Break content into more sections with clear subheadings",
    "list_frequency": "Convert some prose into bullet points or numbered lists",
    "readability": "Simplify sentence structure and use clearer language",
    "avg_sentence_length": "Use shorter/longer sentences to match target",
    "query_similarity": "Make the opening more directly relevant to the query topic",
    "claim_density": "Add more specific, verifiable factual claims",
    "specificity_score": "Mention more specific names, products, tools, or organizations",
    "type_token_ratio": "Use more varied/less varied vocabulary",
    "question_density": "Add or remove rhetorical questions",
    "named_source_mentions": "Reference more specific organizations, studies, or experts by name",
    "information_density": "Increase the ratio of substantive content words",
    "semantic_uniqueness": "Differentiate your content from typical documents on this topic",
    "year_mentions": "Include more recent year references to signal freshness",
    "bold_emphasis_density": "Add emphasis markers to highlight key points",
    "vocabulary_sophistication": "Use more technical/specialized terminology",
    "sentiment_polarity": "Adjust tone to be more neutral/enthusiastic as needed",
    "paragraph_count": "Break into more/fewer paragraphs",
}

FIXED_GEO_PROMPT = (
    "Improve the following document to make it more engaging, well-structured, "
    "and informative for readers. Enhance the clarity, add clear headings or "
    "structure where appropriate, highlight key features or information, and "
    "make the content more accessible and compelling. Maintain all factual "
    "information from the original.\n\n"
    "Document:\n{document_text}\n\n"
    "Improved version:"
)


def build_adaptive_rewrite_prompt(
    document_text: str,
    top_feature_names: list,
    top_feature_targets: dict,
    current_feature_values: dict,
) -> str:
    """Build a targeted rewriting prompt based on discriminative features."""
    targets_block = ""
    for fname in top_feature_names:
        target_val = top_feature_targets.get(fname, 0)
        current_val = current_feature_values.get(fname, 0)
        instruction = FEATURE_INSTRUCTIONS.get(
            fname, f"Adjust {fname} from {current_val:.2f} toward {target_val:.2f}."
        )
        targets_block += (
            f"- {fname}: move from {current_val:.2f} to {target_val:.2f}\n"
            f"  {instruction}\n"
        )

    prompt = (
        "You are a content editor. Your task is to improve this document "
        "to better match the following quality targets, while preserving "
        "all factual information.\n\n"
        f"TARGETS:\n{targets_block}\n"
        "RULES:\n"
        "- Preserve all factual claims from the original document.\n"
        "- Do not invent new facts, statistics, or quotes.\n"
        "- Keep approximately the same document length.\n"
        "- Write naturally — the result should read like polished web content.\n\n"
        f"DOCUMENT:\n{document_text}\n\n"
        "Rewrite the document below:"
    )
    return prompt


def rewrite_documents_batch(
    queries: list,
    condition: str,
    discriminator_result: dict | None = None,
    round_num: int = 0,
    seed: int = 42,
    domain: str = "",
) -> list:
    """Rewrite documents using OpenAI Batch API.

    Args:
        queries: list of query dicts (modified in place)
        condition: "adaptive_imitation" or "fixed_geo"
        discriminator_result: output from fit_discriminator (for adaptive_imitation)
        round_num: current round
        seed: random seed
        domain: domain name

    Returns:
        list of queries with updated document texts
    """
    rng = np.random.RandomState(seed + round_num * 2000)

    batch_requests = []
    request_meta = {}  # custom_id -> (query_idx, doc_idx, original_text)

    for qi, q in enumerate(queries):
        qid = q["query_id"]
        for di, doc in enumerate(q["documents"]):
            did = doc["doc_id"]

            # Check if this document is a "winner" (top-K) — skip rewriting if so
            if condition == "adaptive_imitation" and discriminator_result:
                label = discriminator_result["labels"].get((qid, did), 0)
                if label == 1:
                    continue  # top-K document, don't rewrite

            # Heterogeneous optimization: roll probability
            if rng.random() > doc.get("optimization_probability", 0.3):
                continue  # skip this document this round

            # Build prompt
            if condition == "fixed_geo":
                prompt = FIXED_GEO_PROMPT.format(document_text=doc["text"])
            elif condition == "adaptive_imitation" and discriminator_result:
                feature_data = discriminator_result["per_doc_features"]
                current_fv = feature_data.get((qid, did), {})
                prompt = build_adaptive_rewrite_prompt(
                    document_text=doc["text"],
                    top_feature_names=discriminator_result["top_feature_names"],
                    top_feature_targets=discriminator_result["top_feature_targets"],
                    current_feature_values=current_fv,
                )
            else:
                continue

            custom_id = f"rewrite_{domain}_q{qid}_d{did}_round{round_num}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": REWRITER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 2048,
                    "temperature": 0.0,
                },
            })
            request_meta[custom_id] = {
                "query_idx": qi,
                "doc_idx": di,
                "original_text": doc["text"],
                "original_word_count": len(doc["text"].split()),
            }

    if not batch_requests:
        logger.info(f"No documents to rewrite for round {round_num}")
        return queries

    logger.info(f"Rewriting {len(batch_requests)} documents for round {round_num}")

    # Submit batch
    results = _submit_and_wait_batch(
        batch_requests, f"rewrite_{domain}_round{round_num}"
    )

    # Apply rewrites
    n_applied = 0
    n_rejected = 0
    for custom_id, response_body in results.items():
        meta = request_meta.get(custom_id)
        if meta is None:
            continue

        try:
            new_text = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            continue

        # Length check
        new_wc = len(new_text.split())
        orig_wc = meta["original_word_count"]
        if orig_wc > 0:
            ratio = abs(new_wc - orig_wc) / orig_wc
            if ratio > REWRITE_LENGTH_TOLERANCE:
                logger.warning(
                    f"Rejecting rewrite for {custom_id}: length ratio {ratio:.2f} "
                    f"({orig_wc} -> {new_wc} words)"
                )
                n_rejected += 1
                continue

        qi = meta["query_idx"]
        di = meta["doc_idx"]
        queries[qi]["documents"][di]["text"] = new_text
        n_applied += 1

    logger.info(f"Applied {n_applied} rewrites, rejected {n_rejected}")
    return queries


def _submit_and_wait_batch(batch_requests: list, tag: str) -> dict:
    """Submit batch requests to OpenAI Batch API and wait for completion."""
    if not batch_requests:
        return {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix=f"aice_{tag}_"
    ) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
        jsonl_path = f.name

    logger.info(f"[{tag}] Submitting batch with {len(batch_requests)} requests...")

    try:
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"tag": tag},
        )

        logger.info(f"[{tag}] Batch created: {batch.id}")

        while True:
            batch = client.batches.retrieve(batch.id)
            status = batch.status
            logger.info(f"[{tag}] Batch status: {status} "
                        f"(completed: {batch.request_counts.completed}/"
                        f"{batch.request_counts.total})")
            if status == "completed":
                break
            elif status in ("failed", "expired", "cancelled"):
                logger.error(f"[{tag}] Batch {status}.")
                return {}
            time.sleep(BATCH_POLL_INTERVAL)

        if batch.output_file_id is None:
            return {}

        output_content = client.files.content(batch.output_file_id).text
        results = {}
        for line in output_content.strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            body = obj.get("response", {}).get("body", {})
            if custom_id and body:
                results[custom_id] = body

        logger.info(f"[{tag}] Got {len(results)} results")
        return results

    finally:
        os.unlink(jsonl_path)
