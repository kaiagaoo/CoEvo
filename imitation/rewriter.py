import logging

import numpy as np

from api_client import submit_batch
from config import (
    REWRITER_MODEL,
    REWRITE_LENGTH_TOLERANCE,
)

logger = logging.getLogger(__name__)

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
    discriminator_result: dict,
    round_num: int = 0,
    seed: int = 42,
    domain: str = "",
) -> list:
    """Rewrite documents using adaptive imitation strategy.

    Args:
        queries: list of query dicts (modified in place)
        discriminator_result: output from fit_discriminator
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

            # Skip top-K documents (winners don't get rewritten)
            label = discriminator_result["labels"].get((qid, did), 0)
            if label == 1:
                continue

            # Heterogeneous optimization: roll probability
            if rng.random() > doc.get("optimization_probability", 0.3):
                continue  # skip this document this round

            # Build adaptive rewrite prompt
            feature_data = discriminator_result["per_doc_features"]
            current_fv = feature_data.get((qid, did), {})
            prompt = build_adaptive_rewrite_prompt(
                document_text=doc["text"],
                top_feature_names=discriminator_result["top_feature_names"],
                top_feature_targets=discriminator_result["top_feature_targets"],
                current_feature_values=current_fv,
            )

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
    results = submit_batch(
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


