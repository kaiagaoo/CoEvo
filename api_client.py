"""Unified LLM API client supporting OpenAI (Batch API) and Google Gemini.

Set API_PROVIDER="openai" or API_PROVIDER="gemini" in .env to choose.
Both providers expose the same interface: submit_batch() which accepts a list
of request dicts and returns {custom_id: response_body} where response_body
follows the OpenAI format {"choices": [{"message": {"content": "..."}}]}.
"""

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BATCH_POLL_INTERVAL

logger = logging.getLogger(__name__)

API_PROVIDER = os.environ.get("API_PROVIDER", "gemini").lower()

# Gemini concurrency limit
GEMINI_MAX_WORKERS = int(os.environ.get("GEMINI_MAX_WORKERS", "10"))

# Model mapping: OpenAI -> Gemini equivalents
GEMINI_MODEL_MAP = {
    "gpt-4o-mini": "gemini-2.5-flash-lite",
    "gpt-4o": "gemini-2.5-flash-lite",
}


def _get_gemini_model(openai_model: str) -> str:
    return GEMINI_MODEL_MAP.get(openai_model, "gemini-2.5-flash-lite")


def submit_batch(batch_requests: list, tag: str) -> dict:
    """Submit a batch of LLM requests and return results.

    Args:
        batch_requests: list of dicts, each with:
            - custom_id: str
            - body: {"model": str, "messages": [...], "max_completion_tokens": int, "temperature": float}
            (the "method" and "url" fields are ignored for Gemini)
        tag: descriptive tag for logging

    Returns:
        dict mapping custom_id -> response body in OpenAI format:
        {"choices": [{"message": {"content": "response text"}}]}
    """
    if not batch_requests:
        return {}

    if API_PROVIDER == "openai":
        return _submit_openai_batch(batch_requests, tag)
    elif API_PROVIDER == "gemini":
        return _submit_gemini_concurrent(batch_requests, tag)
    else:
        raise ValueError(f"Unknown API_PROVIDER: {API_PROVIDER}. Use 'openai' or 'gemini'.")


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _submit_openai_batch(batch_requests: list, tag: str) -> dict:
    client = _get_openai_client()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix=f"aice_{tag}_"
    ) as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
        jsonl_path = f.name

    logger.info(f"[{tag}] OpenAI batch: submitting {len(batch_requests)} requests...")

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
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0
            logger.info(f"[{tag}] Batch status: {status} ({completed}/{total})")

            if status == "completed":
                break
            elif status in ("failed", "expired", "cancelled"):
                logger.error(f"[{tag}] Batch {status}. Errors: {batch.errors}")
                return {}
            time.sleep(BATCH_POLL_INTERVAL)

        if batch.output_file_id is None:
            logger.error(f"[{tag}] No output file")
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


# ---------------------------------------------------------------------------
# Google Gemini (concurrent individual requests)
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


GEMINI_MAX_RETRIES = 10
GEMINI_RETRY_BASE_DELAY = 5  # seconds


def _call_gemini_single(custom_id: str, body: dict) -> tuple:
    """Make a single Gemini API call with retry on rate limits.

    Returns (custom_id, response_body).
    """
    client = _get_gemini_client()
    from google.genai import types

    openai_model = body.get("model", "gpt-4o-mini")
    gemini_model = _get_gemini_model(openai_model)

    messages = body.get("messages", [])
    prompt = messages[0]["content"] if messages else ""

    max_tokens = body.get("max_completion_tokens", 1024)
    temperature = body.get("temperature", 0.0)

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            text = response.text or ""
            return custom_id, {
                "choices": [{"message": {"content": text}}]
            }
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                import random
                delay = min(GEMINI_RETRY_BASE_DELAY * (2 ** attempt), 120)
                delay += random.uniform(0, delay * 0.5)  # jitter
                logger.warning(f"Rate limited for {custom_id}, retrying in {delay:.0f}s "
                               f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES})")
                time.sleep(delay)
            else:
                logger.warning(f"Gemini call failed for {custom_id}: {e}")
                return custom_id, None

    logger.error(f"Gemini call exhausted retries for {custom_id}")
    return custom_id, None


def _submit_gemini_concurrent(batch_requests: list, tag: str) -> dict:
    logger.info(f"[{tag}] Gemini concurrent: submitting {len(batch_requests)} requests "
                f"(max_workers={GEMINI_MAX_WORKERS})...")

    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _call_gemini_single,
                req["custom_id"],
                req.get("body", req),
            ): req["custom_id"]
            for req in batch_requests
        }

        for future in as_completed(futures):
            custom_id, response_body = future.result()
            if response_body is not None:
                results[custom_id] = response_body
            completed += 1
            if completed % 50 == 0 or completed == len(batch_requests):
                logger.info(f"[{tag}] Progress: {completed}/{len(batch_requests)}")

    logger.info(f"[{tag}] Got {len(results)}/{len(batch_requests)} results")
    return results
