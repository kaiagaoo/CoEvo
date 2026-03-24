import logging
import math
import re
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_nlp = None
_embedding_model = None
_common_words = None

FEATURE_NAMES = [
    # Structural (8)
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "paragraph_count",
    "heading_density",
    "list_frequency",
    "readability",
    "bold_emphasis_density",
    # Evidentiary (8)
    "citation_density",
    "statistic_density",
    "quote_density",
    "named_source_mentions",
    "year_mentions",
    "claim_density",
    "external_reference_density",
    "question_density",
    # Semantic (9)
    "query_similarity",
    "corpus_centroid_similarity",
    "type_token_ratio",
    "vocabulary_sophistication",
    "sentiment_polarity",
    "avg_word_length",
    "semantic_uniqueness",
    "information_density",
    "specificity_score",
]


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_common_words():
    """Get set of ~5000 most common English words."""
    global _common_words
    if _common_words is not None:
        return _common_words

    # Use a built-in frequency approach: we'll use nltk's word list or a simple fallback
    try:
        from importlib import resources
        # Use a simple heuristic: words from Brown corpus frequency list
        import nltk
        try:
            from nltk.corpus import brown
            freq = Counter(w.lower() for w in brown.words())
        except LookupError:
            nltk.download("brown", quiet=True)
            from nltk.corpus import brown
            freq = Counter(w.lower() for w in brown.words())
        _common_words = set(w for w, _ in freq.most_common(5000))
    except Exception:
        # Fallback: use a minimal set of very common words
        logger.warning("Could not load frequency list, using minimal common words set")
        _common_words = set()

    return _common_words


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute sentence embeddings for a list of texts."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    return np.array(embeddings)


def extract_features_batch(
    queries: list,
    domain: str,
) -> dict:
    """Extract features for all documents in all queries.

    Args:
        queries: list of query dicts
        domain: domain name

    Returns:
        dict mapping (query_id, doc_id) -> feature_vector dict
    """
    nlp = _get_nlp()

    # Collect all texts for batch processing
    all_doc_texts = []
    all_query_texts = []
    all_keys = []
    for q in queries:
        for doc in q["documents"]:
            all_doc_texts.append(doc["text"])
            all_query_texts.append(q["query"])
            all_keys.append((q["query_id"], doc["doc_id"]))

    logger.info(f"Extracting features for {len(all_doc_texts)} documents in domain '{domain}'")

    # Batch compute embeddings
    logger.info("  Computing embeddings...")
    doc_embeddings = compute_embeddings(all_doc_texts)
    query_embeddings = compute_embeddings(all_query_texts)
    corpus_centroid = doc_embeddings.mean(axis=0)

    # Batch compute spaCy NLP
    logger.info("  Running spaCy pipeline...")
    spacy_docs = list(nlp.pipe(all_doc_texts, batch_size=64))

    # Compute per-query group embeddings for semantic_uniqueness
    query_group_embeddings = {}
    for i, key in enumerate(all_keys):
        qid = key[0]
        if qid not in query_group_embeddings:
            query_group_embeddings[qid] = []
        query_group_embeddings[qid].append((key[1], doc_embeddings[i]))

    # Extract features
    features = {}
    for i, key in enumerate(all_keys):
        qid, did = key
        text = all_doc_texts[i]
        doc_emb = doc_embeddings[i]
        query_emb = query_embeddings[i]
        spacy_doc = spacy_docs[i]

        fv = _extract_single(
            text=text,
            doc_embedding=doc_emb,
            query_embedding=query_emb,
            corpus_centroid=corpus_centroid,
            spacy_doc=spacy_doc,
            query_group=query_group_embeddings[qid],
            doc_id=did,
        )
        features[key] = fv

    logger.info(f"  Extracted {len(features)} feature vectors")
    return features


def _extract_single(
    text: str,
    doc_embedding: np.ndarray,
    query_embedding: np.ndarray,
    corpus_centroid: np.ndarray,
    spacy_doc,
    query_group: list,
    doc_id: int,
) -> dict:
    """Extract all 25 features for a single document."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        word_count = 1  # avoid division by zero

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = max(len(paragraphs), 1)

    per_500 = 500.0 / max(word_count, 1)
    per_300 = 300.0 / max(word_count, 1)
    per_100 = 100.0 / max(word_count, 1)

    # --- STRUCTURAL ---
    # heading_density: short lines, no period, possibly caps
    lines = text.split("\n")
    headings = sum(
        1 for line in lines
        if line.strip()
        and len(line.strip().split()) <= 10
        and not line.strip().endswith(".")
        and (line.strip()[0].isupper() if line.strip() else False)
    )
    heading_density = headings * per_500

    # list_frequency
    list_items = sum(
        1 for line in lines
        if re.match(r"^\s*[-*•]|\s*\d+\.", line)
    )
    list_frequency = list_items * per_500

    # readability (Flesch-Kincaid)
    try:
        import textstat
        readability = textstat.flesch_kincaid_grade(text)
    except Exception:
        readability = 8.0  # default

    # bold_emphasis_density
    bold_count = len(re.findall(r"\*\*[^*]+\*\*", text))
    bold_emphasis_density = bold_count * per_500

    # --- EVIDENTIARY ---
    # citation_density
    citation_patterns = (
        len(re.findall(r"\[\d+\]", text))
        + len(re.findall(r"\[source\]", text, re.I))
        + len(re.findall(r"\([A-Z][a-z]+,?\s*\d{4}\)", text))
        + len(re.findall(r"according to", text, re.I))
    )
    citation_density = citation_patterns / max(paragraph_count, 1)

    # statistic_density
    stat_count = (
        len(re.findall(r"\d+\.?\d*%", text))
        + len(re.findall(r"\$\d+", text))
        + len(re.findall(r"\b\d{2,}\b", text))
    )
    statistic_density = stat_count * per_300

    # quote_density
    quote_count = len(re.findall(r'"[^"]{5,}"', text))
    quote_density = quote_count * per_500

    # named_source_mentions (spaCy NER)
    org_person = sum(
        1 for ent in spacy_doc.ents if ent.label_ in ("ORG", "PERSON")
    )
    named_source_mentions = org_person

    # year_mentions
    year_count = len(re.findall(r"\b(199\d|20[0-2]\d|2030)\b", text))
    year_mentions = year_count

    # claim_density
    claim_keywords = re.compile(
        r"\b(more than|better|worse|fastest|largest|smallest|highest|lowest|"
        r"increased|decreased|improved|compared|significant)\b",
        re.I,
    )
    claim_sentences = sum(
        1 for s in sentences
        if re.search(r"\d", s) or claim_keywords.search(s)
    )
    claim_density = claim_sentences / max(sentence_count, 1)

    # external_reference_density
    url_count = len(re.findall(r"https?://", text))
    external_reference_density = url_count * per_500

    # question_density
    question_count = len(re.findall(r"\?", text))
    question_density = question_count * per_500

    # --- SEMANTIC ---
    # query_similarity
    query_similarity = float(_cosine_sim(doc_embedding, query_embedding))

    # corpus_centroid_similarity
    corpus_centroid_similarity = float(_cosine_sim(doc_embedding, corpus_centroid))

    # type_token_ratio
    words_lower = [w.lower() for w in words]
    type_token_ratio = len(set(words_lower)) / max(len(words_lower), 1)

    # vocabulary_sophistication
    common = _get_common_words()
    if common:
        uncommon = sum(1 for w in words_lower if w.isalpha() and w not in common)
        alpha_words = sum(1 for w in words_lower if w.isalpha())
        vocabulary_sophistication = uncommon / max(alpha_words, 1)
    else:
        vocabulary_sophistication = 0.0

    # sentiment_polarity
    try:
        from textblob import TextBlob
        sentiment_polarity = TextBlob(text).sentiment.polarity
    except Exception:
        sentiment_polarity = 0.0

    # avg_word_length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0

    # semantic_uniqueness
    max_sim = 0.0
    for other_did, other_emb in query_group:
        if other_did != doc_id:
            sim = _cosine_sim(doc_embedding, other_emb)
            max_sim = max(max_sim, sim)
    semantic_uniqueness = 1.0 - max_sim

    # information_density (content words / total words)
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    content_words = sum(1 for token in spacy_doc if token.pos_ in content_pos)
    information_density = content_words / max(len(spacy_doc), 1)

    # specificity_score (entities per 100 words)
    entity_count = len(spacy_doc.ents)
    specificity_score = entity_count * per_100

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": word_count / sentence_count,
        "paragraph_count": paragraph_count,
        "heading_density": heading_density,
        "list_frequency": list_frequency,
        "readability": readability,
        "bold_emphasis_density": bold_emphasis_density,
        "citation_density": citation_density,
        "statistic_density": statistic_density,
        "quote_density": quote_density,
        "named_source_mentions": named_source_mentions,
        "year_mentions": year_mentions,
        "claim_density": claim_density,
        "external_reference_density": external_reference_density,
        "question_density": question_density,
        "query_similarity": query_similarity,
        "corpus_centroid_similarity": corpus_centroid_similarity,
        "type_token_ratio": type_token_ratio,
        "vocabulary_sophistication": vocabulary_sophistication,
        "sentiment_polarity": sentiment_polarity,
        "avg_word_length": float(avg_word_length),
        "semantic_uniqueness": semantic_uniqueness,
        "information_density": information_density,
        "specificity_score": specificity_score,
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_content_diversity(queries: list) -> float:
    """Compute mean intra-group cosine similarity across query groups.

    Higher value = more homogeneous = less diverse.
    """
    all_texts_by_query = {}
    for q in queries:
        texts = [doc["text"] for doc in q["documents"]]
        all_texts_by_query[q["query_id"]] = texts

    similarities = []
    for qid, texts in all_texts_by_query.items():
        if len(texts) < 2:
            continue
        embeddings = compute_embeddings(texts)
        # Pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T
        # Extract upper triangle (excluding diagonal)
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(sim_matrix[i, j])

    return float(np.mean(similarities)) if similarities else 0.0
