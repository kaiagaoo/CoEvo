# AICE Implementation Guide
## Adaptive Imitation Co-Evolution Experiment

This document is a step-by-step implementation spec for the AICE experiment. Follow it sequentially. Each section describes one component to build, its inputs/outputs, and the exact logic.

---

## 0. Project Structure

```
aice/
├── config.py              # API keys, model names, hyperparameters
├── data/
│   └── load_data.py       # Load C-SEO Bench from HuggingFace
├── engine/
│   └── ranker.py          # Phase 1: forced ranking via LLM
├── features/
│   ├── extractor.py       # Phase 2A-B: compute 25 features per document
│   └── discriminator.py   # Phase 2C: logistic regression, find top features
├── imitation/
│   └── rewriter.py        # Phase 2D: targeted LLM rewriting
├── evaluation/
│   ├── quality.py         # Phase 3: natural response + quality metrics
│   └── metrics.py         # Compute all round-level metrics
├── simulation/
│   └── runner.py          # Main loop: orchestrate 30 rounds
├── analysis/
│   └── plots.py           # Generate figures from results
├── results/               # Auto-created, stores per-round outputs
└── main.py                # Entry point
```

---

## 1. Configuration (`config.py`)

```python
# config.py

import os

# API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Models
ENGINE_MODEL = "gpt-4o-mini"          # Static engine for ranking
REWRITER_MODEL = "gpt-4o-mini"        # Creator agent rewriter
JUDGE_MODEL = "gpt-4o"                # Quality evaluation judge

# Experiment parameters
N_ROUNDS = 30
N_QUERIES_PER_DOMAIN = 100
N_RANDOMIZATIONS = 5                   # Randomized doc orderings per query per round
TOP_K_FRACTION = 0.1                   # Top 10% are "winners"
TOP_N_FEATURES = 5                     # Number of discriminative features to target
EVALUATION_ROUNDS = [0, 5, 10, 15, 20, 25, 30]  # Rounds where Phase 3 runs
N_SEEDS = 3

# Creator heterogeneity
BETA_A = 2                             # Beta distribution param for optimization probability
BETA_B = 5                             # Most creators optimize moderately, few aggressively

# Domains
DOMAINS = {
    "recommendation": ["retail", "video_games", "books"],
    "qa": ["web", "news", "debate"]
}
```

---

## 2. Data Loading (`data/load_data.py`)

The C-SEO Bench dataset is already imported via HuggingFace `datasets`. 

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("parameterlab/c-seo-bench")

"""
Load C-SEO Bench dataset from HuggingFace.

The dataset has 6 partitions (domains):
- recommendation: retail, video_games, books
- qa: web, news, debate

Each example has:
- "query": the user query string
- "documents": list of document strings (typically 5-10 per query)

Output:
- A dict keyed by domain name, each value is a list of dicts:
  [
    {
      "query_id": int,
      "query": str,
      "documents": [
        {"doc_id": int, "text": str, "original_text": str}
      ]
    }
  ]

Steps:
1. Load each partition from the HuggingFace dataset.
2. Sample 100 queries per domain (use a fixed random seed for reproducibility).
3. For each query, number the documents 0 to N-1.
4. Store "original_text" as a frozen copy that never changes (used for quality evaluation).
5. Store "text" as the mutable copy that gets rewritten each round.
6. Assign each document a fixed optimization_probability drawn from Beta(2, 5).
"""
```

**Key detail:** Inspect the actual dataset schema after loading. The column names may differ slightly across partitions (e.g., `"documents"` vs `"docs"` vs individual doc columns). Print the schema first and adapt accordingly.

---

## 3. Phase 1: Engine Ranking (`engine/ranker.py`)

```python
# engine/ranker.py

"""
Given a query and its documents, prompt the static LLM engine to produce
a forced complete ranking of all documents.

Input:
- query: str
- documents: list of {"doc_id": int, "text": str}
- model: str (e.g., "gpt-4o-mini")

Process:
1. Construct the prompt (see PROMPT TEMPLATES below).
2. Randomize document order in the prompt.
3. Call the OpenAI API.
4. Parse the response to extract the ranking.
5. Map ranks back to original doc_ids (undo the randomization).

Output:
- ranking: dict mapping doc_id -> rank (1 = best, N = worst)

Repeat N_RANDOMIZATIONS times per query per round.
Average the ranks across randomizations to get stable per-document ranks.
"""
```

### Prompt Templates

**For recommendation domains (retail, video_games, books):**

```
You are a product recommendation engine.

Here are product descriptions. Read ALL of them carefully before ranking.

[1] {doc_text_1}

[2] {doc_text_2}

...

[N] {doc_text_N}

User query: {query}

Rank ALL products from most recommended to least recommended.
You MUST include every number from 1 to {N} exactly once.
Format your response EXACTLY as:
1. [number] - one sentence reason
2. [number] - one sentence reason
...
{N}. [number] - one sentence reason
```

**For QA domains (web, news, debate):**

```
You are an information retrieval engine.

Here are documents. Read ALL of them carefully before ranking.

[1] {doc_text_1}

[2] {doc_text_2}

...

[N] {doc_text_N}

User query: {query}

Rank ALL documents from most relevant and helpful to least relevant.
You MUST include every number from 1 to {N} exactly once.
Format your response EXACTLY as:
1. [number] - one sentence reason
2. [number] - one sentence reason
...
{N}. [number] - one sentence reason
```

### Parsing Logic

```
Parse the response line by line.
For each line matching pattern: r"^\d+\.\s*\[(\d+)\]"
Extract the document number inside brackets.
The line's position (1st line = rank 1, 2nd = rank 2, etc.) gives the rank.
Map back to original doc_id using the randomization shuffle.

If parsing fails (LLM didn't follow format):
- Try more lenient regex: look for bracketed numbers in order of appearance.
- If still fails, flag this query-randomization as invalid and exclude from averaging.
- Log the failure for debugging.
```

### Randomization and Averaging

```python
"""
For each query in a given round:

1. Get the current document texts.
2. For r in range(N_RANDOMIZATIONS):
   a. Generate a random permutation of doc indices.
   b. Construct prompt with documents in permuted order.
   c. Call engine, parse ranking.
   d. Map ranks back to original doc_ids.
3. For each doc_id, average its rank across the valid randomizations.
4. Store: {doc_id: average_rank} for this query this round.
"""
```

### Batching

Use the OpenAI Batch API for cost efficiency (50% discount).

```python
"""
Collect all prompts for one round (100 queries × 5 randomizations = 500 calls per domain).
Write to a JSONL file.
Submit as a batch job.
Poll for completion.
Parse all responses.

Each batch request:
{
    "custom_id": "domain-{domain}_query-{qid}_rand-{rid}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 1024,
        "temperature": 0.0
    }
}

Use temperature=0.0 for deterministic ranking given the same prompt.
Randomization of document order (not temperature) provides the variance.
"""
```

---

## 4. Phase 2A-B: Feature Extraction (`features/extractor.py`)

```python
# features/extractor.py

"""
Compute ~25 features for each document.
All features are computed locally — no LLM API calls.

Input:
- document text (str)
- query text (str) — needed for query-relevance features

Output:
- feature_vector: dict of {feature_name: float}

STRUCTURAL FEATURES (~8):
1. word_count: total words
2. sentence_count: total sentences (split by ., !, ?)
3. avg_sentence_length: word_count / sentence_count
4. paragraph_count: split by double newline
5. heading_density: count of lines that look like headers (short, no period, possibly caps/bold markers) per 500 words
6. list_frequency: count of lines starting with -, *, •, or \d+\. per 500 words
7. readability: Flesch-Kincaid grade level (use textstat library)
8. bold_emphasis_density: count of **text** or similar markers per 500 words

EVIDENTIARY FEATURES (~8):
9. citation_density: count of patterns like [1], [source], (Author, Year), "according to" per paragraph
10. statistic_density: count of numbers, percentages (%), dollar amounts per 300 words
11. quote_density: count of quoted strings ("...") per 500 words
12. named_source_mentions: count of capitalized multi-word sequences that look like organization/person names (use spaCy NER: ORG + PERSON entities)
13. year_mentions: count of 4-digit numbers between 1990-2030 (recency signals)
14. claim_density: count of sentences containing a factual assertion. Proxy: sentences containing numbers, comparisons ("more than", "better", "fastest"), or superlatives.
15. external_reference_density: count of URL-like patterns or "http" per 500 words
16. question_density: count of sentences ending with ? per 500 words

SEMANTIC FEATURES (~9):
17. query_similarity: cosine similarity between document embedding and query embedding (use sentence-transformers, model: all-MiniLM-L6-v2 or BGE-small for speed)
18. corpus_centroid_similarity: cosine similarity between document embedding and mean embedding of all documents in this domain. High = generic/average. Low = distinctive.
19. type_token_ratio: unique words / total words
20. vocabulary_sophistication: fraction of words NOT in the 5000 most common English words (use a frequency list)
21. sentiment_polarity: use TextBlob or similar. Range -1 to 1.
22. avg_word_length: average characters per word (proxy for technical vocabulary)
23. semantic_uniqueness: 1 - max cosine similarity to any other document in the same query group. High = this document is unlike its competitors.
24. information_density: ratio of content words (nouns, verbs, adjectives) to total words (use spaCy POS tagging)
25. specificity_score: count of named entities (spaCy NER: all types) per 100 words. More entities = more specific.

Dependencies:
- spacy (python -m spacy download en_core_web_sm)
- textstat
- textblob
- sentence-transformers (for embeddings)
- numpy, scipy

Performance note:
- Embedding computation is the bottleneck. Batch all documents and compute embeddings once per round using sentence-transformers with GPU if available.
- spaCy NER and POS tagging can be batched with nlp.pipe().
- All other features are regex/string operations — very fast.
"""
```

---

## 5. Phase 2C: Discriminative Feature Analysis (`features/discriminator.py`)

```python
# features/discriminator.py

"""
Given feature vectors for all documents and their average ranks from Phase 1,
fit a logistic regression to identify which features discriminate top-ranked from bottom-ranked.

Input:
- feature_matrix: numpy array, shape (N_docs, 25)
- average_ranks: numpy array, shape (N_docs,) — average rank per document across all queries it appears in
- top_k_fraction: float (0.1 = top 10%)

Process:
1. Compute label for each document:
   - Aggregate each document's average rank across all queries it appears in.
   - Label top-K fraction as 1 (winner), rest as 0.
2. Standardize features (zero mean, unit variance). Store the scaler for later use.
3. Fit LogisticRegression(penalty='l2', C=1.0) on (feature_matrix, labels).
4. Record:
   - coefficients: array of shape (25,) — the importance of each feature
   - AUC: sklearn.metrics.roc_auc_score on the training data (this is intentional —
     we're measuring discriminability of the current corpus, not predicting unseen data)
   - top_features: indices of top TOP_N_FEATURES features by absolute coefficient

Output:
- top_feature_names: list of str (e.g., ["citation_density", "statistic_density", "query_similarity"])
- top_feature_targets: for each top feature, the mean value among top-K documents
- current_feature_values: for each non-top-K document, its current value on each top feature
- classifier_auc: float
- all_coefficients: dict {feature_name: coefficient} — for the heatmap figure
"""
```

---

## 6. Phase 2D: Targeted Rewriting (`imitation/rewriter.py`)

```python
# imitation/rewriter.py

"""
Rewrite non-top-K documents to move their features toward the top-K targets.

Input:
- document: {"doc_id": int, "text": str, "optimization_probability": float}
- top_feature_names: list of str
- top_feature_targets: dict {feature_name: target_value}
- current_feature_values: dict {feature_name: current_value}

Process:
1. Roll a random number. If > optimization_probability, SKIP this document (keep current text).
2. Otherwise, construct the rewriting prompt (see below).
3. Call REWRITER_MODEL (gpt-4o-mini).
4. Replace document["text"] with the rewritten text.

Output:
- Updated document text (or unchanged if skipped).
"""
```

### Rewriting Prompt Template

```
You are a content editor. Your task is to improve this document
to better match the following quality targets, while preserving
all factual information.

TARGETS:
- {feature_1_name}: move from {current_value} to {target_value}
  {feature_1_instruction}
- {feature_2_name}: move from {current_value} to {target_value}
  {feature_2_instruction}
- {feature_3_name}: move from {current_value} to {target_value}
  {feature_3_instruction}

RULES:
- Preserve all factual claims from the original document.
- Do not invent new facts, statistics, or quotes.
- Keep approximately the same document length.
- Write naturally — the result should read like polished web content.

DOCUMENT:
{document_text}

Rewrite the document below:
```

### Feature-to-Instruction Mapping

```python
"""
Map each feature name to a human-readable editing instruction.
This dict translates numerical targets into actionable directions.

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

For features not in this dict, use a generic instruction:
"Adjust {feature_name} from {current} toward {target}."
"""
```

### Batching

```
Collect all rewriting prompts for one round.
Submit as a batch job via OpenAI Batch API (same pattern as Phase 1).
Parse responses and update document texts.
```

---

## 7. Phase 3: Quality Evaluation (`evaluation/quality.py`)

**Runs only at EVALUATION_ROUNDS = [0, 5, 10, 15, 20, 25, 30].**

### Step 1: Generate Natural Responses

For each query, prompt the engine to generate a free-form response (not forced ranking):

**For recommendation:**
```
Here are product descriptions.

[1] {doc_1}
...
[N] {doc_N}

User query: {query}

Recommend the best products and explain why each is a good choice.
Cite documents by their number [1], [2], etc.
```

**For QA:**
```
Here are documents.

[1] {doc_1}
...
[N] {doc_N}

User query: {query}

Answer the question thoroughly using the documents.
Cite documents by their number [1], [2], etc.
```

### Step 2: Quality Metrics

```python
# evaluation/metrics.py

"""
Compute all metrics for a given round.

Input:
- rankings: per-query per-document average ranks from Phase 1
- feature_data: per-document feature vectors from Phase 2B
- classifier_data: coefficients + AUC from Phase 2C
- natural_responses: free-form responses from Phase 3 Step 1 (only at eval rounds)
- original_documents: the frozen round-0 texts

METRICS COMPUTED EVERY ROUND (from Phase 1 + Phase 2 data):

1. content_diversity:
   Compute pairwise cosine similarity between all document embeddings within each query group.
   Report mean intra-group similarity. Rising = homogenizing.

2. classifier_auc:
   Direct from Phase 2C output. Dropping = features becoming uninformative.

3. ranking_stability (Kendall's tau):
   For each query, compute Kendall's tau between all pairs of the 5 randomized rankings.
   Average across pairs and queries. High = engine is confident. Low = engine is guessing.

4. feature_coefficients:
   Store the full coefficient vector from Phase 2C. This builds the heatmap over rounds.

METRICS COMPUTED AT EVALUATION ROUNDS ONLY (from Phase 3 data):

5. goodhart_correlation (Spearman rho):
   Variable X: for each document, cosine similarity of its feature vector to the top-K mean feature vector.
   Variable Y: GPT-4o quality score for the document.
     - Prompt GPT-4o (JUDGE_MODEL):
       "Rate how well this document answers the query '{query}'.
        Score from 1-5 on: (a) factual accuracy, (b) completeness, (c) usefulness.
        Respond with ONLY three numbers separated by commas, e.g.: 4,3,5"
     - Average the three scores.
   Compute Spearman rho(X, Y) across all documents in the domain.

6. FOR QA DOMAINS:
   accuracy:
     If ground truth answer exists: compute token-level F1 between response and ground truth.
     If not: skip.
   
   completeness:
     At round 0, generate an aspect checklist per query:
       "For the query '{query}', list 5-8 key aspects that a complete answer should cover.
        Format: one aspect per line."
     At each eval round, check how many aspects the response covers:
       "Given this response and this checklist, which aspects are covered?
        Respond with the numbers of covered aspects."
     Report: fraction of aspects covered.

   source_diversity:
     Count distinct document numbers cited in the natural response. Report mean across queries.

7. FOR RECOMMENDATION DOMAINS:
   recommendation_diversity:
     For each query, extract the list of recommended product numbers from the natural response.
     Compute pairwise Jaccard similarity across all query pairs in the domain.
     Report mean Jaccard. Rising = all queries getting same recommendations.
   
   justification_distinctiveness:
     For each response, extract the justification text for each recommended product.
     Compute pairwise cosine similarity between justification texts within the same response.
     Report mean. Rising = justifications becoming generic/interchangeable.

   constraint_satisfaction:
     For each query, extract key constraints from the query text.
     Check if each recommended product (by its ORIGINAL round-0 description) matches those constraints.
     Prompt GPT-4o:
       "Query: {query}
        Original product description: {original_doc_text}
        Does this product match the query's requirements? Answer YES or NO with one sentence reason."
     Report: fraction of recommended products that match.
"""
```

---

## 8. Simulation Runner (`simulation/runner.py`)

```python
# simulation/runner.py

"""
Main orchestration loop.

Input:
- condition: one of ["adaptive_imitation", "fixed_geo", "no_optimization"]
- domain: one of the 6 domain names
- seed: int

Process:

1. SETUP
   - Load dataset for this domain, sample 100 queries.
   - Initialize documents with original texts.
   - Assign optimization_probability per document from Beta(BETA_A, BETA_B) using seed.
   - Create results storage: results[round][metric_name] = value

2. ROUND 0 (BASELINE)
   - Run Phase 1: forced ranking (5 randomizations per query).
   - Run Phase 2A-B: extract features for all documents.
   - Run Phase 2C: fit classifier, record AUC and coefficients.
   - Run Phase 3: generate natural responses, compute all quality metrics.
   - Store everything as round 0 baseline.
   - FOR QA: generate aspect checklists (used for all subsequent rounds).

3. FOR round in 1..N_ROUNDS:
   
   a. IF condition == "no_optimization":
      - Skip Phase 2D entirely. Documents never change.
   
   b. ELIF condition == "fixed_geo":
      - Skip Phase 2A-C (no feature analysis).
      - In Phase 2D, use a FIXED rewriting prompt every round (C-SEO Bench's
        "Content Improvement" prompt — see their paper's Appendix E):
        "Improve this document to make it more engaging, well-structured,
         and informative. Add clear headings, highlight key features,
         and make the information more accessible."
      - Apply with same heterogeneous probability as adaptive condition.
   
   c. ELIF condition == "adaptive_imitation":
      - Run Phase 2A-B: extract features.
      - Run Phase 2C: fit classifier, identify top discriminative features.
      - Run Phase 2D: rewrite non-top-K documents toward the top feature targets.
   
   d. Run Phase 1: forced ranking on the (possibly updated) corpus.
   
   e. Compute every-round metrics:
      - content_diversity
      - classifier_auc (skip for no_optimization and fixed_geo — no classifier)
      - ranking_stability (Kendall's tau)
      - feature_coefficients (for adaptive_imitation only)
   
   f. IF round in EVALUATION_ROUNDS:
      - Run Phase 3: natural responses + quality metrics.
   
   g. Save round results to disk (JSON or parquet).
      Save current document texts to disk (for reproducibility).

4. After all rounds complete, save full results to results/{condition}_{domain}_seed{seed}/
"""
```

### Execution Order for Full Experiment

```python
"""
main.py

Run all conditions × domains × seeds.

for seed in range(N_SEEDS):
    for domain in all_domains:
        for condition in ["no_optimization", "fixed_geo", "adaptive_imitation"]:
            run_simulation(condition, domain, seed)

Total runs: 3 seeds × 6 domains × 3 conditions = 54 runs.
Each run: 30 rounds.

Recommended execution order:
1. Run "no_optimization" for all domains first (fastest — no rewriting, just ranking).
   This gives you the baseline and sanity check immediately.
2. Run "fixed_geo" next.
3. Run "adaptive_imitation" last (most expensive — has feature extraction + targeted rewriting).

Use Batch API for all OpenAI calls. Each round's batch can be submitted and polled.
A round takes ~15-30 min of wall time (batch processing).
Full experiment: ~27-54 hours of wall time (can be parallelized across domains).
"""
```

---

## 9. Analysis and Plots (`analysis/plots.py`)

```python
# analysis/plots.py

"""
Load results from all runs and generate the paper's figures.

FIGURE 1: Goodhart Collapse (main result)
- 4-panel line plot, x-axis = rounds 0-30.
- 3 lines per panel: adaptive_imitation, fixed_geo, no_optimization.
- Panel (a): content_diversity (intra-corpus cosine similarity). AI rises, FG rises slowly, NO flat.
- Panel (b): ranking_stability (Kendall's tau). AI drops, FG drops slowly, NO flat.
- Panel (c): classifier_auc. AI drops to ~0.5, FG stays higher, NO not applicable.
- Panel (d): quality metric — completeness (QA) or constraint_satisfaction (recommendation).
- Show mean ± std across 3 seeds as shaded band.
- Pick one domain (Retail) for the main figure. Put others in appendix.

FIGURE 2: Feature Waterfall Heatmap
- Rows = 25 features, columns = rounds 1-30.
- Cell color = absolute logistic regression coefficient for that feature at that round.
- From adaptive_imitation condition only.
- Sort rows by the round at which the feature first drops below a threshold (e.g., 0.1).
  This should show surface features at top (saturate early) and semantic features at bottom (persist).

FIGURE 3: Cross-Domain Comparison
- 6 small panels (one per domain), each showing content_diversity over rounds.
- Only adaptive_imitation condition.
- Shows whether the Goodhart dynamic is universal or domain-specific.

FIGURE 4: Ranking Instability Deep Dive
- For one domain, one query: show the actual 10-document rankings from the 5 randomizations.
- Three columns: round 0, round 15, round 30.
- Each column is a small heatmap or parallel coordinates plot showing the 5 ranking permutations.
- At round 0: rankings are nearly identical across randomizations.
- At round 30: rankings are nearly random across randomizations.

TABLE 1: Summary
- Rows = 6 domains.
- Columns: Goodhart Threshold round (when AUC < 0.6), quality degradation (round 30 vs round 0),
  content diversity at round 30, ranking stability at round 30.
- Compare adaptive_imitation vs fixed_geo.

Use matplotlib + seaborn for all plots.
Use a consistent color scheme: adaptive_imitation = red, fixed_geo = blue, no_optimization = gray.
"""
```

---

## 10. Important Implementation Notes

### Rate Limiting and Cost Control

```
- Use OpenAI Batch API for ALL calls (50% cost reduction).
- Each batch can contain up to 50,000 requests.
- Per round per domain: ~500 ranking calls + ~70 rewriting calls = ~570 requests. Well within batch limits.
- Poll batch status every 60 seconds. Typical completion: 5-30 minutes.
- Log all costs using token counts from batch responses.
- Estimated total cost: ~$2,000 for the full 54-run experiment.
```

### Reproducibility

```
- Fix random seeds everywhere: numpy, random, document shuffling.
- Save the full document corpus after each round (text snapshots).
- Save all prompts and raw LLM responses (or at least the batch JSONL files).
- Save feature matrices and classifier weights per round.
- Use temperature=0.0 for all LLM calls.
```

### Error Handling

```
- LLM may not follow forced-ranking format perfectly.
  → Implement lenient parsing with fallback regex.
  → If parsing fails after fallback, log and skip that randomization.
  → Require at least 3 of 5 randomizations to be valid; if not, re-run the failed ones.

- Rewriter may significantly change document length or introduce hallucinated facts.
  → After rewriting, check: if new doc length differs from original by more than 50%, 
     clip or reject the rewrite and keep the original.
  → Note: we do NOT filter for factual accuracy of rewrites during the simulation.
     The point is to see whether the imitation process INTRODUCES inaccuracies.
     We measure accuracy degradation in Phase 3.

- Batch API may time out or return partial results.
  → Save intermediate results after each round.
  → Implement resume-from-round capability in the runner.
```

### Fixed GEO Prompt (from C-SEO Bench)

```
For the "fixed_geo" condition, use C-SEO Bench's "Content Improvement" method.
Their prompt (from their Appendix E) is approximately:

"Improve the following document to make it more engaging, well-structured, 
and informative for readers. Enhance the clarity, add clear headings or 
structure where appropriate, highlight key features or information, and 
make the content more accessible and compelling. Maintain all factual 
information from the original.

Document:
{document_text}

Improved version:"

Use this EXACT prompt every round, regardless of what the engine is currently ranking highly.
This is the control condition — it optimizes without observing the engine's behavior.
```

---

## 11. Quick Start Checklist

```
[ ] 1. Set OPENAI_API_KEY in environment
[ ] 2. Install dependencies: openai, datasets, numpy, scipy, sklearn, 
       spacy, textstat, textblob, sentence-transformers, matplotlib, seaborn, pandas
[ ] 3. Download spaCy model: python -m spacy download en_core_web_sm
[ ] 4. Run load_data.py — verify dataset loads correctly, print schema and sample
[ ] 5. Run Phase 1 on ONE query with ONE randomization — verify ranking parses correctly
[ ] 6. Run feature extractor on ONE document — verify all 25 features compute without error
[ ] 7. Run discriminator on round 0 data for ONE domain — verify AUC is reasonable (~0.6-0.8)
[ ] 8. Run rewriter on ONE document — verify output is reasonable
[ ] 9. Run full round 0 baseline for ONE domain — verify all metrics compute
[ ] 10. Run 5 rounds for ONE domain, adaptive_imitation — verify metrics change across rounds
[ ] 11. If step 10 looks good, launch full experiment
```
