# CoEvo

## What This Is

AICE (Adaptive Imitation Co-Evolution) — a simulation experiment studying how documents evolve when creators optimize based on LLM ranking signals. Runs 20-round loops of rank → analyze → rewrite → re-rank across 6 domains × 3 seeds = 18 runs (adaptive_imitation condition only). 2 of 3 seeds completed so far.

## Setup & Commands

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Configure `.env`:
```
API_PROVIDER=gemini          # or "openai"
GEMINI_API_KEY=your-key      # if gemini
OPENAI_API_KEY=your-key      # if openai
GEMINI_MAX_WORKERS=5        # concurrent Gemini calls
```

```bash
python main.py                              # Run all 18 runs
python main.py --domain retail --seed 0     # Single run
python main.py --plots-only                 # Generate plots from all results
python main.py --plots-only --domain retail # Plots for one domain only
python main.py --verbose                    # Debug logging
```

No test suite, linter, or build system configured.

## Architecture

The simulation runs in 4 phases per round, orchestrated by `simulation/runner.py`:

```
Phase 1 (engine/ranker.py)          → Forced LLM ranking with 5 randomized orderings per query
Phase 2A-B (features/extractor.py)  → Compute 25 features (structural/evidentiary/semantic) per doc
Phase 2C (features/discriminator.py)→ L2 logistic regression: top-10% vs rest → top 5 features
Phase 2D (imitation/rewriter.py)    → Rewrite non-winner docs toward winner feature targets
Phase 3 (evaluation/quality.py)     → Quality scoring + natural responses (eval rounds only: 0,5,10,15,20)
```

Only the `adaptive_imitation` condition is run (`main.py` hardcodes it). The runner still supports `fixed_geo` and `no_optimization` internally if called directly.

## API Client (`api_client.py`)

`submit_batch(requests, tag)` is the single interface used by all LLM consumers (ranker, rewriter, quality). Both providers return OpenAI-shaped dicts: `{"choices": [{"message": {"content": "..."}}]}`.

- **Gemini mode**: concurrent calls via ThreadPoolExecutor with retry + exponential backoff on 429/RESOURCE_EXHAUSTED (5 retries, 2s base delay)
- **OpenAI mode**: async Batch API with 24h completion window

Model mapping: `gpt-4o-mini` → `gemini-3-flash-preview`, `gpt-4o` → `gemini-3-flash-preview`. Currently all three roles (ENGINE, REWRITER, JUDGE) use `gpt-4o-mini` to stay within a single Gemini quota bucket.

## Resilience

- **Atomic writes**: round snapshots write to `.tmp` then `os.replace()` to final path
- **Resume**: `_find_resume_round()` validates both metrics and docs JSON files parse correctly; corrupt rounds are re-run automatically
- **Rate limit retries**: Gemini client retries 429 errors with exponential backoff before giving up

## Domain Awareness

Domains split into two task types with different prompt templates and metrics:
- **Recommendation** (retail, video_games, books): constraint_satisfaction, recommendation_diversity, justification_distinctiveness
- **QA** (web, news, debate): completeness (aspect coverage), source_diversity; aspect checklists generated once at round 0

`engine/ranker.get_domain_type(domain)` is the canonical check used throughout.

## Data

`data/load_data.py` loads C-SEO Bench from HuggingFace (`parameterlab/c-seo-bench`). Each row is one document with columns `[query_id, query, document]`; rows sharing a `query_id` are grouped into a query (5-10 docs each). Split name `videogames` maps to our domain `video_games` (see `SPLIT_MAP`). 100 queries sampled per domain; each doc stores mutable `text` and frozen `original_text`.

## What's Tracked in Git

Metrics JSON files in `results/` are tracked. Large doc snapshots (`*_docs.json`) and generated PDFs are gitignored.

## Key Findings (2 seeds mean, 6 domains, 20 rounds)

### Overall: Goodhart's Law is Domain-Dependent

The co-evolutionary loop improves classifier AUC in 5 of 6 domains, but actual quality (task-specific metric) degrades in 3 of 6 — evidence that optimizing for LLM ranking signals can backfire. Debate is the only domain where AUC declines.

| Domain | AUC Gain | Task Metric Change | Diversity Change | Ranking Stability (R20) |
|---|---|---|---|---|
| Retail | +10.2% | -0.0109 (degraded) | -6.7% (homogenized) | 0.513 |
| Video_Games | +10.5% | +0.0112 (improved) | +7.1% | 0.602 |
| Books | +8.9% | +0.0001 (stable) | 0.0% | 0.629 |
| Web | +0.7% | +0.0087 (improved) | +4.4% | 0.658 |
| News | +4.1% | -0.0140 (degraded) | +2.4% | 0.678 |
| Debate | -2.2% | -0.0081 (degraded) | +2.6% | 0.426 |

### Goodhart Signal (Retail, News, Debate)

These domains show task-metric degradation despite ranking optimization:
- **Retail**: Strongest Goodhart effect. Structural features (word_count, paragraph_count) dominate in both seeds. Content becomes formulaic — diversity drops 6.7% (the only domain with diversity loss). Both seeds agree (S0: -6.4%, S1: -7.0%).
- **News**: Query similarity coefficient (1.90–2.87) is 3–5x larger than any other domain's top feature. Over-optimization for relevance signals hurts completeness. Completeness decline is identical across seeds (-0.014 each).
- **Debate**: Only domain where AUC declines (-2.2%). Ranking stability is lowest (0.426) and Goodhart correlation goes negative (-0.095). Feature profiles diverge across seeds — no stable ranking signal exists.

### Stable Optimization (Video_Games, Books, Web)

These domains improve or maintain task-specific quality alongside ranking:
- Multiple features remain important (no single feature dominates)
- Feature diversity prevents gaming a single signal
- Books is the most reproducible domain — tightest cross-seed agreement on all metrics
- Video Games shows strong AUC gains but wide inter-seed variance (S0: +8.1%, S1: +12.9%)

### Temporal Dynamics

- **R0–R5**: Largest changes across all domains (rapid early adaptation)
- **R5–R10**: Changes slow, patterns stabilize
- **R10–R20**: Asymptotic behavior; most metrics plateau
- Retail diversity drops steeply in R0–R5 then plateaus at ~0.690

### Feature Specialization (Cross-Seed Consensus at R20)

Features appearing in both seeds' top 5 at Round 20:
- **Retail**: word_count (+), paragraph_count (−) — structural formatting
- **Video_Games**: query_similarity (+), avg_word_length (−) — relevance + simplicity
- **Books**: query_similarity (+), word_count (+) — relevance + length
- **Web**: type_token_ratio (−), information_density (−) — lexical simplicity preferred
- **News**: query_similarity (+), semantic_uniqueness (+) — relevance dominates overwhelmingly
- **Debate**: type_token_ratio (−) only shared feature — no cross-seed consensus

### Debate is Anomalous

Only domain where AUC declines. Lowest ranking stability (0.426). Negative Goodhart correlation (-0.095). Feature profiles flip sign between seeds (named_source_mentions: -0.40 in S0, +0.28 in S1). Persuasive content fundamentally resists LLM-based ranking standardization — confirmed across 2 seeds.
