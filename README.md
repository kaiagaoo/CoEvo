# CoEvo

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AICE (Adaptive Imitation Co-Evolution) — a simulation experiment studying how documents evolve when creators optimize based on LLM ranking signals. Runs 20-round loops of rank → analyze → rewrite → re-rank across 6 domains × 3 seeds = 18 runs (adaptive_imitation condition only).

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
