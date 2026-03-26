# CoEvo


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

## Key Findings (seed 0, 6 domains, 20 rounds)

### Overall: Goodhart's Law is Domain-Dependent

The co-evolutionary loop successfully improves classifier AUC in all domains, but actual quality degrades in 3 of 6 domains — evidence that optimizing for LLM ranking signals can backfire.

| Domain | AUC Gain | Quality Change | Diversity Change | Ranking Stability (R20) |
|---|---|---|---|---|
| Retail | +13% | -1.09% (degraded) | -6.5% (homogenized) | 0.516 |
| Video_Games | +8% | +1.89% (improved) | +3.3% | 0.624 |
| Books | +11% | +0.27% (improved) | stable | 0.642 |
| Web | +1% | +1.80% (improved) | +4.8% | 0.657 |
| News | +2% | -1.38% (degraded) | +2.8% | 0.684 |
| Debate | <1% | -0.85% (degraded) | +1.9% | 0.399 |

### Goodhart Signal (Retail, News, Debate)

These domains show quality degradation despite improved ranking performance:
- **Retail**: Strongest Goodhart effect. Structural features (paragraph_count -0.88, word_count +0.70) dominate. Content becomes formulaic — diversity drops 6.5% (the only domain with diversity loss).
- **News**: Query similarity coefficient (2.40) is 5x larger than any other domain's top feature. Over-optimization for relevance signals hurts holistic quality.
- **Debate**: Ranking stability declines from 0.44 → 0.40 (lowest of all domains). Persuasive content inherently resists standardized ranking — the system fails to find reliable signals.

### No Goodhart (Video_Games, Books, Web)

These domains improve quality alongside ranking:
- Multiple features remain important (no single feature dominates)
- Feature diversity prevents gaming a single signal
- Books shows the strongest balanced improvement: AUC +11% with stable diversity

### Temporal Dynamics

- **R0–R5**: Largest changes across all domains (rapid early adaptation)
- **R5–R10**: Changes slow, patterns stabilize
- **R10–R20**: Asymptotic behavior; most metrics plateau
- Retail diversity drops steeply in R0–R5 then plateaus at 0.682

### Feature Specialization

Top discriminative features by domain (Round 20):
- **Retail**: paragraph_count (-0.88), word_count (+0.70), list_frequency (+0.54) — structural
- **Video_Games**: word_count (+1.26), citation_density (-0.89), avg_word_length (-0.83) — length-driven
- **Books**: query_similarity (+0.86), word_count (+0.75) — relevance + structure
- **Web**: type_token_ratio (-0.79), avg_sentence_length (-0.48) — lexical diversity penalized
- **News**: query_similarity (+2.40), named_source_mentions (-1.06) — relevance dominates
- **Debate**: avg_sentence_length (+0.48), readability (-0.46) — syntax complexity valued

### Debate is Anomalous

Lowest ranking stability (0.40), minimal AUC improvement (<1%), yet maintains diversity. Suggests persuasive/argumentative content fundamentally resists LLM-based ranking standardization. Reader disagreement may be inherent to the domain rather than a failure of optimization.
