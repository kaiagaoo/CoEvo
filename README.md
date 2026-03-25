# CoEvo

  ┌─────────────┬──────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │   Module    │       File       │                                                                                                    Purpose                                                                                                     │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ config.py   │ —                │ API keys, model names, hyperparameters (30 rounds, 100 queries/domain, 5 randomizations, 3 seeds, Beta(2,5) heterogeneity)                                                                                     │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ data/       │ load_data.py     │ Loads C-SEO Bench from HuggingFace, samples 100 queries/domain, assigns optimization_probability per doc from Beta(2,5), stores frozen original_text                                                           │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ engine/     │ ranker.py        │ Phase 1: forced LLM ranking with domain-specific prompts (recommendation vs QA), 5 randomized orderings per query, parsing with primary + fallback regex, averaging, Kendall's tau stability, OpenAI Batch API │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ features/   │ extractor.py     │ Phase 2A-B: computes all 25 features (8 structural, 8 evidentiary, 9 semantic) using spaCy, textstat, TextBlob, sentence-transformers                                                                          │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ features/   │ discriminator.py │ Phase 2C: L2 logistic regression on standardized features, identifies top-5 discriminative features, computes AUC and coefficient vector                                                                       │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ imitation/  │ rewriter.py      │ Phase 2D: targeted rewriting via Batch API — adaptive (feature-specific instructions) or fixed GEO (C-SEO Bench prompt), with heterogeneous opt probability and length validation                              │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ evaluation/ │ quality.py       │ Phase 3: natural response generation, GPT-4o quality scoring (accuracy/completeness/usefulness), aspect checklists, constraint satisfaction, source/recommendation diversity, justification distinctiveness    │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ evaluation/ │ metrics.py       │ Orchestrates every-round metrics (content diversity, AUC, Kendall's tau, coefficients) and eval-round metrics (Goodhart correlation, domain-specific quality)                                                  │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ simulation/ │ runner.py        │ Main loop: 3 conditions × 6 domains × 3 seeds × 30 rounds, with resume-from-round capability, per-round JSON snapshots                                                                                         │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ analysis/   │ plots.py         │ Generates Figures 1-4 + Table 1: Goodhart collapse panels, feature waterfall heatmap, cross-domain comparison, ranking instability, summary CSV                                                                │
  ├─────────────┼──────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ main.py     │ —                │ CLI entry point with --condition, --domain, --seed, --plots-only flags                                                                                                                                         │
  └─────────────┴──────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Key design decisions implemented:
  - All OpenAI calls use the Batch API (50% cost savings)
  - Resume capability — if a run crashes, it picks up from the last completed round
  - Heterogeneous creators — each doc has a Beta(2,5)-sampled probability of optimizing each round
  - Length guard — rewrites differing >50% in word count are rejected
  - Reproducibility — fixed seeds everywhere, temperature=0.0, document snapshots saved per round

  To run:
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  export OPENAI_API_KEY="your-key"
  python -m aice.main                    # full experiment (54 runs)
  python -m aice.main --domain retail --condition adaptive_imitation --seed 0  # single run
  python -m aice.main --plots-only       # generate figures from saved results