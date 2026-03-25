from dotenv import load_dotenv

load_dotenv()

# Models (OpenAI names used as keys; Gemini equivalents mapped in api_client.py)
ENGINE_MODEL = "gpt-4o-mini"       # -> gemini-2.5-flash-lite
REWRITER_MODEL = "gpt-4o-mini"     # -> gemini-2.5-flash-lite
JUDGE_MODEL = "gpt-4o-mini"        # -> gemini-2.5-flash-lite

# Experiment parameters
N_ROUNDS = 20
N_QUERIES_PER_DOMAIN = 100
N_RANDOMIZATIONS = 5
TOP_K_FRACTION = 0.1
TOP_N_FEATURES = 5
EVALUATION_ROUNDS = [0, 5, 10, 15, 20]
N_SEEDS = 3

# Creator heterogeneity
BETA_A = 2
BETA_B = 5

# Domains
DOMAINS = {
    "recommendation": ["retail", "video_games", "books"],
    "qa": ["web", "news", "debate"],
}

ALL_DOMAINS = []
for task_type, domain_list in DOMAINS.items():
    ALL_DOMAINS.extend(domain_list)

# Batch API polling
BATCH_POLL_INTERVAL = 60  # seconds

# Rewriting length tolerance
REWRITE_LENGTH_TOLERANCE = 0.5  # reject if length differs by more than 50%

# Minimum valid randomizations required per query
MIN_VALID_RANDOMIZATIONS = 3
