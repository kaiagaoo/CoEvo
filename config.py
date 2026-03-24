import os

from dotenv import load_dotenv

load_dotenv()

# API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Models
ENGINE_MODEL = "gpt-4o-mini"
REWRITER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"

# Experiment parameters
N_ROUNDS = 30
N_QUERIES_PER_DOMAIN = 100
N_RANDOMIZATIONS = 5
TOP_K_FRACTION = 0.1
TOP_N_FEATURES = 5
EVALUATION_ROUNDS = [0, 5, 10, 15, 20, 25, 30]
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

CONDITIONS = ["no_optimization", "fixed_geo", "adaptive_imitation"]

# Batch API polling
BATCH_POLL_INTERVAL = 60  # seconds

# Rewriting length tolerance
REWRITE_LENGTH_TOLERANCE = 0.5  # reject if length differs by more than 50%

# Minimum valid randomizations required per query
MIN_VALID_RANDOMIZATIONS = 3
