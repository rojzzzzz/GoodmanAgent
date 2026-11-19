"""Extraction paths and defaults. Importing this file creates nothing.

Change OUTPUT_DIR to relocate all generated extraction data. Individual paths
can also be changed independently. Defaults preserve the existing datasets.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "pipelines" / "initial"
OUTPUT_DIR = PROJECT_DIR / "output"

METADATA_PATH = OUTPUT_DIR / "digesto_metadata.jsonl"
TEXT_PATH = OUTPUT_DIR / "digesto_main_text.jsonl"
CACHE_DIR = OUTPUT_DIR / "digesto_cache"
RAW_RESPONSE_DIR = CACHE_DIR / "main_responses"
CONTRACT_PATH = PROJECT_DIR / "data_contract.yaml"
QUALITY_REPORT_FILENAME = "quality_report.json"

TEXT_BATCH_SIZE = 25
REQUEST_DELAY_SECONDS = 1.0
MIN_REQUEST_DELAY_SECONDS = 0.25
REQUEST_RETRIES = 2
TEXT_REQUEST_CONCURRENCY = 4

#Postgres Config
DB_CREDENTIALS = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "db",
    "user": "db_user",
    "password": "db_password"
}

SQL_TABLES = ROOT_DIR / "database" / "schema.sql"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MAX_TOKENS = 8191

# At most 262,112 tokens per request, below the API's 300,000-token limit.
EMBEDDING_BATCH_SIZE = 32
K_NEAREST_NEIGHBORS = 20
