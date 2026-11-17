"""Extraction paths and defaults. Importing this file creates nothing.

Change OUTPUT_DIR to relocate all generated extraction data. Individual paths
can also be changed independently. Defaults preserve the existing datasets.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"

METADATA_PATH = OUTPUT_DIR / "digesto_metadata.jsonl"
TEXT_PATH = OUTPUT_DIR / "digesto_main_text.jsonl"
CACHE_DIR = OUTPUT_DIR / "digesto_cache"
RAW_RESPONSE_DIR = CACHE_DIR / "main_responses"
CONTRACT_PATH = PROJECT_DIR / "data_contract.yaml"
QUALITY_REPORT_FILENAME = "quality_report.json"

TEXT_BATCH_SIZE = 25
REQUEST_DELAY_SECONDS = 1.0
REQUEST_RETRIES = 2

#Postgres Config
DB_CREDENTIALS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "db",
    "user": "db_user",
    "password": "db_password"
}

SQL_TABLES = PROJECT_DIR / "database" / "schema.sql"
