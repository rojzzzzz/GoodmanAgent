"""Run the initial extraction, transformation, quality gate, and PostgreSQL load.

From the project root: python -m pipelines.initial.main
Use --limit 25 to bound text fetching; the catalog is still collected in full.
The PostgreSQL schema must already exist. Embeddings are a separate stage.
"""

import argparse
import json
import logging
from pathlib import Path

from pipelines.initial import config
from pipelines.initial.extract import run_extraction
from pipelines.initial.load import upload_to_postgres
from pipelines.initial.transform import extract_data


logger = logging.getLogger(__name__)


def run_initial_pipeline(*, limit: int | None = None) -> dict[str, int]:
    """Extract current snapshots, validate both tables, and insert a valid batch.

    Paths and connection settings come from config.py. None fetches all eligible
    texts; an integer limits new text fetches, not the catalog or loaded snapshot.
    Warnings allow loading, but failed/incomplete checks stop before database
    writes. The insert-only loader rejects keys already present in PostgreSQL.
    """
    logger.info("Starting initial extraction")
    result = run_extraction(stage="all", limit=limit)
    if result.catalog_complete is not True:
        raise ValueError("Initial loading requires a complete, reconciled catalog")
    if result.text_path is None:
        raise ValueError("Extraction produced no text snapshot to transform")

    logger.info("Transforming snapshots and measuring quality")
    tables = extract_data(
        config.OUTPUT_DIR, metadata_path=result.metadata_path, text_path=result.text_path,
    )
    report_path = Path(config.OUTPUT_DIR) / config.QUALITY_REPORT_FILENAME
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    status = report.get("status") if isinstance(report, dict) else None
    if status not in ("passed", "warning"):
        raise ValueError(f"Quality checks did not pass ({status}); inspect {report_path}")
    if status == "warning":
        logger.warning("Quality warnings reported; see %s", report_path)

    logger.info("Loading both tables into PostgreSQL")
    counts = upload_to_postgres(tables["legal_instruments"], tables["document_nodes"])
    logger.info("Initial pipeline completed: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> dict[str, int]:
    """Parse the text-fetch limit, configure logging, and run the initial pipeline."""
    parser = argparse.ArgumentParser(description="Load the initial Digesto dataset into PostgreSQL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum new text fetches; omitted means all eligible documents")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return run_initial_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
