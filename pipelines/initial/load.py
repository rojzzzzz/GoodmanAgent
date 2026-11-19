"""Initial load of the two validated V1 DataFrames."""

from contextlib import closing
import logging
from time import monotonic, sleep

import pandas as pd
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from config import DB_CREDENTIALS, SQL_TABLES
from pipelines.initial.transform import _read_contract

logger = logging.getLogger(__name__)
INSERT_BATCH_SIZE = 1000
RECONNECT_INTERVAL_SECONDS = 5
RECONNECT_WINDOW_SECONDS = 180


def initialize_database(*, require_empty: bool = False) -> None:
    """Apply the V1 schema and optionally check readiness for the initial load."""
    with psycopg.connect(**DB_CREDENTIALS, autocommit=True) as connection:
        connection.execute(SQL_TABLES.read_text(encoding="utf-8"))
        if require_empty and any(connection.execute("""
            SELECT EXISTS (SELECT 1 FROM legal_instruments),
                   EXISTS (SELECT 1 FROM document_nodes)
        """).fetchone()):
            raise ValueError("Initial loading requires empty legal_instruments and document_nodes tables")


def _record_batches(frame: pd.DataFrame):
    """Bound the number of Python row dictionaries created for SQL binding."""
    for start in range(0, len(frame), INSERT_BATCH_SIZE):
        yield frame.iloc[start:start + INSERT_BATCH_SIZE].to_dict("records")


def upload_to_postgres(
    instruments_df: pd.DataFrame, nodes_df: pd.DataFrame
) -> dict[str, int]:
    """Insert a validated batch into the existing schema in one transaction.

    This is an initial loader: existing instrument keys cause an error and
    rollback, rather than updating records. Importing this module does nothing.
    """

    tables = {"legal_instruments": instruments_df, "document_nodes": nodes_df}

    contract = _read_contract()["tables"]
    for k, v in tables.items():
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f"{k} must be a DataFrame")
        columns = set(v.columns) - ({"embedding"} if k == "document_nodes" else set())
        if not v.columns.is_unique or columns != set(contract[k]["columns"]):
            raise ValueError(f"{k} columns must match the data contract")

    if not instruments_df["source"].eq("digesto").fillna(False).all():
        raise ValueError("V1 supports only the digesto source")
    if instruments_df["source_id"].isna().any() or instruments_df["source_id"].duplicated().any():
        raise ValueError("Instrument source IDs must be present and unique")
    
    accepted_ids = instruments_df.loc[instruments_df["extraction_status"].eq("ok"), "source_id"]

    if not nodes_df["source_id"].isin(accepted_ids).all():
        raise ValueError("Each node must belong to an ok instrument in this batch")
    
    if not accepted_ids.isin(nodes_df["source_id"]).all():
        raise ValueError("Every ok instrument must have nodes")

    counts = {name: len(frame) for name, frame in tables.items()}
    
    if instruments_df.empty:
        return counts

    def parameters(row, fields):
        """Bind values in field order, converting pandas nulls, integers, and JSON."""
        values = []
        for column, field in fields.items():
            value = row[column]
            if pd.api.types.is_scalar(value) and pd.isna(value):
                value = None
            elif field["type"] == "integer":
                if isinstance(value, bool) or int(value) != value:
                    raise ValueError(f"{column} must contain whole numbers")
                value = int(value)
            elif field["type"] in ("array", "object"):
                value = Jsonb(value)
            values.append(value)
        return values

    instrument_fields = contract["legal_instruments"]["columns"]
    node_fields = {key: value for key, value in contract["document_nodes"]["columns"].items()
                   if key != "source_id"}

    def insert_statement(table, columns):
        """Build a parameterized INSERT with explicitly quoted table and column names."""
        return sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )

    insert_instrument = insert_statement("legal_instruments", instrument_fields) + sql.SQL(" RETURNING id")
    insert_node = insert_statement("document_nodes", ["legal_instrument_id", *node_fields, "embedding"])
    credentials = {"connect_timeout": 5, **DB_CREDENTIALS}
    deadline = None
    while True:
        commit_started = False
        try:
            with closing(psycopg.connect(**credentials)) as connection:
                if deadline is not None:
                    logger.info("PostgreSQL reconnected. Starting the upload transaction again.")
                # Cursors inherit the connection's adapters when they are created.
                register_vector(connection)
                with connection.cursor() as cursor:
                    instrument_ids = {}
                    for batch in _record_batches(instruments_df):
                        for row in batch:
                            cursor.execute(insert_instrument, parameters(row, instrument_fields))
                            instrument_ids[row["source_id"]] = cursor.fetchone()[0]
                    staged = 0
                    for batch in _record_batches(nodes_df):
                        cursor.executemany(insert_node, (
                            [instrument_ids[row["source_id"]], *parameters(row, node_fields),
                             Vector(row["embedding"]) if row.get("embedding") is not None else None]
                            for row in batch
                        ))
                        staged += len(batch)
                        if staged % (25 * INSERT_BATCH_SIZE) == 0 or staged == len(nodes_df):
                            logger.info("Staged %s/%s document nodes", staged, len(nodes_df))
                commit_started = True
                connection.commit()
            break
        except psycopg.OperationalError as error:
            state = error.sqlstate
            if state and not (state.startswith("08") or state in {"57P01", "57P02", "57P03"}):
                raise
            if commit_started:
                logger.error("Connection lost during commit. Check the target before retrying.")
                raise
            if deadline is None:
                deadline = monotonic() + RECONNECT_WINDOW_SECONDS
                logger.warning("PostgreSQL disconnected before commit. Retrying with the prepared embeddings for up to %s seconds.",
                               RECONNECT_WINDOW_SECONDS)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(RECONNECT_INTERVAL_SECONDS, remaining))

    logger.info("Loaded %s instruments and %s nodes", counts["legal_instruments"], counts["document_nodes"])
    return counts
