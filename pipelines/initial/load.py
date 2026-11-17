"""Initial load of the two validated V1 DataFrames."""


from pipelines.initial.transform import _read_contract
from pipelines.initial.config import DB_CREDENTIALS
from psycopg.types.json import Jsonb
from contextlib import closing
from psycopg import sql
import pandas as pd
import logging
import psycopg

logger = logging.getLogger(__name__)

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
        if not v.columns.is_unique or set(v.columns) != set(contract[k]["columns"]):
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
    insert_node = insert_statement("document_nodes", ["legal_instrument_id", *node_fields])

    # The connection context commits on success and rolls back on any error,
    # including deferred parent-node constraints checked at commit.
    # Explicit closing also covers errors raised while committing deferred checks.
    with closing(psycopg.connect(**DB_CREDENTIALS)) as connection:
        with connection, connection.cursor() as cursor:
            instrument_ids = {}
            for row in instruments_df.to_dict("records"):
                cursor.execute(insert_instrument, parameters(row, instrument_fields))
                instrument_ids[row["source_id"]] = cursor.fetchone()[0]
            cursor.executemany(insert_node, [
                [instrument_ids[row["source_id"]], *parameters(row, node_fields)]
                for row in nodes_df.to_dict("records")
            ])

    logger.info("Loaded %s instruments and %s nodes", counts["legal_instruments"], counts["document_nodes"])
    return counts
