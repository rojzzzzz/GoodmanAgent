from pipelines.initial.config import METADATA_PATH, TEXT_PATH, CONTRACT_PATH, QUALITY_REPORT_FILENAME
from pipelines.initial.quality import measure_table, measure_cross_tables
from pathlib import Path
import pandas as pd
import datetime
import json
import os
import tempfile
import yaml


def _read_contract(path: Path | str | None = None) -> dict:
    """Read the YAML safely; relative paths are resolved by the caller."""
    with Path(CONTRACT_PATH if path is None else path).open(encoding="utf-8") as file:
        contract = yaml.safe_load(file)
    if not isinstance(contract, dict) or not isinstance(contract.get("tables"), dict):
        raise ValueError("The data contract must define a tables mapping")
    if set(contract["tables"]) != {"legal_instruments", "document_nodes"}:
        raise ValueError("The V1 contract must describe legal_instruments and document_nodes")
    return contract


def _measure_quality_dimensions(
    df: pd.DataFrame, *, table_name: str | None = None, contract: dict | None = None
) -> dict:
    """Return a JSON-ready table report without changing the DataFrame or writing files.

    Normal table shapes are recognized automatically. Supply table_name when
    profiling a DataFrame with missing/ambiguous columns. Combined checks and
    the single JSON file are handled once by extract_data().
    """
    return measure_table(df, _read_contract() if contract is None else contract, table_name)


def _write_quality_report(tables: dict[str, pd.DataFrame], path: Path) -> dict:
    """Measure each table once and atomically write one combined JSON report.

    Include cross-table checks and return the same report dictionary. Reported
    failures do not modify the DataFrames or raise validation exceptions.
    """
    contract = _read_contract()
    reports = {name: _measure_quality_dimensions(df, table_name=name, contract=contract)
               for name, df in tables.items()}
    cross = measure_cross_tables(tables, contract, reports)
    statuses = [report["status"] for report in reports.values()] + [cross["status"]]
    status = next((value for value in ("failed", "incomplete", "warning") if value in statuses), "passed")
    report = {"contract_version": contract["contract_version"],
              "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": status, "tables": reports, "cross_table_checks": cross,
              "scope": "transformed data quality; database refresh policy and legal currency are not evaluated"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + "-", suffix=".tmp", delete=False) as file:
            temp_path = Path(file.name)
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return report

def _parse_source_date(value) -> datetime.date | None:
    """Parse a source DD/MM/YYYY date, returning None for missing or blank values."""
    if pd.isna(value):
        return None

    value = value.strip()
    if not value:
        return None

    return datetime.datetime.strptime(value, "%d/%m/%Y").date()

def _create_node_df(text_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten accepted documents into node rows, keyed within each source ID."""
    columns = [
        "source_id", "node_key", "parent_node_key", "position", "node_type",
        "label", "source_label", "text", "start_char", "end_char",
        "source_locator", "links", "attributes",
    ]
    if text_df.empty:
        return pd.DataFrame(columns=columns)

    missing = {"iunpid", "status"} - set(text_df.columns)
    if missing:
        raise ValueError(f"Text data is missing required columns: {sorted(missing)}")

    required_node_fields = {"id", "type", "text", "start_char", "end_char", "source_locator"}
    mapped_node_fields = required_node_fields | {"parent_id", "label", "source_label", "links"}
    records = []
    for document in text_df.loc[text_df["status"].eq("ok")].to_dict("records"):
        source_id = document["iunpid"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("An accepted document has no usable iunpid")
        nodes = document.get("document_nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"Document {source_id}: document_nodes must be a list; re-extract legacy records")

        for position, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise ValueError(f"Document {source_id}, node {position}: expected a dictionary")
            missing = required_node_fields - node.keys()
            if missing:
                raise ValueError(f"Document {source_id}, node {position}: missing fields {sorted(missing)}")
            records.append({
                "source_id": source_id,
                "node_key": node["id"],
                "parent_node_key": node.get("parent_id"),
                "position": position,
                "node_type": node["type"],
                "label": node.get("label"),
                "source_label": node.get("source_label"),
                "text": node["text"],
                "start_char": node["start_char"],
                "end_char": node["end_char"],
                "source_locator": node["source_locator"],
                "links": node.get("links", []),
                "attributes": {key: value for key, value in node.items() if key not in mapped_node_fields},
            })

    return pd.DataFrame.from_records(records, columns=columns)

def _extract_text_jsonl(path: Path | str = TEXT_PATH) -> dict[str, pd.DataFrame]:
    """Read text JSONL into document and accepted-node DataFrames, preserving strings."""
    string_fields = ("iunpid", "text", "text_sha256", "source_url", "status", "status_reason", "extractor_version")
    df = pd.read_json(path, lines=True, dtype={column: "string" for column in string_fields})
    nodes_df = _create_node_df(df)
    df = df.reindex(columns=["iunpid", "text", "text_sha256", "text_chars", "source_url", "status",
                             "status_reason", "fetched_at_utc", "extractor_version"])
    df["fetched_at_utc"] = pd.to_datetime(df["fetched_at_utc"], errors="coerce")
    df = df.rename(columns={"iunpid": "source_id", "fetched_at_utc": "text_fetched_at"})
    return {"documents": df, "nodes": nodes_df}

def _extract_metadata_jsonl(path: Path | str = METADATA_PATH) -> pd.DataFrame:
    """Read metadata JSONL, parse source dates, and rename fields to contract names."""

    string_fields = ("iunpid", "categoria", "rango", "numero", "titulo", "fechaAprobacion", "fechaPublicacion", "url")
    df = pd.read_json(path, lines=True, dtype={column: "string" for column in string_fields})
    df = df[["iunpid", "categoria", "rango", "numero", "titulo", "fechaAprobacion", "fechaPublicacion",
             "url", "metadata_fetched_at"]]

    df["metadata_fetched_at"] = pd.to_datetime(df["metadata_fetched_at"], errors="coerce")

    for column in ("fechaAprobacion", "fechaPublicacion"):
        df[column] = df[column].map(_parse_source_date)

    return df.rename(columns={"iunpid": "source_id", "fechaAprobacion": "fecha_aprobacion", 
                              "fechaPublicacion": "fecha_publicacion"})

def _build_tables(
    metadata_df: pd.DataFrame, text_df: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Build V1 tables from metadata and the text reader's documents/nodes.

    Keep source_id for the loader to resolve database-generated instrument IDs;
    DataFrame indexes are not database IDs.
    """
    instrument_columns = [
        "source", "source_id", "categoria", "numero", "rango", "titulo",
        "fecha_aprobacion", "fecha_publicacion", "url", "metadata_fetched_at",
        "text", "text_sha256", "text_chars", "extraction_status", "status_reason",
        "text_fetched_at", "extractor_version",
    ]
    if not isinstance(text_df, dict) or not {"documents", "nodes"}.issubset(text_df):
        raise TypeError("text_df must contain the 'documents' and 'nodes' DataFrames")
    frames = {"metadata": metadata_df, "documents": text_df["documents"], "nodes": text_df["nodes"]}
    if not all(isinstance(frame, pd.DataFrame) for frame in frames.values()):
        raise TypeError("Metadata, documents and nodes must be DataFrames")
    metadata = metadata_df.copy()
    documents = text_df["documents"].copy()
    nodes_df = text_df["nodes"].copy()
    if documents.empty:
        documents = documents.reindex(columns=["source_id", "text", "text_sha256", "text_chars",
                                              "status", "status_reason", "source_url", "text_fetched_at",
                                              "extractor_version"])
    if nodes_df.empty:
        nodes_df = nodes_df.reindex(columns=_create_node_df(pd.DataFrame()).columns)

    for name, frame in (("metadata", metadata), ("documents", documents), ("nodes", nodes_df)):
        if "source_id" not in frame.columns and not frame.empty:
            raise ValueError(f"{name} is missing source_id")
        if frame.empty:
            frame["source_id"] = pd.Series(index=frame.index, dtype="str")
        if not frame.empty and not frame["source_id"].map(
                lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} contains missing or invalid source IDs")
        if name != "nodes" and frame["source_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate source IDs")

    statuses = {"ok", "empty_response", "no_usable_text", "needs_review", "http_error"}
    if "status" not in documents or not documents["status"].isin(statuses).all():
        raise ValueError("Documents contain missing or unknown extraction statuses")
    if not documents["source_id"].isin(metadata["source_id"]).all():
        raise ValueError("Text records refer to source IDs absent from metadata")
    accepted_ids = documents.loc[documents["status"].eq("ok"), "source_id"]
    if not nodes_df["source_id"].isin(accepted_ids).all():
        raise ValueError("Nodes must belong to an accepted document in the metadata")
    if "node_key" not in nodes_df or nodes_df.duplicated(["source_id", "node_key"]).any():
        raise ValueError("Node keys must be present and unique within each source ID")
    overlap = (set(metadata.columns) & set(documents.columns)) - {"source_id"}
    if overlap:
        raise ValueError(f"Metadata and text have ambiguous overlapping columns: {sorted(overlap)}")

    instruments_df = metadata.merge(documents, on="source_id", how="left", validate="one_to_one", sort=False)
    if "source_url" in instruments_df:
        if "url" in instruments_df:
            missing_url = instruments_df["url"].isna() | instruments_df["url"].eq("")
            instruments_df["url"] = instruments_df["url"].mask(missing_url, instruments_df["source_url"])
        else:
            instruments_df["url"] = instruments_df["source_url"]
        instruments_df = instruments_df.drop(columns="source_url")
    instruments_df = instruments_df.rename(columns={"status": "extraction_status"})
    instruments_df["extraction_status"] = instruments_df["extraction_status"].astype("string").fillna("not_attempted")
    if "source" not in instruments_df:
        instruments_df["source"] = "digesto"
    instruments_df = instruments_df.reindex(columns=instrument_columns)
    return {
        "legal_instruments": instruments_df,
        "document_nodes": nodes_df,
    }

def extract_data(
    output_path: Path | str, *, metadata_path: Path | str | None = None,
    text_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build both tables and write their quality report to output_path.

    Optional input paths select exact extraction artifacts. Otherwise read the
    configured JSONL filenames within output_path, preserving existing calls.
    """
    output_path = Path(output_path)
    metadata_df = _extract_metadata_jsonl(Path(metadata_path) if metadata_path is not None
                                        else output_path / Path(METADATA_PATH).name)
    text_df = _extract_text_jsonl(Path(text_path) if text_path is not None
                                else output_path / Path(TEXT_PATH).name)
    dfs = _build_tables(metadata_df, text_df)
    _write_quality_report(dfs, output_path / QUALITY_REPORT_FILENAME)
    return dfs


    

