"""Build article embedding inputs from the initial pipeline's DataFrames."""

from collections.abc import Mapping
import logging
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
NodeKey = tuple[str, str]
Node = Mapping[str, Any]


def _optional_text(value: Any) -> str:
    """Return nonblank metadata as text; represent pandas nulls as empty text."""
    return str(value).strip() if pd.notna(value) else ""

def _ancestor_source_label_chain(node: Node, nodes_by_key: Mapping[NodeKey, Node]) -> str:
    """Return heading source labels as ``root > ... > parent`` for one node.

    Follow parent_node_key within the same source_id, excluding the node itself.
    Skip non-heading ancestors and blank labels, but keep following their parents.
    Return an empty string when there are no labeled heading ancestors. Raise
    ValueError if a parent is missing or the ancestry contains a cycle.
    """
    labels = []
    source_id = node["source_id"]
    parent_key = node["parent_node_key"]
    visited = {node["node_key"]}
    while pd.notna(parent_key):
        if parent_key in visited:
            raise ValueError(f"Cyclic ancestry for node {node['node_key']} in {source_id}")
        visited.add(parent_key)
        parent = nodes_by_key.get((source_id, parent_key))
        if parent is None:
            raise ValueError(f"Missing parent {parent_key} in instrument {source_id}")
        if parent["node_type"] == "heading":
            source_label = _optional_text(parent.get("source_label"))
            if source_label:
                labels.append(source_label)
        parent_key = parent["parent_node_key"]
    return " > ".join(reversed(labels))

def build_embedding_inputs(
    instruments_df: pd.DataFrame, nodes_df: pd.DataFrame,
) -> dict[NodeKey, str]:
    """Build one embedding input string per article in an accepted instrument.

    Args:
        instruments_df: Transformed legal_instruments rows, keyed by source_id.
        nodes_df: Transformed document_nodes rows, including the articles and
            their ancestors. Node keys are local to each source_id.

    Returns:
        A dictionary mapping (source_id, node_key) to the complete input text.
        Each value contains the instrument designation, heading source labels
        from root to parent, article label (falling back to source_label), and
        the unchanged article text. Missing optional metadata is omitted.
        Only article nodes whose instrument has extraction_status == 'ok' are
        included. No eligible articles produces an empty dictionary.

    Raises:
        ValueError: Instrument IDs or instrument/node key pairs are duplicated,
            or an eligible article has a missing parent or cyclic ancestry.

    Neither DataFrame is modified. This function prepares text only; it does
    not call an embedding model or read/write PostgreSQL.
    """
    if instruments_df["source_id"].duplicated().any():
        raise ValueError("Instrument source IDs must be unique")
    if nodes_df.duplicated(["source_id", "node_key"]).any():
        raise ValueError("Node keys must be unique within each instrument")
    nodes = nodes_df.to_dict("records")
    nodes_by_key = {(node["source_id"], node["node_key"]): node for node in nodes}
    instruments = {
        row["source_id"]: row
        for row in instruments_df.loc[instruments_df["extraction_status"].eq("ok")].to_dict("records")
    }
    inputs: dict[NodeKey, str] = {}
    for node in nodes:
        instrument = instruments.get(node["source_id"])
        if node["node_type"] != "article" or instrument is None:
            continue
        ubication = _ancestor_source_label_chain(node, nodes_by_key)
        label = _optional_text(node["label"]) or _optional_text(node["source_label"])
        designation = _optional_text(instrument["rango"])
        number = _optional_text(instrument["numero"])
        if number:
            designation = f"{designation} N°. {number}".strip()
        title = _optional_text(instrument["titulo"])
        norma = " - ".join(part for part in (designation, title) if part)
        lines = [f"Norma: {norma}"] if norma else []
        if ubication:
            lines.append(f"Ubicacion: {ubication}")
        if label:
            lines.append(f"Articulo: {label}")
        lines.append(f"Contenido: {node['text']}")
        inputs[(node["source_id"], node["node_key"])] = "\n".join(lines)
    return inputs

def embedder(instruments_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of nodes_df with generated article vectors in 'embedding'.

    Match each API result to its input using the response index, then to its
    node using (source_id, node_key). Preserve row order and index; replace
    existing embeddings with float32 arrays, matching PostgreSQL's vector precision.
    Oversized articles and unselected rows get None.
    Neither input DataFrame is modified. No eligible articles means no API call.
    """
    inputs = build_embedding_inputs(instruments_df, nodes_df)
    result = nodes_df.copy()
    result["embedding"] = None
    if not inputs:
        return result

    from dotenv import load_dotenv
    from openai import OpenAI
    import tiktoken

    encoding = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)
    eligible = {}
    for key, text in inputs.items():
        tokens = len(encoding.encode(text, disallowed_special=()))
        if tokens > config.EMBEDDING_MAX_TOKENS:
            logger.warning("Holding article %s/%s: %s tokens exceeds %s; text preserved",
                           *key, tokens, config.EMBEDDING_MAX_TOKENS)
        else:
            eligible[key] = text
    if not eligible:
        return result

    load_dotenv(config.ROOT_DIR / ".env")
    keys = list(eligible)
    embeddings = {}
    with OpenAI() as client:
        for start in range(0, len(keys), config.EMBEDDING_BATCH_SIZE):
            batch = keys[start:start + config.EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                dimensions=config.EMBEDDING_DIMENSIONS,
                input=[eligible[key] for key in batch],
            )
            if sorted(item.index for item in response.data) != list(range(len(batch))):
                raise ValueError("Embedding response must contain exactly one vector per input index")
            for item in response.data:
                vector = np.asarray(item.embedding, dtype=np.float32)
                if vector.shape != (config.EMBEDDING_DIMENSIONS,) or not np.isfinite(vector).all():
                    raise ValueError("Embedding response contains an invalid vector")
                embeddings[batch[item.index]] = vector
            logger.info("Embedded %s/%s articles", min(start + len(batch), len(keys)), len(keys))

    result["embedding"] = [
        embeddings.get(key)
        for key in result[["source_id", "node_key"]].itertuples(index=False, name=None)
    ]
    return result
