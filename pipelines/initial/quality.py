"""JSON-ready quality measurements for the two V1 DataFrames.

Field constraints are read from YAML. Named row rules use explicit Python
checks: contract prose is never evaluated as code. This module does not write
data, fix values, connect to PostgreSQL, or enforce database update policy.
"""

import base64
import binascii
import datetime
import hashlib
import json
import math
from numbers import Integral, Real
import re
from urllib.parse import parse_qs, urlsplit

import pandas as pd


def _missing(value):
    """Recognize missing pandas scalars without treating lists or dictionaries as null."""
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _integer(value):
    """Accept integral numbers and finite whole-number floats, excluding booleans."""
    if isinstance(value, bool):
        return False
    return isinstance(value, Integral) or (isinstance(value, Real) and math.isfinite(value) and int(value) == value)


def _date(value):
    """Check for a date value without a time component."""
    return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)


def _url(value, schemes, hosts=None):
    """Validate URL structure, allowed schemes and hosts, and forbidden whitespace."""
    if not isinstance(value, str) or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in schemes and bool(parsed.hostname) and (not hosts or parsed.hostname in hosts)
    except ValueError:
        return False


def _json(value):
    """Check JSON serializability with string object keys and valid UTF-8 text."""
    def string_keys(item):
        """Recursively require string dictionary keys inside JSON containers."""
        if isinstance(item, dict):
            return all(isinstance(key, str) and string_keys(child) for key, child in item.items())
        return not isinstance(item, list) or all(string_keys(child) for child in item)
    try:
        return string_keys(value) and bool(json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        return False


def _valid(value, field, contract):
    """Check a value against YAML field constraints; raise for unsupported types."""
    if _missing(value):
        return field["nullable"]
    kind = field["type"]
    valid_type = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: _integer(value),
        "date": lambda: _date(value),
        "timestamp": lambda: isinstance(value, datetime.datetime) and value.utcoffset() == datetime.timedelta(0),
        "array": lambda: isinstance(value, list) and _json(value),
        "object": lambda: isinstance(value, dict) and _json(value),
    }
    if kind not in valid_type:
        raise ValueError(f"Unsupported contract type: {kind}")
    if not valid_type[kind]():
        return False
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError:
            return False
        if (field.get("non_blank") or field.get("non_blank_when_present")) and not value.strip():
            return False
        if field.get("strip_whitespace_allowed") is False and value != value.strip():
            return False
        if "pattern" in field and re.fullmatch(field["pattern"], value) is None:
            return False
        if "allowed_schemes" in field and not _url(value, field["allowed_schemes"], field.get("allowed_hosts")):
            return False
    if "allowed_values" in field and value not in field["allowed_values"]:
        return False
    if kind == "integer" and not field.get("minimum", -math.inf) <= value <= field.get("maximum", math.inf):
        return False
    if kind == "array" and "items" in field:
        item_spec = field["items"]
        for item in value:
            if not isinstance(item, dict) or not set(item_spec["required_keys"]).issubset(item):
                return False
            if any(not isinstance(item[key], str) for key in item_spec["string_fields"]):
                return False
            if any(not item[key].strip() for key in item_spec["non_blank_fields"]):
                return False
            if not _url(item["url"], item_spec["url_allowed_schemes"]):
                return False
    return True


def _duplicates(records, columns):
    """Return positions of duplicate keys, skipping missing or unhashable key values."""
    groups = {}
    for position, row in enumerate(records):
        key = tuple(row.get(column) for column in columns)
        try:
            if any(_missing(value) for value in key):
                continue
            groups.setdefault(key, []).append(position)
        except TypeError:
            continue  # Column validity reports unhashable key values.
    return {position for group in groups.values() if len(group) > 1 for position in group}


def _check(records, failures, severity="error"):
    """Summarize failed row positions with severity and at most five identifier examples."""
    failures = sorted(set(failures))
    examples = []
    for position in failures[:5]:
        row = records[position]
        example = {"row_position": position}
        for column in ("source_id", "node_key"):
            value = row.get(column)
            example[column] = None if _missing(value) else str(value)[:100]
        examples.append(example)
    return {"severity": severity, "failed_rows": len(failures), "examples": examples,
            "status": ("failed" if severity == "error" else "warning") if failures else "passed"}


def _state_valid(row, spec):
    """Check required, null, exact, and minimum values for an extraction state."""
    for column in spec.get("null_fields", []):
        if not _missing(row.get(column)):
            return False
    for column in spec.get("required_fields", []):
        if _missing(row.get(column)):
            return False
    for column in spec.get("non_blank_fields", []):
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            return False
    for column, expected in spec.get("exact_values", {}).items():
        value = row.get(column)
        if expected is None:
            if not _missing(value):
                return False
        elif _missing(value) or not pd.api.types.is_scalar(value) or value != expected:
            return False
    minimum = spec.get("minimum_text_chars")
    return minimum is None or (_integer(row.get("text_chars")) and row["text_chars"] >= minimum)


def measure_table(df, contract, table_name=None):
    """Return JSON-ready schema, completeness, and contract checks for one DataFrame.

    Infer an unambiguous table name when omitted. Preserve input values and
    flag unchecked row rules rather than silently reporting them as passed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    column_count = len(df.columns)
    duplicate_columns = [str(column) for column in df.columns[df.columns.duplicated()]]
    df = df.loc[:, ~df.columns.duplicated()]
    if table_name is None:
        scores = {name: len(set(df.columns) & set(spec["columns"])) for name, spec in contract["tables"].items()}
        best = max(scores.values())
        candidates = [name for name, score in scores.items() if score == best]
        if not best or len(candidates) != 1:
            raise ValueError("Cannot identify the table; supply table_name")
        table_name = candidates[0]
    if table_name not in contract["tables"]:
        raise ValueError(f"Unknown contract table: {table_name}")
    spec = contract["tables"][table_name]
    records = df.to_dict("records")
    fields = spec["columns"]
    checks = {}
    metrics = {}

    def check(rule_id, failures, severity="error"):
        """Store a named rule result in the current table report."""
        checks[rule_id] = _check(records, failures, severity)

    for column, field in fields.items():
        values = [row.get(column) for row in records]
        nulls = sum(_missing(value) for value in values)
        blanks = sum(isinstance(value, str) and not value.strip() for value in values)
        failures = [i for i, value in enumerate(values) if not _valid(value, field, contract)]
        if column == "text":
            forbidden = [chr(int(code[2:], 16)) for code in contract["normalization"]["text"]["forbidden_characters"]]
            failures += [i for i, value in enumerate(values) if isinstance(value, str) and any(char in value for char in forbidden)]
        check(f"columns.{column}", failures)
        metrics[column] = {"dtype": str(df[column].dtype) if column in df else None,
                           "missing_count": nulls, "blank_count": blanks,
                           "completeness_pct": round(100 * (len(df) - nulls - blanks) / len(df), 2) if len(df) else None,
                           "invalid_count": len(set(failures))}

    identity_rule = "instrument_identity_unique" if table_name == "legal_instruments" else "node_identity_and_order_unique"
    duplicate_rows = _duplicates(records, spec["business_key"])
    if table_name == "document_nodes":
        duplicate_rows |= _duplicates(records, ["source_id", "position"])
    check(identity_rule, duplicate_rows)

    if table_name == "legal_instruments":
        bad_ids, bad_urls, bad_hashes, bad_lengths, bad_states = [], [], [], [], []
        for i, row in enumerate(records):
            try:
                decoded = base64.b64decode(row.get("source_id"), validate=True).decode("ascii")
                if not decoded.isdigit() or int(decoded) <= 0:
                    bad_ids.append(i)
            except (TypeError, ValueError, binascii.Error, UnicodeError):
                bad_ids.append(i)
            try:
                parsed = urlsplit(row.get("url"))
                if parsed.path != "/consultas/normas/shownorms.php" or parse_qs(parsed.query).get("idnorm") != [row.get("source_id")]:
                    bad_urls.append(i)
            except (TypeError, ValueError, AttributeError):
                bad_urls.append(i)
            status_value = row.get("extraction_status")
            state = spec["extraction_states"].get(status_value) if isinstance(status_value, str) else None
            if state is None or not _state_valid(row, state):
                bad_states.append(i)
            if isinstance(status_value, str) and status_value == "ok":
                text = row.get("text")
                try:
                    hash_matches = isinstance(text, str) and isinstance(row.get("text_sha256"), str) and row["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
                except UnicodeError:
                    hash_matches = False
                if not hash_matches:
                    bad_hashes.append(i)
                if not isinstance(text, str) or not _integer(row.get("text_chars")) or row["text_chars"] != len(text):
                    bad_lengths.append(i)
        check("source_id_base64", bad_ids)
        check("instrument_source_url_matches_id", bad_urls)
        check("instrument_content_hash", bad_hashes)
        check("instrument_content_length", bad_lengths)
        check("instrument_state_payload", bad_states)
        check("missing_optional_metadata", [i for i, row in enumerate(records) if any(
            _missing(row.get(key)) or row.get(key) == "" for key in ("fecha_aprobacion", "fecha_publicacion", "categoria", "rango", "numero"))], "warning")
        check("unexpected_date_order", [i for i, row in enumerate(records) if _date(row.get("fecha_aprobacion"))
              and _date(row.get("fecha_publicacion")) and row["fecha_aprobacion"] > row["fecha_publicacion"]], "warning")
    else:
        bykey = {(row["source_id"], row["node_key"]): row for row in records
                 if isinstance(row.get("source_id"), str) and isinstance(row.get("node_key"), str)}
        bad_spans, bad_parents, bad_tree, bad_tables, bad_json, self_parents = [], [], [], [], [], []
        for i, row in enumerate(records):
            start, end, text = row.get("start_char"), row.get("end_char"), row.get("text")
            if not (_integer(start) and _integer(end) and isinstance(text, str) and 0 <= start < end and len(text) == end - start):
                bad_spans.append(i)
            if not _valid(row.get("links"), fields["links"], contract) or not _valid(row.get("attributes"), fields["attributes"], contract):
                bad_json.append(i)
            parent = row.get("parent_node_key")
            if not _missing(parent):
                if isinstance(parent, str) and isinstance(row.get("node_key"), str) and parent == row["node_key"]:
                    self_parents.append(i)
                owner = row.get("source_id")
                ancestor = bykey.get((owner, parent)) if isinstance(owner, str) and isinstance(parent, str) else None
                if ancestor is None:
                    bad_parents.append(i)
                elif not (all(_integer(value) for value in (row.get("position"), ancestor.get("position"), start, end,
                                                            ancestor.get("start_char"), ancestor.get("end_char")))
                          and ancestor["position"] < row["position"] and ancestor["start_char"] <= start < end <= ancestor["end_char"]):
                    bad_tree.append(i)
            if isinstance(row.get("node_type"), str) and row["node_type"] == "table":
                attributes = row.get("attributes")
                rows = attributes.get("rows") if isinstance(attributes, dict) else None
                if not (isinstance(rows, list) and rows and all(isinstance(cells, list) and all(isinstance(cell, str) for cell in cells) for cells in rows)):
                    bad_tables.append(i)
        check("node_span_length", bad_spans)
        check("node_not_own_parent", self_parents)
        check("node_json_shape", bad_json)
        check("node_table_attributes", bad_tables)
        check("parent_in_same_instrument", bad_parents)
        check("tree_order_and_containment", bad_tree)
        groups = {}
        for i, row in enumerate(records):
            if isinstance(row.get("source_id"), str):
                groups.setdefault(row["source_id"], []).append(i)
        bad_order = []
        for positions in groups.values():
            values = [records[i].get("position") for i in positions]
            if not all(_integer(value) for value in values) or sorted(values) != list(range(len(values))):
                bad_order.extend(positions)
        check("node_positions_contiguous", bad_order)
        check("unrecognized_node_label", [i for i, row in enumerate(records) if _missing(row.get("label"))
              and isinstance(row.get("source_label"), str) and row["source_label"].strip()], "warning")
        check("source_structure_is_not_legal_meaning", [i for i, row in enumerate(records) if isinstance(row.get("node_type"), str) and row["node_type"] == "article"
              and (_missing(row.get("source_label")) or row.get("source_label") == "")], "warning")

    missing_columns = sorted(set(fields) - set(df.columns))
    extra_columns = sorted(str(column) for column in set(df.columns) - set(fields))
    unchecked = [rule["id"] for rule in spec["row_rules"] if rule["id"] not in checks]
    failed = bool(missing_columns or extra_columns or duplicate_columns or any(value["status"] == "failed" for value in checks.values()))
    status = "failed" if failed else "incomplete" if unchecked else "warning" if any(value["status"] == "warning" for value in checks.values()) else "passed"
    return {"table": table_name, "row_count": len(df), "column_count": column_count, "status": status,
            "schema": {"missing_columns": missing_columns, "unexpected_columns": extra_columns, "duplicate_columns": duplicate_columns},
            "columns": metrics, "checks": checks, "unchecked_rules": unchecked,
            "deferred_rules": [rule["id"] for rule in contract["cross_table_rules"] if rule["id"] not in checks],
            "value_counts": {column: {str(key): int(value) for key, value in df[column].map(
                lambda item: "<missing>" if _missing(item) else str(item)[:100]).value_counts().items()}
                             for column in ("extraction_status", "node_type") if column in df},
            "scope": "single_table; instrument/node ownership and text slices require the combined report"}


def measure_cross_tables(tables, contract, reports):
    """Check node ownership and text slices, reusing measured tree checks.

    Return an incomplete result when missing columns or ambiguous source IDs
    prevent comparisons between the two tables.
    """
    if any(not table.columns.is_unique for table in tables.values()):
        return {"status": "incomplete", "reason": "Duplicate columns prevent cross-table checks", "checks": {}}
    instruments = tables["legal_instruments"].to_dict("records")
    nodes = tables["document_nodes"].to_dict("records")
    required = {"legal_instruments": {"source_id", "extraction_status", "text"},
                "document_nodes": {"source_id", "start_char", "end_char", "text"}}
    if any(not columns.issubset(tables[name].columns) for name, columns in required.items()):
        return {"status": "incomplete", "reason": "Missing columns prevent cross-table checks", "checks": {}}
    if any(not isinstance(row.get("source_id"), str) for row in instruments + nodes) or _duplicates(instruments, ["source_id"]):
        return {"status": "incomplete", "reason": "Invalid/duplicate source IDs prevent unambiguous checks", "checks": {}}
    owners = {row["source_id"]: row for row in instruments}
    counts = {}
    missing_owners, bad_slices, rejected_nodes = [], [], []
    for i, node in enumerate(nodes):
        owner = owners.get(node["source_id"])
        counts[node["source_id"]] = counts.get(node["source_id"], 0) + 1
        if owner is None:
            missing_owners.append(i)
            continue
        if not isinstance(owner["extraction_status"], str) or owner["extraction_status"] != "ok":
            rejected_nodes.append(i)
        start, end, text = node["start_char"], node["end_char"], owner["text"]
        if not (_integer(start) and _integer(end) and isinstance(text, str) and isinstance(node["text"], str) and 0 <= start < end <= len(text)
                and node["text"] == text[int(start):int(end)]):
            bad_slices.append(i)
    bad_counts = []
    states = contract["tables"]["legal_instruments"]["extraction_states"]
    for i, row in enumerate(instruments):
        state = states.get(row["extraction_status"], {}) if isinstance(row["extraction_status"], str) else {}
        count = counts.get(row["source_id"], 0)
        if count < state.get("minimum_node_count", 0) or ("node_count" in state and count != state["node_count"]):
            bad_counts.append(i)
    checks = {"node_instrument_exists": _check(nodes, missing_owners),
              "node_matches_instrument_slice": _check(nodes, bad_slices),
              "nodes_only_for_accepted_content": {"status": "failed" if rejected_nodes or bad_counts else "passed",
                  "invalid_node_rows": len(rejected_nodes), "invalid_instrument_rows": len(bad_counts)}}
    for rule in ("parent_in_same_instrument", "tree_order_and_containment", "node_positions_contiguous"):
        checks[rule] = reports["document_nodes"]["checks"][rule]
    unchecked = [rule["id"] for rule in contract["cross_table_rules"] if rule["id"] not in checks]
    return {"status": "failed" if any(check["status"] == "failed" for check in checks.values()) else "incomplete" if unchecked else "passed",
            "checks": checks, "unchecked_rules": unchecked}
