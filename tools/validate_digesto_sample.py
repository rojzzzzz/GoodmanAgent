"""Validate the saved learning sample offline; never fetch or change source data.

Run from the project with: python tools/validate_digesto_sample.py
Checks technical fidelity to the returned HTML, not completeness or legal force.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def content_characters(text):
    # Ignore layout whitespace and the cell separators inserted by the extractor.
    # This measures character retention, not correct word separation or typography.
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", text).replace("|", "")


def source_html(raw, encoding):
    source = raw.decode(encoding or "utf-8", errors="strict").strip()
    for _ in range(3):
        if not source or source[0] not in '{["':
            break
        data = json.loads(source)
        if isinstance(data, str):
            source = data.strip()
        elif isinstance(data, dict):
            source = next((data[key] for key in ("html", "content", "texto", "text")
                           if isinstance(data.get(key), str)), "").strip()
        else:
            source = ""
    return re.sub(r"^\s*<\?xml\b[^>]*\?>", "", source, count=1, flags=re.I | re.S)


def inspect_record(row, case):
    result = {"numeric_id": row["numeric_id"], "case": case["case"], "title": case["title"],
              "status": row["status"], "reason": row.get("status_reason"),
              "text_chars": len(row["text"]), "http_status": row.get("http_status"),
              "raw_file": row.get("raw_html_file"), "checks": {}}
    checks = result["checks"]
    raw = Path(row["raw_html_file"]).read_bytes() if row.get("raw_html_file") else b""
    checks["raw_hash_matches"] = bool(row.get("raw_sha256")) and hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
    checks["main_endpoint_only"] = row.get("request", {}).get("operation") == "getNormaHtmlAccordion"
    soup = BeautifulSoup(source_html(raw, row.get("encoding")), "html.parser")
    for tag in list(soup(["script", "style", "noscript", "head", "nav", "form", "button", "footer"])):
        tag.decompose()
    source_articles = [tag for tag in soup.select(".article") if tag.get_text(strip=True)]
    nodes = row.get("document_nodes", [])
    articles = [node for node in nodes if node["type"] == "article"]
    result.update(source_article_containers=len(source_articles), extracted_article_nodes=len(articles),
                  source_html_tables=len(soup.find_all("table")),
                  extracted_table_nodes=sum(node["type"] == "table" for node in nodes),
                  source_character_count=len(content_characters(soup.get_text())))
    result["source_preview"] = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:400]
    if row["status"] != "ok":
        checks["rejected_content_not_exported"] = not row["text"] and not nodes
        result["verdict"] = "needs_review" if row["status"] == "needs_review" else "unavailable"
        if not all(checks.values()):
            result["verdict"] = "failed"
        return result
    checks["text_hash_matches"] = hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"]
    checks["character_count_matches"] = len(row["text"]) == row["text_chars"]
    checks["no_replacement_characters"] = "\ufffd" not in row["text"]
    checks["source_content_characters_match"] = content_characters(soup.get_text()) == content_characters(row["text"])
    checks["node_ids_unique"] = len({node["id"] for node in nodes}) == len(nodes)
    checks["spans_match_text"] = all(0 <= node["start_char"] < node["end_char"] <= len(row["text"])
                                         and row["text"][node["start_char"]:node["end_char"]] == node["text"]
                                         for node in nodes)
    positions = {node["id"]: index for index, node in enumerate(nodes)}
    byid = {node["id"]: node for node in nodes}
    checks["parents_exist_and_contain_children"] = all(
        node.get("parent_id") is None or (
            node["parent_id"] in byid and positions[node["parent_id"]] < positions[node["id"]]
            and byid[node["parent_id"]]["start_char"] <= node["start_char"]
            and node["end_char"] <= byid[node["parent_id"]]["end_char"])
        for node in nodes)
    missing_labels = [node.get("source_label", "") for node in articles if not node.get("label")]
    result["article_nodes_without_normalized_label"] = len(missing_labels)
    result["unnormalized_source_labels"] = list(dict.fromkeys(missing_labels))[:10]
    if source_articles:
        checks["article_counts_match"] = len(source_articles) == len(articles)
        mismatches = [index + 1 for index, (source, node) in enumerate(zip(source_articles, articles))
                      if content_characters(source.get_text()) != content_characters(node["text"])]
        checks["article_content_boundaries_match"] = not mismatches and checks["article_counts_match"]
        result["article_boundary_mismatch_positions"] = mismatches[:20]
        # Spot-check DOM locators independently, including the beginning/middle/end.
        samples = sorted({0, len(articles) // 2, len(articles) - 1}) if articles else []
        checks["sampled_article_locators_match"] = all(
            soup.select_one(articles[index]["source_locator"]) is source_articles[index]
            for index in samples if index < len(source_articles))
    result["verdict"] = "passed" if all(checks.values()) else "failed"
    if result["verdict"] == "passed" and (missing_labels or not source_articles):
        result["verdict"] = "manual_semantic_review"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "learning/validation_text.jsonl")
    parser.add_argument("--selection", type=Path, default=ROOT / "learning/validation_selection.json")
    parser.add_argument("--report", type=Path, default=ROOT / "learning/validation_report.json")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    byid = {row["numeric_id"]: row for row in rows}
    results = [inspect_record(byid[case["numeric_id"]], case) for case in selection["selected"]
               if case["numeric_id"] in byid]
    missing = [case["numeric_id"] for case in selection["selected"] if case["numeric_id"] not in byid]
    unchanged = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == selection["original_dataset_hashes"][name]
                    for name in ("digesto_metadata.jsonl", "digesto_main_text.jsonl"))
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "sampling": selection["sampling"], "selected": len(selection["selected"]),
              "records": len(results), "missing_ids": missing,
              "duplicate_output_ids": len(rows) - len(byid), "original_datasets_unchanged": unchanged,
              "status_counts": dict(Counter(row["status"] for row in results)),
              "verdict_counts": dict(Counter(row["verdict"] for row in results)),
              "limits": ["A deliberate sample is not a population-wide accuracy estimate.",
                         "Content comparison ignores whitespace and inserted table separators.",
                         "Source article classes do not prove that a block is a numbered legal article.",
                         "Matching returned HTML does not prove completeness or current legal force."],
              "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=True))
    for result in results:
        print(json.dumps({"id": result["numeric_id"], "status": result["status"], "verdict": result["verdict"],
                          "source_articles": result["source_article_containers"],
                          "extracted_articles": result["extracted_article_nodes"],
                          "unlabeled_articles": result.get("article_nodes_without_normalized_label"),
                          "failed_checks": [key for key, value in result["checks"].items() if not value]}, ensure_ascii=True))
    if missing or not unchanged or len(rows) != len(byid) or any(row["verdict"] == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
