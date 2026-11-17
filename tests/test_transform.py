"""Node transformation preserves document identity and the extracted tree."""

from pathlib import Path
import datetime
import json
import tempfile
import unittest

import pandas as pd

from pipelines.initial.digesto_document import extract_document
from pipelines.initial.transform import _build_tables, _create_node_df, extract_data


def node(key, parent=None, **extra):
    return {"id": key, "parent_id": parent, "type": "paragraph", "label": None,
            "text": "Text", "start_char": 0, "end_char": 4,
            "source_locator": "div:nth-of-type(1)", **extra}


class CreateNodeDataFrameTests(unittest.TestCase):
    def test_document_identity_local_keys_order_and_input_are_preserved(self):
        frame = pd.DataFrame([
            {"iunpid": "LAW_A", "status": "ok", "document_nodes": [
                node("n000001", type="article", label="1", source_label="Art. 1"),
                node("n000002", "n000001")]},
            {"iunpid": "LAW_B", "status": "ok", "document_nodes": [node("n000001")]},
        ])
        result = _create_node_df(frame)
        self.assertEqual(result["source_id"].tolist(), ["LAW_A", "LAW_A", "LAW_B"])
        self.assertEqual(result["node_key"].tolist(), ["n000001", "n000002", "n000001"])
        self.assertEqual(result["position"].tolist(), [0, 1, 0])
        self.assertEqual(result.iloc[1]["parent_node_key"], "n000001")
        self.assertEqual(result.iloc[0]["source_label"], "Art. 1")
        self.assertTrue(pd.isna(result.iloc[1]["label"]))
        self.assertEqual(frame.iloc[0]["document_nodes"][0]["id"], "n000001")
        self.assertNotIn("source_id", frame.columns)

    def test_only_ok_documents_produce_nodes(self):
        frame = pd.DataFrame([
            {"iunpid": "A", "status": "ok", "document_nodes": [node("n1")]},
            {"iunpid": "B", "status": "empty_response", "document_nodes": []},
            {"iunpid": "C", "status": "needs_review", "document_nodes": [node("n2")]},
        ])
        result = _create_node_df(frame)
        self.assertEqual(result["source_id"].tolist(), ["A"])

    def test_empty_inputs_keep_the_same_output_columns(self):
        populated = _create_node_df(pd.DataFrame([
            {"iunpid": "A", "status": "ok", "document_nodes": [node("n1")]}
        ]))
        for frame in (pd.DataFrame(), pd.DataFrame([
                {"iunpid": "A", "status": "empty_response", "document_nodes": []}]),
                pd.DataFrame([{"iunpid": "A", "status": "ok", "document_nodes": []}])):
            with self.subTest(frame=frame):
                result = _create_node_df(frame)
                self.assertTrue(result.empty)
                self.assertEqual(result.columns.tolist(), populated.columns.tolist())

    def test_table_data_links_and_extra_source_attributes_survive(self):
        table = node("n1", type="table", rows=[["A", "5"]],
                     cells=[[{"text": "A", "colspan": "2"}]], source_internal_id="table",
                     links=[{"url": "https://example.invalid/law"}])
        result = _create_node_df(pd.DataFrame([
            {"iunpid": "A", "status": "ok", "document_nodes": [table]}]))
        row = result.iloc[0]
        self.assertEqual(row["attributes"]["rows"], [["A", "5"]])
        self.assertEqual(row["attributes"]["cells"], table["cells"])
        self.assertEqual(row["attributes"]["source_internal_id"], "table")
        self.assertEqual(row["links"], table["links"])

    def test_missing_or_malformed_accepted_nodes_do_not_disappear_silently(self):
        for value in (None, float("nan"), "not a list", ["not a node"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _create_node_df(pd.DataFrame([
                    {"iunpid": "A", "status": "ok", "document_nodes": value}]))

    def test_real_extractor_nodes_can_be_flattened(self):
        body = (Path(__file__).parent / "fixtures" / "digesto_3041.html").read_bytes()
        extracted = extract_document(body)
        result = _create_node_df(pd.DataFrame([
            {"iunpid": "MzA0MQ==", "status": extracted.status, "document_nodes": extracted.nodes}]))
        self.assertEqual(len(result), len(extracted.nodes))
        self.assertEqual(result.loc[result["node_type"].eq("article"), "label"].tolist(), ["1o", "2o", "3o"])
        for row in result.to_dict("records"):
            self.assertEqual(row["text"], extracted.text[row["start_char"]:row["end_char"]])


class BuildTablesTests(unittest.TestCase):
    def metadata(self, identifiers=("A", "B")):
        return pd.DataFrame([{
            "source_id": identifier, "categoria": "Ley", "numero": "1", "rango": "Ley",
            "titulo": f"Law {identifier}", "fecha_aprobacion": datetime.date(2015, 6, 4),
            "fecha_publicacion": datetime.date(2015, 10, 9),
            "url": f"https://example.invalid/{identifier}",
            "metadata_fetched_at": pd.Timestamp("2026-11-15T00:00:00Z"),
        } for identifier in identifiers])

    def text_data(self, identifiers=("A",), status="ok"):
        raw = pd.DataFrame([{"iunpid": identifier, "status": status,
                             "document_nodes": [node("n000001")] if status == "ok" else []}
                            for identifier in identifiers])
        documents = pd.DataFrame([{
            "source_id": identifier, "status": status,
            "text": "Text" if status == "ok" else "",
            "text_sha256": "a" * 64 if status == "ok" else None,
            "text_chars": 4 if status == "ok" else 0,
            "source_url": f"https://text.invalid/{identifier}",
            "text_fetched_at": pd.Timestamp("2026-11-15T01:00:00Z"),
            "status_reason": "Source result", "extractor_version": "3",
        } for identifier in identifiers])
        return {"documents": documents, "nodes": _create_node_df(raw)}

    def test_left_join_retains_dates_and_unattempted_instruments_without_mutation(self):
        metadata = self.metadata()
        text_data = self.text_data()
        before_metadata = metadata.copy(deep=True)
        before_documents = text_data["documents"].copy(deep=True)
        before_nodes = text_data["nodes"].copy(deep=True)
        tables = _build_tables(metadata, text_data)
        self.assertEqual(set(tables), {"legal_instruments", "document_nodes"})
        instruments = tables["legal_instruments"].set_index("source_id")
        self.assertEqual(instruments.index.tolist(), ["A", "B"])
        self.assertEqual(instruments.loc["A", "fecha_aprobacion"], datetime.date(2015, 6, 4))
        self.assertEqual(instruments.loc["A", "fecha_publicacion"], datetime.date(2015, 10, 9))
        self.assertEqual(instruments.loc["A", "url"], "https://example.invalid/A")
        self.assertEqual(instruments.loc["A", "extraction_status"], "ok")
        self.assertEqual(instruments.loc["A", "status_reason"], "Source result")
        self.assertEqual(instruments.loc["B", "extraction_status"], "not_attempted")
        self.assertTrue(pd.isna(instruments.loc["B", "text"]))
        self.assertTrue(pd.isna(instruments.loc["B", "text_fetched_at"]))
        self.assertEqual(instruments["source"].tolist(), ["digesto", "digesto"])
        self.assertNotIn("source_url", instruments.columns)
        pd.testing.assert_frame_equal(metadata, before_metadata)
        pd.testing.assert_frame_equal(text_data["documents"], before_documents)
        pd.testing.assert_frame_equal(text_data["nodes"], before_nodes)
        pd.testing.assert_frame_equal(tables["document_nodes"], before_nodes)

    def test_failed_extraction_is_preserved_without_node_rows(self):
        tables = _build_tables(self.metadata(("A",)), self.text_data(status="empty_response"))
        self.assertEqual(tables["legal_instruments"].iloc[0]["extraction_status"], "empty_response")
        self.assertEqual(tables["legal_instruments"].iloc[0]["text"], "")
        self.assertTrue(tables["document_nodes"].empty)

    def test_source_url_fills_an_empty_metadata_url(self):
        metadata = self.metadata(("A",))
        metadata.loc[0, "url"] = ""
        result = _build_tables(metadata, self.text_data())
        self.assertEqual(result["legal_instruments"].iloc[0]["url"], "https://text.invalid/A")

    def test_unknown_or_missing_status_is_rejected(self):
        for status in (None, "success"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                _build_tables(self.metadata(), self.text_data(status=status))

    def test_duplicate_instrument_or_text_ids_are_rejected(self):
        for metadata, data in ((self.metadata(("A", "A")), self.text_data()),
                               (self.metadata(), self.text_data(("A", "A")))):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                _build_tables(metadata, data)

    def test_missing_ids_are_rejected_instead_of_matching_null_rows(self):
        for value in (None, "", "   "):
            metadata = self.metadata(("A",))
            metadata.loc[0, "source_id"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                _build_tables(metadata, self.text_data())

    def test_orphan_documents_and_nodes_are_rejected(self):
        with self.assertRaises(ValueError):
            _build_tables(self.metadata(("B",)), self.text_data(("A",)))
        data = self.text_data()
        data["nodes"].loc[0, "source_id"] = "B"
        with self.assertRaises(ValueError):
            _build_tables(self.metadata(), data)

    def test_local_node_keys_can_repeat_across_instruments_but_not_within_one(self):
        data = self.text_data(("A", "B"))
        result = _build_tables(self.metadata(), data)
        self.assertEqual(result["document_nodes"]["node_key"].tolist(), ["n000001", "n000001"])
        data["nodes"] = pd.concat([data["nodes"], data["nodes"].iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            _build_tables(self.metadata(), data)

    def test_empty_inputs_keep_output_schema_and_metadata_only_is_supported(self):
        data = {"documents": pd.DataFrame(), "nodes": _create_node_df(pd.DataFrame())}
        empty = _build_tables(pd.DataFrame(), data)
        self.assertTrue(empty["legal_instruments"].empty)
        self.assertIn("fecha_aprobacion", empty["legal_instruments"].columns)
        metadata_only = _build_tables(self.metadata(), data)
        self.assertEqual(metadata_only["legal_instruments"]["extraction_status"].tolist(), ["not_attempted"] * 2)

    def test_extract_data_reads_given_directory_and_returns_both_tables(self):
        from pipelines.initial.config import METADATA_PATH, TEXT_PATH
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            metadata = {"iunpid": "MQ==", "categoria": "Ley", "numero": "1", "rango": "Ley",
                        "titulo": "Law A", "fechaAprobacion": "04/06/2015", "fechaPublicacion": "",
                        "url": "https://example.invalid/A", "metadata_fetched_at": "2026-11-15T00:00:00Z"}
            text = {"iunpid": "MQ==", "status": "ok", "status_reason": "Accepted", "text": "Text",
                    "text_sha256": "a" * 64, "text_chars": 4, "source_url": metadata["url"],
                    "fetched_at_utc": "2026-11-15T01:00:00Z", "extractor_version": "3",
                    "document_nodes": [node("n000001")]}
            (folder / Path(METADATA_PATH).name).write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            (folder / Path(TEXT_PATH).name).write_text(json.dumps(text) + "\n", encoding="utf-8")
            tables = extract_data(folder)
        instrument = tables["legal_instruments"].iloc[0]
        self.assertEqual(instrument["fecha_aprobacion"], datetime.date(2015, 6, 4))
        self.assertTrue(pd.isna(instrument["fecha_publicacion"]))
        self.assertEqual(instrument["status_reason"], "Accepted")
        self.assertEqual(len(tables["document_nodes"]), 1)

    def test_extract_data_supports_an_empty_text_snapshot(self):
        from pipelines.initial.config import METADATA_PATH, TEXT_PATH
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            metadata = {"iunpid": "MQ==", "categoria": "Ley", "numero": "1", "rango": "Ley",
                        "titulo": "Law A", "fechaAprobacion": "", "fechaPublicacion": "",
                        "url": "https://example.invalid/A", "metadata_fetched_at": "2026-11-15T00:00:00Z"}
            (folder / Path(METADATA_PATH).name).write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            (folder / Path(TEXT_PATH).name).write_text("", encoding="utf-8")
            tables = extract_data(folder)
        self.assertEqual(tables["legal_instruments"].iloc[0]["extraction_status"], "not_attempted")
        self.assertTrue(tables["document_nodes"].empty)


if __name__ == "__main__":
    unittest.main()
