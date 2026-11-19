"""Contract-driven quality reporting without database or network access."""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pipelines.initial import transform
from config import METADATA_PATH, TEXT_PATH, QUALITY_REPORT_FILENAME


class QualityReportTests(unittest.TestCase):
    def tables(self):
        metadata = pd.DataFrame([{
            "source_id": "MQ==", "categoria": "Ley", "numero": "001", "rango": "Ley", "titulo": "Test law",
            "fecha_aprobacion": datetime.date(2015, 6, 4), "fecha_publicacion": datetime.date(2015, 10, 9),
            "url": "https://digesto.asamblea.gob.ni/consultas/normas/shownorms.php?idnorm=MQ%3D%3D",
            "metadata_fetched_at": pd.Timestamp("2026-11-15T00:00:00Z"),
        }])
        nodes = [{"id": "n1", "parent_id": None, "type": "article", "label": "1", "source_label": "Art. 1",
                  "text": "Text", "start_char": 0, "end_char": 4, "source_locator": "div:nth-of-type(1)", "links": []}]
        documents = pd.DataFrame([{
            "source_id": "MQ==", "text": "Text", "text_sha256": hashlib.sha256(b"Text").hexdigest(),
            "text_chars": 4, "source_url": metadata.iloc[0]["url"], "status": "ok", "status_reason": "Accepted",
            "text_fetched_at": pd.Timestamp("2026-11-15T01:00:00Z"), "extractor_version": "3",
        }])
        raw = pd.DataFrame([{"iunpid": "MQ==", "status": "ok", "document_nodes": nodes}])
        return transform._build_tables(metadata, {"documents": documents, "nodes": transform._create_node_df(raw)})

    def test_reports_are_json_ready_and_do_not_modify_the_input(self):
        for name, frame in self.tables().items():
            with self.subTest(table=name):
                before = frame.copy(deep=True)
                report = transform._measure_quality_dimensions(frame)
                self.assertEqual(report["table"], name)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["row_count"], 1)
                self.assertTrue(report["deferred_rules"])
                json.dumps(report, allow_nan=False)
                pd.testing.assert_frame_equal(frame, before)

    def test_missing_columns_required_values_wrong_types_and_duplicates_are_reported(self):
        frame = self.tables()["legal_instruments"].drop(columns="titulo")
        frame.loc[0, "metadata_fetched_at"] = pd.NaT
        frame["numero"] = 1
        frame = pd.concat([frame, frame], ignore_index=True)
        report = transform._measure_quality_dimensions(frame, table_name="legal_instruments")
        self.assertEqual(report["status"], "failed")
        self.assertIn("titulo", report["schema"]["missing_columns"])
        self.assertEqual(report["columns"]["metadata_fetched_at"]["invalid_count"], 2)
        self.assertEqual(report["columns"]["numero"]["invalid_count"], 2)
        self.assertEqual(report["checks"]["instrument_identity_unique"]["failed_rows"], 2)

    def test_yaml_field_constraints_are_used(self):
        frame = self.tables()["legal_instruments"]
        contract = copy.deepcopy(transform._read_contract())
        contract["tables"]["legal_instruments"]["columns"]["numero"]["pattern"] = "^[A-Z]+$"
        report = transform._measure_quality_dimensions(frame, contract=contract)
        self.assertEqual(report["columns"]["numero"]["invalid_count"], 1)

    def test_hash_length_and_extraction_state_failures_are_reported(self):
        frame = self.tables()["legal_instruments"]
        frame.loc[0, "text_sha256"] = "0" * 64
        frame.loc[0, "text_chars"] = 999
        frame.loc[0, "extractor_version"] = None
        report = transform._measure_quality_dimensions(frame)
        for rule in ("instrument_content_hash", "instrument_content_length", "instrument_state_payload"):
            self.assertEqual(report["checks"][rule]["status"], "failed")

    def test_missing_ok_hash_and_duplicate_columns_are_reported_without_crashing(self):
        frame = self.tables()["legal_instruments"]
        frame["text_sha256"] = pd.NA
        report = transform._measure_quality_dimensions(frame)
        self.assertEqual(report["checks"]["instrument_content_hash"]["status"], "failed")
        frame = pd.concat([frame, frame[["titulo"]]], axis=1)
        report = transform._measure_quality_dimensions(frame)
        self.assertEqual(report["schema"]["duplicate_columns"], ["titulo"])
        json.dumps(report, allow_nan=False)

    def test_failed_and_unattempted_payloads_are_valid_contract_states(self):
        contract = transform._read_contract()
        for status, state in contract["tables"]["legal_instruments"]["extraction_states"].items():
            if status == "ok":
                continue
            frame = self.tables()["legal_instruments"]
            frame["extraction_status"] = status
            if status == "not_attempted":
                for column in state["null_fields"]:
                    frame[column] = None
            else:
                frame["text"] = ""
                frame["text_sha256"] = None
                frame["text_chars"] = 0
                frame["status_reason"] = "No usable source text"
            with self.subTest(status=status):
                self.assertEqual(transform._measure_quality_dimensions(frame)["status"], "passed")

    def test_valid_missing_optional_metadata_and_unrecognized_labels_are_warnings(self):
        tables = self.tables()
        tables["legal_instruments"]["fecha_publicacion"] = None
        tables["document_nodes"]["label"] = None
        report = transform._measure_quality_dimensions(tables["legal_instruments"])
        self.assertEqual(report["status"], "warning")
        node_report = transform._measure_quality_dimensions(tables["document_nodes"])
        self.assertEqual(node_report["checks"]["unrecognized_node_label"]["status"], "warning")

    def test_parent_errors_invalid_json_and_malformed_nodes_are_reported(self):
        frame = self.tables()["document_nodes"]
        frame.loc[0, "parent_node_key"] = "missing"
        frame.at[0, "attributes"] = {"value": float("nan")}
        frame.loc[0, "end_char"] = 9
        report = transform._measure_quality_dimensions(frame)
        for rule in ("parent_in_same_instrument", "node_json_shape", "node_span_length"):
            self.assertEqual(report["checks"][rule]["status"], "failed")
        json.dumps(report, allow_nan=False)

    def test_nullable_scalars_and_invalid_enum_values_do_not_crash_reporting(self):
        for name, column in (("legal_instruments", "extraction_status"), ("document_nodes", "node_type")):
            frame = self.tables()[name]
            frame[column] = pd.NA
            report = transform._measure_quality_dimensions(frame, table_name=name)
            self.assertEqual(report["status"], "failed")
            json.dumps(report, allow_nan=False)

    def test_empty_tables_and_cross_checks_are_valid(self):
        tables = {name: frame.iloc[:0] for name, frame in self.tables().items()}
        with tempfile.TemporaryDirectory() as temporary:
            report = transform._write_quality_report(tables, Path(temporary) / "report.json")
        self.assertEqual(report["status"], "passed")
        self.assertIsNone(report["tables"]["legal_instruments"]["columns"]["titulo"]["completeness_pct"])

    def test_one_combined_file_and_one_measurement_per_table(self):
        tables = self.tables()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            with patch.object(transform, "_measure_quality_dimensions", wraps=transform._measure_quality_dimensions) as measure:
                with patch.object(transform, "_read_contract", wraps=transform._read_contract) as read:
                    report = transform._write_quality_report(tables, output)
            self.assertEqual(measure.call_count, 2)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(set(json.loads(output.read_text())["tables"]), set(tables))
            transform._write_quality_report(tables, output)
            self.assertEqual([path.name for path in Path(temporary).iterdir()], ["report.json"])

    def test_cross_table_ownership_and_text_slices_are_checked(self):
        tables = self.tables()
        tables["document_nodes"].loc[0, "text"] = "Else"
        with tempfile.TemporaryDirectory() as temporary:
            report = transform._write_quality_report(tables, Path(temporary) / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cross_table_checks"]["checks"]["node_matches_instrument_slice"]["status"], "failed")
        tables["document_nodes"].loc[0, "source_id"] = "Mg=="
        with tempfile.TemporaryDirectory() as temporary:
            report = transform._write_quality_report(tables, Path(temporary) / "report.json")
        self.assertEqual(report["cross_table_checks"]["checks"]["node_instrument_exists"]["status"], "failed")

    def test_accepted_instrument_without_nodes_fails_the_combined_check(self):
        tables = self.tables()
        tables["document_nodes"] = tables["document_nodes"].iloc[:0]
        with tempfile.TemporaryDirectory() as temporary:
            report = transform._write_quality_report(tables, Path(temporary) / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cross_table_checks"]["checks"]["nodes_only_for_accepted_content"]["invalid_instrument_rows"], 1)

    def test_extract_data_writes_the_combined_quality_file(self):
        tables = self.tables()
        instrument = tables["legal_instruments"].iloc[0]
        metadata = {"iunpid": "MQ==", "categoria": "Ley", "numero": "001", "rango": "Ley", "titulo": "Test",
                    "fechaAprobacion": "04/06/2015", "fechaPublicacion": "09/10/2015", "url": instrument["url"],
                    "metadata_fetched_at": "2026-11-15T00:00:00Z"}
        text = {"iunpid": "MQ==", "text": "Text", "text_sha256": instrument["text_sha256"], "text_chars": 4,
                "source_url": instrument["url"], "status": "ok", "status_reason": "Accepted",
                "fetched_at_utc": "2026-11-15T01:00:00Z", "extractor_version": "3", "document_nodes": [
                    {"id": "n1", "parent_id": None, "type": "article", "label": "1", "source_label": "Art. 1",
                     "text": "Text", "start_char": 0, "end_char": 4, "source_locator": "div", "links": []}]}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / Path(METADATA_PATH).name).write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            (folder / Path(TEXT_PATH).name).write_text(json.dumps(text) + "\n", encoding="utf-8")
            result = transform.extract_data(folder)
            report = json.loads((folder / QUALITY_REPORT_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(set(result), set(tables))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(result["legal_instruments"].iloc[0]["numero"], "001")
            self.assertEqual(result["legal_instruments"].iloc[0]["extractor_version"], "3")
            self.assertEqual(len(list(folder.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
