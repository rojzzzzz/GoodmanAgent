"""Initial orchestration tests with local snapshots and mocked external services."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipelines.initial import config, extract, main as pipeline, transform
import test_digesto_extract


class InitialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Deliberately put artifacts in different directories to test path handoff.
        self.paths = {
            "OUTPUT_DIR": self.root / "reports",
            "METADATA_PATH": self.root / "catalog" / "custom_metadata.jsonl",
            "TEXT_PATH": self.root / "texts" / "custom_text.jsonl",
            "CACHE_DIR": self.root / "cache", "RAW_RESPONSE_DIR": self.root / "raw",
        }
        patcher = patch.multiple(config, **self.paths)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.instrument = {
            **test_digesto_extract.instrument(), "categoria": "Ley", "rango": "Ley",
            "fechaAprobacion": "04/06/2015", "fechaPublicacion": "09/10/2015",
        }

    def client(self):
        """Provide a fake API client serving one complete catalog and legal document."""
        patcher = patch.object(extract, "DigestoClient")
        client = patcher.start()
        self.addCleanup(patcher.stop)
        client.return_value.list_page.side_effect = [([self.instrument], 1), ([], 1)]
        client.return_value.main_text.return_value = test_digesto_extract.response()
        return client

    def test_offline_pipeline_writes_report_then_loads_both_real_frames(self):
        self.client()
        with patch.object(pipeline, "run_extraction", wraps=extract.run_extraction) as run, \
                patch.object(transform, "_measure_quality_dimensions", wraps=transform._measure_quality_dimensions) as measure, \
                patch.object(pipeline, "upload_to_postgres", return_value={"legal_instruments": 1, "document_nodes": 8}) as upload:
            result = pipeline.run_initial_pipeline(limit=1)
        run.assert_called_once_with(stage="all", limit=1)
        self.assertEqual(measure.call_count, 2)
        report_path = self.paths["OUTPUT_DIR"] / config.QUALITY_REPORT_FILENAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(list(self.paths["OUTPUT_DIR"].glob("*.json"))), 1)
        instruments, nodes = upload.call_args.args
        self.assertEqual(len(instruments), 1)
        self.assertGreater(len(nodes), 0)
        self.assertEqual(str(instruments.iloc[0]["fecha_aprobacion"]), "2015-06-04")
        self.assertEqual(result, upload.return_value)

    def test_actual_failed_quality_report_blocks_loading(self):
        self.instrument["titulo"] = " "
        self.client()
        with patch.object(pipeline, "upload_to_postgres") as upload:
            with self.assertRaisesRegex(ValueError, "Quality checks did not pass"):
                pipeline.run_initial_pipeline(limit=1)
        upload.assert_not_called()
        report = json.loads((self.paths["OUTPUT_DIR"] / config.QUALITY_REPORT_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_optional_metadata_warnings_allow_loading(self):
        self.instrument["fechaPublicacion"] = ""
        self.client()
        with patch.object(pipeline, "upload_to_postgres") as upload, self.assertLogs(pipeline.logger, level="WARNING"):
            pipeline.run_initial_pipeline(limit=1)
        upload.assert_called_once()

    def test_incomplete_catalog_or_missing_text_stops_before_transformation(self):
        for complete, text_path in ((False, Path("text.jsonl")), (None, Path("text.jsonl")), (True, None)):
            result = extract.ExtractionResult(Path("metadata.jsonl"), text_path, None, complete)
            with self.subTest(complete=complete, text_path=text_path), \
                    patch.object(pipeline, "run_extraction", return_value=result), \
                    patch.object(pipeline, "extract_data") as transform_call, \
                    patch.object(pipeline, "upload_to_postgres") as upload:
                with self.assertRaises(ValueError):
                    pipeline.run_initial_pipeline()
                transform_call.assert_not_called()
                upload.assert_not_called()

    def test_missing_or_incomplete_report_status_blocks_loading(self):
        result = extract.ExtractionResult(Path("metadata.jsonl"), Path("text.jsonl"), None, True)
        self.paths["OUTPUT_DIR"].mkdir()
        for status in ("incomplete", "unknown", None):
            report_path = self.paths["OUTPUT_DIR"] / config.QUALITY_REPORT_FILENAME
            report_path.write_text(json.dumps({"status": status}), encoding="utf-8")
            with self.subTest(status=status), patch.object(pipeline, "run_extraction", return_value=result), \
                    patch.object(pipeline, "extract_data", return_value={}), \
                    patch.object(pipeline, "upload_to_postgres") as upload:
                with self.assertRaisesRegex(ValueError, "Quality checks did not pass"):
                    pipeline.run_initial_pipeline()
                upload.assert_not_called()

    def test_cli_selects_full_or_limited_text_fetching(self):
        with patch.object(pipeline, "run_initial_pipeline") as run, patch.object(pipeline.logging, "basicConfig"):
            pipeline.main([])
            run.assert_called_once_with(limit=None)
            run.reset_mock()
            pipeline.main(["--limit", "25"])
            run.assert_called_once_with(limit=25)
            run.reset_mock()
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                pipeline.main(["--limit", "0"])
            run.assert_not_called()

    def test_import_does_not_start_extraction_or_database_calls(self):
        project = Path(__file__).resolve().parents[1]
        code = (
            "import sys; from unittest.mock import patch; "
            f"sys.path.insert(0, {str(project)!r}); "
            "import pipelines.initial.extract as e, pipelines.initial.load as l; "
            "a=patch.object(e, 'run_extraction'); b=patch.object(l, 'upload_to_postgres'); "
            "x=a.start(); y=b.start(); import pipelines.initial.main; "
            "x.assert_not_called(); y.assert_not_called()"
        )
        subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True, check=True)
        self.assertEqual(list(self.root.iterdir()), [])
