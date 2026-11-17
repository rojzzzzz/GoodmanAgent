"""Offline regression cases for the V1 main-text extractor."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests

import pipelines.initial.config
import pipelines.initial.extract as extractor


LEGAL_HTML = """<html><body><h1>DECRETO DE PRUEBA</h1>
<h2>CAPÍTULO I</h2><p>Artículo <b>1</b>. Objeto</p>
<p>Se establece una obligación con <a href="/ley/2">Ley 2</a>.</p>
<p>Artículo 2. Excepciones</p><p>No será aplicable al supuesto indicado.</p>
<table><tr><th>Concepto</th><th>Valor</th></tr>
<tr><td>Tasa</td><td>5</td></tr></table></body></html>""".encode()


def response(body=LEGAL_HTML, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result.headers["Content-Type"] = "text/html; charset=utf-8"
    result.url = extractor.API
    return result


def instrument(number=1):
    return {"iunpid": base64.b64encode(str(number).encode()).decode(),
            "titulo": "Decreto de prueba", "numero": str(number)}


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def text_args(self, **changes):
        args = dict(metadata=self.root / "metadata.jsonl", output=self.root / "texts.jsonl",
                    raw_dir=self.root / "raw", delay=1.0, retries=0, all=True, limit=25,
                    retry_failed=False, refresh=False, ids=None)
        args.update(changes)
        return argparse.Namespace(**args)

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def parse(self, body, **kwargs):
        return extractor.extract_document(body, expected_title="Decreto de prueba",
                                          document_url=extractor.BASE, **kwargs)

    def test_error_and_navigation_pages_are_not_legal_text(self):
        for body in (b"<html><body>Error interno del servidor</body></html>",
                     b'{"error":"fallo","text":"Decreto de prueba"}',
                     b'<html><nav><a href="/">Inicio</a></nav><form>Buscar normas</form></html>',
                     b'{"documents":[{"name":"version"}]}'):
            with self.subTest(body=body):
                result = self.parse(body)
                self.assertNotEqual(result.status, "ok")
                self.assertEqual(result.text, "")

    def test_comments_and_news_mentions_do_not_establish_a_legal_document(self):
        for body in (b'<!DOCTYPE html><!-- Art. 1. Legal text --><p>Inicio</p>',
                     '<p>El Presidente de la República visitó ayer una escuela.</p>'.encode()):
            self.assertNotEqual(self.parse(body).status, "ok")

    def test_invalid_delay_is_rejected_before_network_access(self):
        with patch("sys.argv", ["digesto", "texts", "--delay", "nan"]):
            with patch.object(extractor, "texts") as operation:
                with self.assertRaises(SystemExit) as error:
                    extractor.main()
                self.assertEqual(error.exception.code, 2)
                operation.assert_not_called()

    def test_articles_inline_text_links_tables_and_spans(self):
        result = self.parse(LEGAL_HTML)
        self.assertEqual(result.status, "ok")
        self.assertIn("Artículo 1. Objeto", result.text)
        articles = [node for node in result.nodes if node["type"] == "article"]
        self.assertEqual([node["label"] for node in articles], ["1", "2"])
        self.assertIn("Se establece", articles[0]["text"])
        self.assertNotIn("Excepciones", articles[0]["text"])
        self.assertIn("No será aplicable", articles[1]["text"])
        self.assertTrue(any(node.get("rows") == [["Concepto", "Valor"], ["Tasa", "5"]]
                            for node in result.nodes))
        links = [link for node in result.nodes for link in node.get("links", [])]
        self.assertTrue(any(link["url"] == extractor.BASE + "/ley/2" for link in links))
        for node in result.nodes:
            self.assertEqual(node["text"], result.text[node["start_char"]:node["end_char"]])
            self.assertTrue(node["source_locator"])

    def test_real_digesto_article_containers_override_printed_label_format(self):
        body = (Path(__file__).parent / "fixtures" / "digesto_3041.html").read_bytes()
        result = self.parse(body)
        self.assertEqual(result.status, "ok")
        articles = [node for node in result.nodes if node["type"] == "article"]
        self.assertEqual([node["label"] for node in articles], ["1o", "2o", "3o"])
        self.assertNotIn("La Municipalidad puede", articles[1]["text"])
        self.assertIn("La Municipalidad puede", articles[2]["text"])
        self.assertNotIn("Comuníquese", articles[2]["text"])
        closing = next(node for node in result.nodes if node["type"] == "conclusion")
        self.assertIsNone(closing["parent_id"])
        for node in result.nodes:
            self.assertEqual(node["text"], result.text[node["start_char"]:node["end_char"]])

    def test_markup_article_spans_nested_paragraphs_and_keeps_internal_references(self):
        body = '''<div class="article extra hcontainer"><span class="inline num">3o —</span>
        <p>Se concede el permiso.</p><p>Art. 99 se cita como referencia.</p>
        <table><tr><td>Tasa</td><td>5</td></tr></table></div>
        <div class="container conclusions">Comuníquese.</div>'''.encode()
        result = self.parse(body)
        self.assertEqual(result.status, "ok")
        articles = [node for node in result.nodes if node["type"] == "article"]
        self.assertEqual([node["label"] for node in articles], ["3o"])
        self.assertIn("Art. 99", articles[0]["text"])
        self.assertNotIn("Comuníquese", articles[0]["text"])
        table = next(node for node in result.nodes if node["type"] == "table")
        self.assertEqual(table["parent_id"], articles[0]["id"])

    def test_short_unnumbered_instrument_is_accepted(self):
        result = self.parse("<p>El Presidente de la República acuerda: concédese el permiso.</p>".encode())
        self.assertEqual(result.status, "ok")

    def test_layout_tables_do_not_hide_articles_or_nested_data_tables(self):
        body = b'<table><tr><td><p>Art. 1. Objeto</p><p>Se establece el permiso.</p>' \
               b'<table><tr><td colspan="2">Tasa</td></tr><tr><td>A</td><td>5</td></tr></table>' \
               b'<p>Art. 2. Excepciones</p><p>No se aplica al supuesto indicado.</p></td></tr></table>'
        result = self.parse(body)
        self.assertEqual(result.status, "ok")
        articles = [node for node in result.nodes if node["type"] == "article"]
        self.assertEqual([node["label"] for node in articles], ["1", "2"])
        table = next(node for node in result.nodes if node["type"] == "table")
        self.assertEqual(table["parent_id"], articles[0]["id"])
        self.assertEqual(table["cells"][0][0]["colspan"], "2")

    def test_ambiguous_nested_data_table_is_not_silently_flattened(self):
        body = b'<p>Art. 1. Derecho</p><table><tr><td>100 <table><tr><td>20</td><td>30</td>' \
               b'</tr></table></td><td>40</td></tr></table>'
        self.assertEqual(self.parse(body).status, "needs_review")

    def test_article_reference_in_data_table_keeps_table_type(self):
        body = '<p>La Asamblea Nacional ha dictado el siguiente decreto.</p>' \
               '<table><tr><td>Artículo 1</td><td>5</td></tr></table>'
        result = self.parse(body.encode())
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.nodes[-1]["type"], "table")

    def test_declared_encoding_and_unknown_encoding(self):
        body = "<p>El Presidente de la República acuerda: concédese el permiso.</p>".encode("latin-1")
        accepted = self.parse(body, content_type="text/html; charset=iso-8859-1")
        self.assertEqual(accepted.status, "ok")
        self.assertIn("República", accepted.text)
        self.assertNotEqual(self.parse(body).status, "ok")

    def test_short_article_labels_and_json_html_wrapper(self):
        for label in ("Arto. 1.-", "Artículo Único.", "Art. 2 bis."):
            wrapped = json.dumps({"html": f"<p>{label} Se concede el permiso.</p>"}).encode()
            result = self.parse(wrapped)
            self.assertEqual(result.status, "ok")
            self.assertEqual(len([node for node in result.nodes if node["type"] == "article"]), 1)

    def test_partial_catalog_does_not_replace_full_output(self):
        cache = self.root / "catalog_pages"
        cache.mkdir()
        for page in (1, 2):
            extractor.atomic_json(cache / f"page_{page:06d}.json", {
                "page": page, "fetched_at_utc": "2026-11-14T00:00:00+00:00",
                "total_reported": 3, "items": [instrument(page)]})
        output = self.root / "catalog.jsonl"
        original = '{"keep":"existing complete catalog"}\n'
        output.write_text(original, encoding="utf-8")
        extractor.catalog(argparse.Namespace(cache_dir=self.root, refresh=False,
                          max_pages=2, output=output, delay=1.0, retries=0))
        self.assertEqual(output.read_text(encoding="utf-8"), original)
        partial = self.root / "catalog.partial.jsonl"
        self.assertEqual(len(list(extractor.jsonl_rows(partial))), 2)
        manifest = json.loads((self.root / "catalog.partial.manifest.json").read_text())
        self.assertFalse(manifest["complete"])

    def test_unreconciled_catalog_preserves_existing_output(self):
        cache = self.root / "catalog_pages"
        cache.mkdir()
        for page, items in ((1, [instrument()]), (2, [])):
            extractor.atomic_json(cache / f"page_{page:06d}.json", {
                "page": page, "fetched_at_utc": "2026-11-14T00:00:00+00:00",
                "total_reported": 2, "items": items})
        output = self.root / "catalog.jsonl"
        output.write_text("original\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            extractor.catalog(argparse.Namespace(cache_dir=self.root, refresh=False,
                              max_pages=None, output=output, delay=1.0, retries=0))
        self.assertEqual(output.read_text(), "original\n")

    def test_empty_catalog_cannot_replace_existing_output_even_if_total_is_zero(self):
        cache = self.root / "catalog_pages"
        cache.mkdir()
        extractor.atomic_json(cache / "page_000001.json", {
            "page": 1, "fetched_at_utc": "2026-11-14T00:00:00+00:00",
            "total_reported": 0, "items": []})
        output = self.root / "catalog.jsonl"
        output.write_text("original\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            extractor.catalog(argparse.Namespace(cache_dir=self.root, refresh=False,
                              max_pages=None, output=output, delay=1.0, retries=0))
        self.assertEqual(output.read_text(), "original\n")

    def test_interrupted_refresh_cannot_mix_new_and_active_cached_pages(self):
        cache = self.root / "catalog_pages"
        cache.mkdir()
        for page, items in ((1, [instrument(1)]), (2, [instrument(2)]), (3, [])):
            extractor.atomic_json(cache / f"page_{page:06d}.json", {
                "page": page, "fetched_at_utc": "2026-11-14T00:00:00+00:00",
                "total_reported": 2, "items": items})
        args = argparse.Namespace(cache_dir=self.root, refresh=True, max_pages=None,
                                  output=self.root / "catalog.jsonl", delay=1.0, retries=0)
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.list_page.side_effect = [([instrument(3)], 2), requests.Timeout()]
            with self.assertRaises(requests.Timeout):
                extractor.catalog(args)
        args.refresh = False
        with patch.object(extractor, "DigestoClient") as client:
            extractor.catalog(args)
            client.assert_not_called()
        self.assertEqual([row["iunpid"] for row in extractor.jsonl_rows(args.output)], ["MQ==", "Mg=="])

    def test_successful_refresh_promotes_one_complete_cache(self):
        args = argparse.Namespace(cache_dir=self.root, refresh=True, max_pages=None,
                                  output=self.root / "catalog.jsonl", delay=1.0, retries=0)
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.list_page.side_effect = [([instrument(3)], 1), ([], 1)]
            extractor.catalog(args)
        args.refresh = False
        with patch.object(extractor, "DigestoClient") as client:
            extractor.catalog(args)
            client.assert_not_called()
        self.assertEqual([row["iunpid"] for row in extractor.jsonl_rows(args.output)], ["Mw=="])

    def test_retry_failures_and_refresh_keep_one_current_record(self):
        args = self.text_args(retry_failed=True)
        self.write_rows(args.metadata, [instrument(1), instrument(2), instrument(3)])
        self.write_rows(args.output, [{"iunpid": "MQ==", "status": "empty_response", "text": ""},
                                     {"iunpid": "Mg==", "status": "ok", "text": "Existing good text"}])
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.return_value = response()
            extractor.texts(args)
            self.assertEqual(client.return_value.main_text.call_args_list[0].args, ("MQ==",))
            self.assertEqual(client.return_value.main_text.call_count, 1)
        rows = list(extractor.jsonl_rows(args.output))
        self.assertEqual(len(rows), 2)
        current = next(row for row in rows if row["iunpid"] == "MQ==")
        self.assertEqual(current["status"], "ok")
        self.assertEqual(Path(current["raw_html_file"]).read_bytes(), LEGAL_HTML)
        self.assertEqual(current["raw_sha256"], hashlib.sha256(LEGAL_HTML).hexdigest())
        args.retry_failed = False
        args.refresh = True
        args.ids = ["1"]
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.return_value = response(b"")
            extractor.texts(args)
        self.assertEqual(list(extractor.jsonl_rows(args.output)), rows)
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.return_value = response(LEGAL_HTML.replace(b"5</td>", b"8</td>"))
            extractor.texts(args)
        refreshed = list(extractor.jsonl_rows(args.output))
        self.assertEqual(len(refreshed), 2)
        self.assertIn("8", refreshed[0]["text"])
        self.assertNotEqual(refreshed[0]["text_sha256"], rows[0]["text_sha256"])

    def test_failed_snapshot_publication_is_recoverable(self):
        args = self.text_args(retry_failed=True)
        self.write_rows(args.metadata, [instrument()])
        previous = {"iunpid": "MQ==", "status": "empty_response", "text": ""}
        self.write_rows(args.output, [previous])
        original = args.output.read_bytes()
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.return_value = response()
            with patch.object(extractor, "atomic_jsonl", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    extractor.texts(args)
        self.assertEqual(args.output.read_bytes(), original)
        args.retry_failed = False
        with patch.object(extractor, "DigestoClient") as client:
            extractor.texts(args)
            client.assert_not_called()
        self.assertEqual(list(extractor.jsonl_rows(args.output))[0]["status"], "ok")

    def test_document_http_failure_does_not_discard_later_success(self):
        args = self.text_args()
        self.write_rows(args.metadata, [instrument(1), instrument(2)])
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.side_effect = [requests.Timeout("timed out"), response()]
            extractor.texts(args)
        rows = list(extractor.jsonl_rows(args.output))
        self.assertEqual([row["status"] for row in rows], ["http_error", "ok"])

    def test_completed_records_survive_later_access_block(self):
        args = self.text_args()
        self.write_rows(args.metadata, [instrument(1), instrument(2)])
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.side_effect = [response(), RuntimeError("access blocked")]
            with self.assertRaises(RuntimeError):
                extractor.texts(args)
        rows = list(extractor.jsonl_rows(args.output))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")

    def test_pending_current_record_is_recovered_without_refetch(self):
        args = self.text_args()
        self.write_rows(args.metadata, [instrument()])
        pending = args.output.with_name(args.output.name + ".pending")
        pending.mkdir()
        extractor.atomic_json(pending / "1.json", {"iunpid": "MQ==", "status": "ok", "text": "Recovered"})
        with patch.object(extractor, "DigestoClient") as client:
            extractor.texts(args)
            client.assert_not_called()
        self.assertEqual(list(extractor.jsonl_rows(args.output))[0]["text"], "Recovered")

    def test_transient_http_retry_respects_blocked_statuses(self):
        for statuses, expected_calls in (((503, 200), 2), ((403,), 1), ((429,), 1)):
            with self.subTest(statuses=statuses):
                client = extractor.DigestoClient(1.0, retries=1)
                with patch.object(client.session, "request", side_effect=[response(status=s) for s in statuses]) as call:
                    with patch("time.sleep"):
                        if statuses[0] in (403, 429):
                            with self.assertRaises(RuntimeError):
                                client.main_text("MQ==")
                        else:
                            self.assertEqual(client.main_text("MQ==").status_code, 200)
                self.assertEqual(call.call_count, expected_calls)

    def test_invalid_ids_do_not_pass_validation(self):
        self.assertFalse(extractor.is_real_record({"iunpid": "not-base64"}))
        self.assertFalse(extractor.is_real_record({"iunpid": "MA=="}))

    def configured_paths(self):
        return {"METADATA_PATH": self.root / "metadata.jsonl", "TEXT_PATH": self.root / "texts.jsonl",
                "CACHE_DIR": self.root / "cache", "RAW_RESPONSE_DIR": self.root / "raw"}

    def test_callable_runs_full_pipeline_using_configured_paths_and_returns_results(self):
        paths = self.configured_paths()
        with patch.multiple(pipelines.initial.config, **paths), patch.object(extractor, "DigestoClient") as client:
            client.return_value.list_page.side_effect = [([instrument()], 1), ([], 1)]
            client.return_value.main_text.return_value = response()
            result = extractor.run_extraction(limit=None)
        self.assertTrue(result.catalog_complete)
        self.assertEqual(result.metadata_path, paths["METADATA_PATH"])
        self.assertEqual(result.text_path, paths["TEXT_PATH"])
        self.assertTrue(result.manifest_path.is_file())
        self.assertEqual((result.fetched, result.accepted, result.rejected), (1, 1, 0))
        self.assertTrue((paths["CACHE_DIR"] / "catalog_pages/page_000001.json").is_file())
        row = next(extractor.jsonl_rows(result.text_path))
        self.assertEqual(Path(row["raw_html_file"]).parent, paths["RAW_RESPONSE_DIR"])

    def test_callable_text_stage_supports_path_overrides_numeric_ids_and_resume(self):
        paths = self.configured_paths()
        self.write_rows(paths["METADATA_PATH"], [instrument(1), instrument(2)])
        with patch.object(extractor, "DigestoClient") as client:
            client.return_value.main_text.return_value = response()
            result = extractor.run_extraction(stage="texts", metadata_path=paths["METADATA_PATH"],
                                             text_path=paths["TEXT_PATH"], raw_dir=paths["RAW_RESPONSE_DIR"],
                                             ids=[2], limit=1)
            client.return_value.main_text.assert_called_once_with("Mg==")
        self.assertIsNone(result.catalog_complete)
        self.assertIsNone(result.manifest_path)
        self.assertEqual(result.accepted, 1)
        with patch.object(extractor, "DigestoClient") as client:
            again = extractor.run_extraction(stage="texts", metadata_path=paths["METADATA_PATH"],
                                            text_path=paths["TEXT_PATH"], raw_dir=paths["RAW_RESPONSE_DIR"], ids=[2])
            client.assert_not_called()
        self.assertEqual(again.fetched, 0)
        self.assertEqual(again.text_path, paths["TEXT_PATH"])

    def test_callable_partial_pipeline_uses_sample_without_replacing_canonical_metadata(self):
        paths = self.configured_paths()
        self.write_rows(paths["METADATA_PATH"], [instrument(2)])
        original = paths["METADATA_PATH"].read_bytes()
        with patch.multiple(pipelines.initial.config, **paths), patch.object(extractor, "DigestoClient") as client:
            client.return_value.list_page.return_value = ([instrument(1)], 2)
            client.return_value.main_text.return_value = response()
            result = extractor.run_extraction(max_pages=1)
            client.return_value.main_text.assert_called_once_with("MQ==")
        self.assertFalse(result.catalog_complete)
        self.assertEqual(result.metadata_path.name, "metadata.partial.jsonl")
        self.assertEqual(paths["METADATA_PATH"].read_bytes(), original)
        self.assertEqual([row["iunpid"] for row in extractor.jsonl_rows(result.text_path)], ["MQ=="])

    def test_callable_invalid_options_and_colliding_paths_fail_before_writes(self):
        cases = [{"delay": 0}, {"retries": 6}, {"limit": 0}, {"ids": "3041"},
                 {"ids": [0]}, {"refresh_texts": True, "retry_failed": True},
                 {"stage": "unknown"}, {"stage": "texts", "max_pages": 1},
                 {"stage": "catalog", "ids": [1]}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch.object(extractor, "DigestoClient") as client:
                with self.assertRaises(ValueError):
                    extractor.run_extraction(**kwargs)
                client.assert_not_called()
        destination = self.root / "same.jsonl"
        with self.assertRaises(ValueError):
            extractor.run_extraction(metadata_path=destination, text_path=destination)
        self.assertFalse(destination.exists())
        with self.assertRaises(ValueError):
            extractor.run_extraction(metadata_path=destination,
                                     text_path=self.root / "same.partial.jsonl")
        self.assertFalse(destination.exists())
        with self.assertRaises(ValueError):
            extractor.run_extraction(metadata_path=self.root)
        occupied = self.root / "occupied"
        occupied.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            extractor.run_extraction(cache_dir=occupied)
        self.assertEqual(occupied.read_text(), "keep")

    def test_callable_missing_metadata_raises_instead_of_exiting(self):
        paths = self.configured_paths()
        with patch.multiple(pipelines.initial.config, **paths), patch.object(extractor, "DigestoClient") as client:
            with self.assertRaises(FileNotFoundError):
                extractor.run_extraction(stage="texts")
            client.assert_not_called()

    def test_cli_uses_config_defaults_and_the_callable_entrypoint(self):
        paths = self.configured_paths()
        with patch.multiple(pipelines.initial.config, **paths), patch.object(extractor, "run_extraction") as run:
            with patch.object(extractor.logging, "basicConfig"):
                result = extractor.main(["texts", "--all", "--refresh", "--ids", "3041"])
            self.assertIs(result, run.return_value)
            self.assertEqual(run.call_args.kwargs["metadata_path"], paths["METADATA_PATH"])
            self.assertEqual(run.call_args.kwargs["text_path"], paths["TEXT_PATH"])
            self.assertEqual(run.call_args.kwargs["raw_dir"], paths["RAW_RESPONSE_DIR"])
            self.assertIsNone(run.call_args.kwargs["limit"])
            self.assertTrue(run.call_args.kwargs["refresh_texts"])
            self.assertEqual(run.call_args.kwargs["ids"], ["3041"])

    def test_import_is_inert_and_defaults_do_not_depend_on_working_directory(self):
        project = Path(__file__).resolve().parents[1]
        code = f"import sys; sys.path.insert(0, {str(project)!r}); import pipelines.initial.extract, pipelines.initial.config; print(pipelines.initial.config.METADATA_PATH)"
        result = subprocess.run([sys.executable, "-c", code], cwd=self.root, text=True,
                                capture_output=True, check=True)
        self.assertEqual(Path(result.stdout.strip()), project / "pipelines" / "initial" / "output" / "digesto_metadata.jsonl")
        self.assertFalse(list(self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
