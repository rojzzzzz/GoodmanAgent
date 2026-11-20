from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAIError

import config
from tools import tools


class RetrievalTests(unittest.TestCase):
    @patch.object(tools.psycopg, "connect")
    @patch.object(tools, "OpenAI")
    def test_blank_query_does_not_call_external_services(self, client, connect):
        self.assertEqual(tools.retrieve(" \t\n "), [])
        client.assert_not_called()
        connect.assert_not_called()

    @patch.object(tools.psycopg, "connect")
    @patch.object(tools, "OpenAI")
    def test_explicit_citation_uses_normalized_number_and_article(self, client, connect):
        tools.retrieve("¿Qué dice el artículo 12.º bis de la Ley N.º 42?")
        connection = connect.return_value.__enter__.return_value
        _, params = connection.execute.call_args.args
        self.assertEqual(
            params, {"kind": "ley", "number": "42", "article": "12bis", "limit": config.K_NEAREST_NEIGHBORS}
        )
        client.assert_not_called()

    @patch.object(tools.psycopg, "connect")
    @patch.object(tools, "OpenAI", side_effect=OpenAIError("unavailable"))
    def test_topical_query_still_searches_when_embeddings_fail(self, client, connect):
        with self.assertLogs(tools.logger, level="WARNING"):
            tools.retrieve("  Ruido \n de noche ", limit=2)
        connection = connect.return_value.__enter__.return_value
        _, params = connection.execute.call_args.args
        self.assertEqual(params, {"query": "Ruido de noche", "embedding": None, "candidates": 8, "limit": 2})
        client.assert_called_once()

    @patch.object(tools.psycopg, "connect")
    @patch.object(tools, "OpenAI")
    def test_query_uses_the_ingestion_model_and_dimensions(self, client, connect):
        api = client.return_value.__enter__.return_value.embeddings.create
        api.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.125, 0.25])])
        connection = connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchone.return_value = {"ready": False}
        tools.retrieve("Ruido")
        _, params = connection.execute.call_args.args
        self.assertEqual(params["embedding"], "[0.125,0.25]")
        api.assert_called_once_with(
            model=config.EMBEDDING_MODEL, dimensions=config.EMBEDDING_DIMENSIONS, input="Ruido"
        )


class VigenciaTests(unittest.TestCase):
    source_id = "MTMzMDY="
    url = "https://digesto.asamblea.gob.ni/consultas/normas/shownorms.php?idnorm=MTMzMDY%3D"

    def response(self, script="", status=200):
        return httpx.Response(
            status, text=f"<script>\r\n{script}\r\n</script>", request=httpx.Request("GET", self.url)
        )

    @patch("httpx.get")
    def test_reports_official_instrument_classifications(self, get):
        for label, classification, status in (
            ("Vigente", "Vigente", "in_force"),
            ("Consolidada", "Consolidada", "in_force"),
            ("Sin Vigencia o Derecho Hist&oacute;rico", "Sin Vigencia o Derecho Histórico", "not_in_force"),
            ("", None, "unresolved"),
            ("Instrumento Internacional", "Instrumento Internacional", "unresolved"),
        ):
            with self.subTest(label=label):
                get.return_value = self.response(f"var idNorma = '{self.source_id}';\r\nvar registroNorma = '{label}';")
                result = tools.verify_vigencia(self.source_id)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["instrument_status"], status)
                self.assertEqual(result["classification"], classification)
                self.assertEqual(result["source_url"], self.url)
                self.assertIsNotNone(result["checked_at"])
        get.assert_called_with(self.url, timeout=8, follow_redirects=False)

    @patch("httpx.get")
    def test_instrument_status_does_not_confirm_a_specific_article(self, get):
        get.return_value = self.response(f"var idNorma = '{self.source_id}';\nvar registroNorma = 'Consolidada';")
        result = tools.verify_vigencia(self.source_id, " 45 ")
        self.assertEqual(result["article"], "45")
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["instrument_status"], "in_force")

    @patch("httpx.get")
    def test_requires_an_unambiguous_record_for_the_requested_id(self, get):
        for script in (
            "var idNorma = 'Mzkx';\nvar registroNorma = 'Vigente';",
            f"var idNorma = '{self.source_id}';\nvar registroNorma = 'Vigente';\nvar registroNorma = 'Consolidada';",
            f"var idNorma = '{self.source_id}';\n// var registroNorma = 'Vigente';",
        ):
            with self.subTest(script=script):
                get.return_value = self.response(script)
                self.assertEqual(tools.verify_vigencia(self.source_id)["status"], "unresolved")

    @patch("httpx.get")
    def test_reports_unresolved_when_the_official_record_cannot_be_read(self, get):
        for response in (self.response(status=429), self.response("// access denied")):
            get.return_value = response
            self.assertEqual(tools.verify_vigencia(self.source_id)["status"], "unresolved")
        get.side_effect = httpx.ReadTimeout("timeout")
        self.assertEqual(tools.verify_vigencia(self.source_id)["status"], "unresolved")

    @patch("httpx.get")
    def test_rejects_invalid_ids_without_a_request(self, get):
        self.assertEqual(tools.verify_vigencia("https://example.com")["status"], "unresolved")
        get.assert_not_called()
        self.assertIn(tools.verify_vigencia, tools.TOOLS)
