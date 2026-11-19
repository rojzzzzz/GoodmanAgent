"""Article input context follows the instrument's heading ancestry."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from pipelines.initial.embedding import build_embedding_inputs, embedder


class EmbeddingInputTests(unittest.TestCase):
    def prepare(self, nodes, **instrument_overrides):
        instruments = pd.DataFrame([
            {"source_id": source_id, "extraction_status": "ok", "rango": "Ley",
             "numero": "1", "titulo": "Trabajo", **instrument_overrides}
            for source_id in ("A", "B")
        ])
        frame = pd.DataFrame([
            {"source_id": "A", "node_key": "article", "parent_node_key": None,
             "node_type": "article", "label": "1", "source_label": "Art. 1",
             "text": "Texto completo.\nSegundo párrafo.", **node}
            for node in nodes
        ])
        original = frame.copy(deep=True)
        original_instruments = instruments.copy(deep=True)
        result = build_embedding_inputs(instruments, frame)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(instruments, original_instruments)
        return result

    def test_blank_article_label_falls_back_to_source_label(self):
        result = self.prepare([{"label": "  "}])
        self.assertIn("Articulo: Art. 1\n", result[("A", "article")])

    def test_optional_metadata_is_omitted_without_null_markers(self):
        result = self.prepare([{"label": None, "source_label": pd.NA}],
                              rango=pd.NA, numero=float("nan"))
        self.assertEqual(result[("A", "article")],
                         "Norma: Trabajo\nContenido: Texto completo.\nSegundo párrafo.")

    def test_unaccepted_instruments_and_non_articles_are_excluded(self):
        self.assertEqual(self.prepare([{}], extraction_status="needs_review"), {})
        self.assertEqual(self.prepare([{"node_type": "paragraph"}]), {})

    def test_duplicate_node_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Node keys must be unique"):
            self.prepare([{}, {}])

    def test_duplicate_instrument_ids_are_rejected(self):
        instruments = pd.DataFrame([{"source_id": "A"}, {"source_id": "A"}])
        with self.assertRaisesRegex(ValueError, "Instrument source IDs must be unique"):
            build_embedding_inputs(instruments, pd.DataFrame())

    def test_headings_follow_ancestry_not_row_order_or_parent_text(self):
        result = self.prepare([
            {"parent_node_key": "chapter"},
            {"node_key": "chapter", "parent_node_key": "title", "node_type": "heading",
             "source_label": "Capítulo II", "text": "Parent text includes descendants"},
            {"node_key": "title", "node_type": "heading", "source_label": "Título I"},
            {"node_key": "unrelated", "node_type": "heading", "source_label": "Unrelated"},
        ])
        self.assertEqual(result, {("A", "article"):
            "Norma: Ley N°. 1 - Trabajo\nUbicacion: Título I > Capítulo II\n"
            "Articulo: 1\nContenido: Texto completo.\nSegundo párrafo."})

    def test_local_keys_never_cross_instruments(self):
        result = self.prepare([
            {"parent_node_key": "chapter"},
            {"node_key": "chapter", "node_type": "heading", "source_label": "Chapter A"},
            {"source_id": "B", "node_key": "chapter", "node_type": "heading",
             "source_label": "Chapter B"},
        ])
        self.assertIn("Ubicacion: Chapter A\n", result[("A", "article")])
        self.assertNotIn("Chapter B", result[("A", "article")])

    def test_missing_labels_and_non_heading_ancestors_do_not_stop_traversal(self):
        result = self.prepare([
            {"parent_node_key": "blank", "label": pd.NA},
            {"node_key": "blank", "parent_node_key": "paragraph", "node_type": "heading",
             "source_label": pd.NA},
            {"node_key": "paragraph", "parent_node_key": "title", "node_type": "paragraph",
             "source_label": "Not a heading"},
            {"node_key": "title", "node_type": "heading", "source_label": "Título I"},
        ])
        self.assertIn("Ubicacion: Título I\nArticulo: Art. 1\n", result[("A", "article")])

    def test_root_article_has_no_location_line(self):
        for parent in (None, float("nan"), pd.NA):
            with self.subTest(parent=parent):
                result = self.prepare([{"parent_node_key": parent}])
                self.assertNotIn("Ubicacion:", result[("A", "article")])

    def test_missing_parent_cannot_resolve_in_another_instrument(self):
        with self.assertRaisesRegex(ValueError, "Missing parent chapter in instrument A"):
            self.prepare([
                {"parent_node_key": "chapter"},
                {"source_id": "B", "node_key": "chapter", "node_type": "heading"},
            ])

    def test_cycles_fail_instead_of_looping(self):
        with self.assertRaisesRegex(ValueError, "Cyclic ancestry"):
            self.prepare([
                {"parent_node_key": "chapter"},
                {"node_key": "chapter", "parent_node_key": "article", "node_type": "heading"},
            ])


class EmbedderTests(unittest.TestCase):
    def setUp(self):
        # Control token counts without downloading tokenizer data in unit tests.
        tokenizer = patch("tiktoken.encoding_for_model")
        self.encoder = tokenizer.start().return_value
        self.encoder.encode.side_effect = lambda text, **kwargs: text.split()
        self.addCleanup(tokenizer.stop)
        dimensions = patch.object(config, "EMBEDDING_DIMENSIONS", 2)
        dimensions.start()
        self.addCleanup(dimensions.stop)
        self.instruments = pd.DataFrame([
            {"source_id": source_id, "extraction_status": "ok", "rango": "Ley",
             "numero": "1", "titulo": "Trabajo"}
            for source_id in ("A", "B")
        ])
        self.nodes = pd.DataFrame([
            {"source_id": "B", "node_key": "n1", "node_type": "article"},
            {"source_id": "A", "node_key": "n2", "node_type": "paragraph"},
            {"source_id": "A", "node_key": "n1", "node_type": "article"},
        ], index=[8, 2, 8]).assign(
            parent_node_key=None, label="1", source_label="Art. 1", text="Article text",
        )

    def vector_values(self, result):
        return [None if vector is None else vector.tolist() for vector in result["embedding"]]

    @patch("openai.OpenAI")
    def test_vectors_match_both_keys_even_when_response_order_differs(self, client):
        client.return_value.__enter__.return_value.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0.125, 0.25]),
            SimpleNamespace(index=0, embedding=[0.375, 0.5]),
        ])
        original = self.nodes.copy(deep=True)
        result = embedder(self.instruments, self.nodes)
        self.assertEqual(self.vector_values(result), [[0.375, 0.5], None, [0.125, 0.25]])
        self.assertEqual(result.iloc[0]["embedding"].dtype, np.float32)
        pd.testing.assert_frame_equal(result.drop(columns="embedding"), original)
        pd.testing.assert_frame_equal(self.nodes, original)
        request = client.return_value.__enter__.return_value.embeddings.create.call_args.kwargs
        self.assertEqual(request["input"], list(build_embedding_inputs(self.instruments, self.nodes).values()))
        self.assertEqual(request["dimensions"], 2)
        self.assertEqual(request["model"], "text-embedding-3-small")

    @patch("openai.OpenAI")
    def test_no_articles_returns_copy_without_calling_api(self, client):
        for nodes in (self.nodes.iloc[:0], self.nodes.iloc[[1]]):
            with self.subTest(rows=len(nodes)):
                result = embedder(self.instruments, nodes)
                self.assertIsNot(result, nodes)
                self.assertEqual(result["embedding"].tolist(), [None] * len(nodes))
        client.assert_not_called()

    @patch("openai.OpenAI")
    def test_existing_embeddings_are_replaced_and_unselected_rows_are_cleared(self, client):
        self.nodes["embedding"] = [[9.0], [8.0], [7.0]]
        client.return_value.__enter__.return_value.embeddings.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1.0, 1.0]),
            SimpleNamespace(index=1, embedding=[2.0, 2.0]),
        ])
        result = embedder(self.instruments, self.nodes)
        self.assertEqual(self.vector_values(result), [[1.0, 1.0], None, [2.0, 2.0]])
        self.assertEqual(self.nodes["embedding"].tolist(), [[9.0], [8.0], [7.0]])

    @patch("openai.OpenAI")
    def test_incomplete_duplicate_or_invalid_response_indices_are_rejected(self, client):
        for indices in ([0], [0, 0], [0, 2], [-1, 0]):
            with self.subTest(indices=indices):
                client.return_value.__enter__.return_value.embeddings.create.return_value = SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=[1.0]) for index in indices
                ])
                with self.assertRaisesRegex(ValueError, "exactly one vector per input index"):
                    embedder(self.instruments, self.nodes)
        self.assertNotIn("embedding", self.nodes.columns)

    @patch("openai.OpenAI")
    def test_wrong_dimension_and_non_finite_response_vectors_are_rejected(self, client):
        api = client.return_value.__enter__.return_value.embeddings.create
        for vector in ([1.0], [float("nan"), 1.0], [1.0, float("inf")]):
            with self.subTest(vector=vector):
                api.return_value = SimpleNamespace(data=[
                    SimpleNamespace(index=0, embedding=vector),
                    SimpleNamespace(index=1, embedding=[1.0, 2.0]),
                ])
                with self.assertRaisesRegex(ValueError, "invalid vector"):
                    embedder(self.instruments, self.nodes)

    @patch("openai.OpenAI")
    def test_batches_map_response_indices_within_each_request(self, client):
        api = client.return_value.__enter__.return_value.embeddings.create
        api.side_effect = [SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[value, value])])
                           for value in (1.0, 2.0)]
        with patch.object(config, "EMBEDDING_BATCH_SIZE", 1):
            result = embedder(self.instruments, self.nodes)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(self.vector_values(result), [[1.0, 1.0], None, [2.0, 2.0]])

    @patch("openai.OpenAI")
    def test_oversized_article_is_held_without_truncating_text(self, client):
        self.nodes.iloc[0, self.nodes.columns.get_loc("text")] = " palabra" * 9000
        original = self.nodes.copy(deep=True)
        api = client.return_value.__enter__.return_value.embeddings.create
        api.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 1.0])])
        with self.assertLogs("pipelines.initial.embedding", level="WARNING") as logs:
            result = embedder(self.instruments, self.nodes)
        self.assertIn("B/n1", logs.output[0])
        self.assertEqual(self.vector_values(result), [None, None, [1.0, 1.0]])
        self.assertEqual(len(api.call_args.kwargs["input"]), 1)
        pd.testing.assert_frame_equal(result.drop(columns="embedding"), original)
        with patch.object(config, "EMBEDDING_MAX_TOKENS", 1):
            client.reset_mock()
            with self.assertLogs("pipelines.initial.embedding", level="WARNING"):
                result = embedder(self.instruments, self.nodes)
            client.assert_not_called()
            self.assertTrue(result["embedding"].isna().all())

    @patch("openai.OpenAI")
    def test_token_limit_includes_context_and_never_truncates(self, client):
        self.encoder.encode.side_effect = [range(8191), range(8192)]
        api = client.return_value.__enter__.return_value.embeddings.create
        api.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 1.0])])
        with self.assertLogs("pipelines.initial.embedding", level="WARNING"):
            result = embedder(self.instruments, self.nodes)
        inputs = build_embedding_inputs(self.instruments, self.nodes)
        self.assertEqual(api.call_args.kwargs["input"], [inputs[("B", "n1")]])
        self.assertEqual(self.vector_values(result), [[1.0, 1.0], None, None])
        self.assertEqual(self.encoder.encode.call_args.args[0], inputs[("A", "n1")])


if __name__ == "__main__":
    unittest.main()
