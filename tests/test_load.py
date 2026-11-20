"""Loader checks; PostgreSQL cases use an explicitly supplied disposable server."""

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import uuid

import pandas as pd
import psycopg
from psycopg import sql

import config
import test_digesto_extract
import test_quality
from pipelines.initial import extract, load, main as pipeline
from tools.tools import retrieve


def sample_tables():
    return test_quality.QualityReportTests().tables()


class LoadInputTests(unittest.TestCase):
    def test_empty_batch_does_not_connect(self):
        frames = sample_tables()
        with patch.object(load.psycopg, "connect") as connect:
            result = load.upload_to_postgres(*(frame.iloc[:0] for frame in frames.values()))
        self.assertEqual(result, {"legal_instruments": 0, "document_nodes": 0})
        connect.assert_not_called()

    def test_orphan_nodes_and_duplicate_instruments_fail_before_connection(self):
        frames = sample_tables()
        with patch.object(load.psycopg, "connect") as connect:
            nodes = frames["document_nodes"].copy()
            nodes["source_id"] = "unknown"
            with self.assertRaises(ValueError):
                load.upload_to_postgres(frames["legal_instruments"], nodes)
            with self.assertRaises(ValueError):
                load.upload_to_postgres(pd.concat([frames["legal_instruments"]] * 2), frames["document_nodes"])
        connect.assert_not_called()

    def test_missing_columns_and_ok_instruments_without_nodes_are_rejected(self):
        frames = sample_tables()
        with patch.object(load.psycopg, "connect") as connect:
            with self.assertRaises(ValueError):
                load.upload_to_postgres(frames["legal_instruments"].drop(columns="fecha_aprobacion"), frames["document_nodes"])
            with self.assertRaises(ValueError):
                load.upload_to_postgres(frames["legal_instruments"], frames["document_nodes"].iloc[:0])
        connect.assert_not_called()

    def test_connection_loss_during_commit_does_not_repeat_the_upload(self):
        instruments, nodes = sample_tables().values()
        with patch.object(load.psycopg, "connect") as connect, patch.object(load, "register_vector"):
            connection = connect.return_value
            connection.cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
            connection.commit.side_effect = psycopg.OperationalError("Connection lost during commit")
            with self.assertLogs(load.logger, level="ERROR"), self.assertRaises(psycopg.OperationalError):
                load.upload_to_postgres(instruments, nodes)
        connect.assert_called_once()
        connection.commit.assert_called_once()


@unittest.skipUnless(os.environ.get("GOODMAN_TEST_POSTGRES_PORT"), "Needs disposable PostgreSQL")
class LoadPostgresTests(unittest.TestCase):
    def setUp(self):
        self.schema = "loader_test_" + uuid.uuid4().hex
        self.credentials = {
            "host": "127.0.0.1", "port": int(os.environ["GOODMAN_TEST_POSTGRES_PORT"]),
            "dbname": "goodman_loader_test", "user": "loader_test", "password": "loader_test",
            "options": f"-c search_path={self.schema}",
        }
        with psycopg.connect(**self.credentials, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
            schema = Path(__file__).resolve().parents[1] / "database" / "schema.sql"
            connection.execute(schema.read_text(encoding="utf-8"))
        self.addCleanup(self.drop_schema)
        self.patcher = patch.object(load, "DB_CREDENTIALS", self.credentials)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def drop_schema(self):
        with psycopg.connect(**self.credentials, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))

    def test_dates_nulls_integer_floats_json_and_generated_ids_survive(self):
        frames = sample_tables()
        instruments, nodes = frames.values()
        instruments.loc[0, "fecha_publicacion"] = None
        instruments.loc[0, "categoria"] = pd.NA
        instruments["text_chars"] = instruments["text_chars"].astype(float)
        nodes["start_char"] = nodes["start_char"].astype(float)
        nodes["attributes"] = [{"rows": [["á", "1"]]}]
        nodes["links"] = [[{"url": "https://example.com/ley", "text": "Ley"}]]
        instruments.index = [500]
        nodes.index = [900]
        before = {name: frame.copy(deep=True) for name, frame in frames.items()}
        self.assertEqual(load.upload_to_postgres(instruments, nodes), {"legal_instruments": 1, "document_nodes": 1})
        with psycopg.connect(**self.credentials) as connection:
            row = connection.execute("""
                SELECT i.id, i.fecha_aprobacion, i.fecha_publicacion, i.categoria,
                       i.text_chars, n.legal_instrument_id, n.links, n.attributes
                FROM legal_instruments i JOIN document_nodes n ON n.legal_instrument_id = i.id
            """).fetchone()
        self.assertEqual(row[0], row[5])
        self.assertNotEqual(row[0], 500)
        self.assertEqual(str(row[1]), "2015-06-04")
        self.assertEqual(row[2:5], (None, None, 4))
        self.assertEqual(row[6], nodes.iloc[0]["links"])
        self.assertEqual(row[7], nodes.iloc[0]["attributes"])
        for name in frames:
            pd.testing.assert_frame_equal(frames[name], before[name])

    def test_invalid_deferred_parent_rolls_back_both_tables(self):
        instruments, nodes = sample_tables().values()
        nodes["parent_node_key"] = "missing-parent"
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            load.upload_to_postgres(instruments, nodes)
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM legal_instruments").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone()[0], 0)

    def test_child_can_be_inserted_before_its_parent(self):
        instruments, nodes = sample_tables().values()
        child = nodes.copy(deep=True)
        child.loc[0, ["node_key", "parent_node_key", "position", "text", "end_char"]] = ["n2", "n1", 1, "T", 1]
        with patch.object(load, "INSERT_BATCH_SIZE", 1):
            load.upload_to_postgres(instruments, pd.concat([child, nodes], ignore_index=True))
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone()[0], 2)

    def test_metadata_only_instrument_keeps_null_content(self):
        instruments, nodes = sample_tables().values()
        instruments.loc[0, "extraction_status"] = "not_attempted"
        for column in ("text", "text_sha256", "text_chars", "status_reason", "text_fetched_at", "extractor_version"):
            instruments[column] = None
        self.assertEqual(load.upload_to_postgres(instruments, nodes.iloc[:0]),
                         {"legal_instruments": 1, "document_nodes": 0})
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT text, text_chars FROM legal_instruments").fetchone(), (None, None))

    def test_duplicate_batch_does_not_overwrite_existing_rows(self):
        instruments, nodes = sample_tables().values()
        load.upload_to_postgres(instruments, nodes)
        with self.assertRaisesRegex(ValueError, "requires empty"):
            load.initialize_database(require_empty=True)
        instruments.loc[0, "titulo"] = "Must not replace the original"
        with self.assertRaises(psycopg.errors.UniqueViolation):
            load.upload_to_postgres(instruments, nodes)
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT titulo FROM legal_instruments").fetchall(), [("Test law",)])
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone()[0], 1)

    def test_schema_upgrade_is_repeatable_and_preserves_existing_rows(self):
        instruments, nodes = sample_tables().values()
        load.upload_to_postgres(instruments, nodes)
        with psycopg.connect(**self.credentials) as connection:
            connection.execute("ALTER TABLE document_nodes DROP COLUMN embedding")
        load.initialize_database()
        load.initialize_database()
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT text, embedding FROM document_nodes").fetchall(),
                             [("Text", None)])

    def test_retrieval_combines_rankings_and_resolves_numbered_citations(self):
        instruments, nodes = sample_tables().values()
        instrument_rows = []
        node_rows = []
        query_vector = [1.0] + [0.0] * 1535
        perpendicular = [0.0, 1.0] + [0.0] * 1534
        for source_id, kind, title, label, text, embedding in (
            ("agreement", "Decreto Ejecutivo", "Decreto", "12 bis", "Ruido.", perpendicular),
            ("code", "Código", "Código de convivencia", "12º bis", "Se prohíben ruidos nocturnos.", query_vector),
            ("long", "Ley", "Protección ambiental", "5", "Ruido nocturno persistente.", None),
            ("title", "Ley", "Pesca artesanal", "1", "Disposiciones generales.", None),
            ("amendment", "Ley", "Reforma sobre pesca artesanal", "1", "Pesca artesanal en cooperativas.", None),
        ):
            instrument_rows.append({
                **instruments.iloc[0].to_dict(),
                "source_id": source_id, "rango": kind, "titulo": title, "numero": "42",
                "text": text, "text_chars": len(text),
            })
            node_rows.append({
                **nodes.iloc[0].to_dict(), "source_id": source_id, "node_type": "article",
                "label": label, "text": text, "end_char": len(text), "embedding": embedding,
            })
        load.upload_to_postgres(pd.DataFrame(instrument_rows), pd.DataFrame(node_rows))

        with patch.object(config, "DB_CREDENTIALS", self.credentials), \
                patch("tools.tools.OpenAI") as client:
            embed = client.return_value.__enter__.return_value.embeddings.create
            embed.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=query_vector)])
            exact = retrieve("¿Qué dice el artículo 12 bis de la Ley N.º 42?")
            self.assertEqual([row["source_id"] for row in exact], ["code"])
            self.assertEqual(exact[0]["text"], "Se prohíben ruidos nocturnos.")
            self.assertIsNone(exact[0]["score"])
            self.assertEqual(retrieve("Ley 42, art. 13"), [])
            embed.assert_not_called()

            results = retrieve("ruido nocturno", limit=3)
            self.assertEqual(results[0]["source_id"], "code")
            self.assertEqual(results[0]["distance"], 0)
            self.assertEqual({row["source_id"] for row in results}, {"agreement", "code", "long"})
            self.assertIsNone(next(row["distance"] for row in results if row["source_id"] == "long"))
            self.assertEqual(
                [row["source_id"] for row in retrieve("pesca artesanal", semantic=False)],
                ["title", "amendment"],
            )
            self.assertEqual(retrieve("de y", semantic=False), [])
            embed.assert_called_once_with(
                model=config.EMBEDDING_MODEL,
                dimensions=config.EMBEDDING_DIMENSIONS,
                input="ruido nocturno",
            )

            with psycopg.connect(**self.credentials, autocommit=True) as connection:
                index_sql = config.ROOT_DIR / "database" / "retrieval_index.sql"
                connection.execute(index_sql.read_text(encoding="utf-8"))
            self.assertEqual(retrieve("ruido nocturno", limit=3), results)

    def test_complete_pipeline_stores_vectors_and_preserves_document_spans(self):
        instrument = {**test_digesto_extract.instrument(), "categoria": "Ley", "rango": "Ley",
                      "fechaAprobacion": "04/06/2015", "fechaPublicacion": "09/10/2015"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"OUTPUT_DIR": root, "METADATA_PATH": root / "metadata.jsonl",
                     "TEXT_PATH": root / "texts.jsonl", "CACHE_DIR": root / "cache",
                     "RAW_RESPONSE_DIR": root / "raw"}
            with patch.multiple(config, **paths), patch.object(extract, "DigestoClient", autospec=True) as source, \
                    patch("openai.OpenAI") as openai, patch("tiktoken.encoding_for_model") as tokenizer:
                source.return_value.list_page.side_effect = [([instrument], 1), ([], 1)]
                source.return_value.main_text.return_value = test_digesto_extract.response()
                tokenizer.return_value.encode.side_effect = lambda text, **kwargs: text.split()
                api = openai.return_value.__enter__.return_value.embeddings.create
                api.side_effect = lambda **kwargs: SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=[float(index + 1)] * 1536)
                    for index in reversed(range(len(kwargs["input"])))
                ])
                counts = pipeline.run_initial_pipeline(limit=1)
        with psycopg.connect(**self.credentials) as connection:
            rows = connection.execute("""
                SELECT n.node_type, n.text, i.text, n.start_char, n.end_char,
                       vector_dims(n.embedding), (n.embedding::real[])[1],
                       n.embedding <=> n.embedding
                FROM document_nodes n JOIN legal_instruments i ON i.id = n.legal_instrument_id
                ORDER BY n.position
            """).fetchall()
        self.assertEqual(counts, {"legal_instruments": 1, "document_nodes": len(rows)})
        article_index = 0
        for kind, text, document, start, end, dimensions, value, distance in rows:
            self.assertEqual(text, document[start:end])
            if kind == "article":
                article_index += 1
                self.assertEqual((dimensions, value), (1536, float(article_index)))
                self.assertAlmostEqual(distance, 0)
            else:
                self.assertIsNone(dimensions)
        self.assertGreater(article_index, 0)

    def test_wrong_vector_dimension_rolls_back_both_tables(self):
        instruments, nodes = sample_tables().values()
        nodes["embedding"] = [[1.0, 2.0]]
        with self.assertRaises(psycopg.errors.DataException):
            load.upload_to_postgres(instruments, nodes)
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM legal_instruments").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone()[0], 0)

    def test_connection_loss_before_commit_reuses_the_prepared_rows(self):
        instruments, nodes = sample_tables().values()
        execute_many = psycopg.Cursor.executemany
        terminated = False

        def disconnect_once(cursor, statement, rows):
            nonlocal terminated
            execute_many(cursor, statement, rows)
            if not terminated:
                terminated = True
                with psycopg.connect(**self.credentials, autocommit=True) as observer:
                    self.assertEqual(observer.execute("SELECT count(*) FROM document_nodes").fetchone(), (0,))
                    self.assertTrue(observer.execute("SELECT pg_terminate_backend(%s, 5000)",
                                                     (cursor.connection.info.backend_pid,)).fetchone()[0])
                cursor.execute("SELECT 1")

        with patch.object(psycopg.Cursor, "executemany", disconnect_once), patch.object(load, "sleep"):
            counts = load.upload_to_postgres(instruments, nodes)
        self.assertTrue(terminated)
        self.assertEqual(counts, {"legal_instruments": 1, "document_nodes": 1})
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM legal_instruments").fetchone(), (1,))
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone(), (1,))
