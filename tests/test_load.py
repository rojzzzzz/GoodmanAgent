"""Loader checks; PostgreSQL cases use an explicitly supplied disposable server."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import pandas as pd
import psycopg
from psycopg import sql

from pipelines.initial import load
import test_quality


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
        instruments.loc[0, "titulo"] = "Must not replace the original"
        with self.assertRaises(psycopg.errors.UniqueViolation):
            load.upload_to_postgres(instruments, nodes)
        with psycopg.connect(**self.credentials) as connection:
            self.assertEqual(connection.execute("SELECT titulo FROM legal_instruments").fetchall(), [("Test law",)])
            self.assertEqual(connection.execute("SELECT count(*) FROM document_nodes").fetchone()[0], 1)
