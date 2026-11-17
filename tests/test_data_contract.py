"""Check that the YAML contract matches transformation and SQL interfaces."""

from pathlib import Path
import re
import unittest

import pandas as pd
import yaml

from pipelines.initial.transform import _build_tables, _create_node_df


ROOT = Path(__file__).resolve().parents[1]


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = yaml.safe_load((ROOT / "pipelines/initial/data_contract.yaml").read_text(encoding="utf-8"))

    def test_only_two_tables_and_their_columns_match_transformation_output(self):
        tables = _build_tables(pd.DataFrame(), {
            "documents": pd.DataFrame(), "nodes": _create_node_df(pd.DataFrame()),
        })
        self.assertEqual(set(self.contract["tables"]), {"legal_instruments", "document_nodes"})
        self.assertEqual(set(tables), set(self.contract["input"]["required_keys"]))
        for name, dataframe in tables.items():
            with self.subTest(table=name):
                fields = self.contract["tables"][name]["columns"]
                self.assertEqual(set(dataframe.columns), set(fields))
                self.assertTrue(all(type(field["nullable"]) is bool for field in fields.values()))

    def test_all_statuses_have_state_rules_and_rule_ids_are_unique(self):
        instrument = self.contract["tables"]["legal_instruments"]
        statuses = instrument["columns"]["extraction_status"]["allowed_values"]
        self.assertEqual(set(statuses), set(instrument["extraction_states"]))
        self.assertTrue(all(isinstance(status, str) for status in statuses))
        rules = self.contract["cross_table_rules"] + self.contract["warnings"]
        for table in self.contract["tables"].values():
            rules += table["row_rules"]
        self.assertEqual(len(rules), len({rule["id"] for rule in rules}))
        self.assertEqual(instrument["columns"]["fecha_aprobacion"]["source_field"], "fechaAprobacion")
        self.assertEqual(instrument["columns"]["fecha_publicacion"]["source_field"], "fechaPublicacion")

    def test_sql_creates_only_the_declared_tables_and_resolves_node_source_ids(self):
        schema = (ROOT / "database/schema.sql").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r"CREATE TABLE\s+(\w+)", schema)), set(self.contract["tables"]))
        self.assertNotRegex(schema, r"\bDROP\s+TABLE\b")
        node = self.contract["tables"]["document_nodes"]
        self.assertIs(node["columns"]["source_id"]["persisted"], False)
        self.assertEqual(node["database_columns"]["legal_instrument_id"]["references"], "legal_instruments.id")
        self.assertIn("FOREIGN KEY (legal_instrument_id, parent_node_key)", schema)


if __name__ == "__main__":
    unittest.main()
