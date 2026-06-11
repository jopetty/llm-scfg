import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts import build_hf_dataset


class BuildHFDatasetTest(unittest.TestCase):
    def test_build_writes_current_schema_and_fewshot_shots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            exp_dir = data_dir / "fewshot_exp"
            old_dir = data_dir / "old" / "archived_exp"
            output_dir = root / "hf"
            exp_dir.mkdir(parents=True)
            old_dir.mkdir(parents=True)

            grammar_name = "abc123"
            grammar = {
                "name": grammar_name,
                "grammar_str": "S -> N V",
                "n_rules": 1,
                "n_words": 2,
                "agreement_metadata": {"enabled": False},
                "lexical_frequency_metadata": {
                    "a": {
                        "profile": "zipf_length",
                        "exponent": 1.0,
                        "length_unit": "chars",
                    },
                    "b": {
                        "profile": "zipf_length",
                        "exponent": 1.0,
                        "length_unit": "chars",
                    },
                },
                "a": {
                    "head_initial": True,
                    "spec_initial": True,
                    "orthography": "latin",
                    "rng_seed": 1,
                    "lexical_frequency_profile": "zipf_length",
                    "lexical_frequency_exponent": 1.0,
                    "lexical_frequency_length_unit": "chars",
                    "verbs": ["go"],
                    "nouns": ["cat"],
                },
                "b": {
                    "head_initial": False,
                    "spec_initial": True,
                    "orthography": "latin",
                    "rng_seed": 2,
                    "lexical_frequency_profile": "zipf_length",
                    "lexical_frequency_exponent": 1.0,
                    "lexical_frequency_length_unit": "chars",
                    "verbs": ["ga"],
                    "nouns": ["ka"],
                },
            }
            (exp_dir / f"grammar_{grammar_name}.json").write_text(json.dumps(grammar))

            sample = {
                "grammar_name": grammar_name,
                "left": "cat go",
                "right": "ka ga",
                "left_phonetic": "cat go",
                "right_phonetic": "ka ga",
                "left_tree": "(S cat go)",
                "right_tree": "(S ka ga)",
                "depth": 0,
                "min_depth": 0,
                "max_depth": 1,
                "rng_seed": 3,
                "agreement_trace": {"ok": True},
                "possible_right": ["ka ga"],
                "possible_right_phonetic": ["ka ga"],
            }
            (exp_dir / f"samples_{grammar_name}.jsonl").write_text(
                json.dumps(sample) + "\n"
            )
            shot = dict(sample, rng_seed=4, left="dog go", right="da ga")
            (exp_dir / f"shots_{grammar_name}.jsonl").write_text(
                json.dumps(shot) + "\n"
            )

            (old_dir / "grammar_old.json").write_text(json.dumps(grammar))

            build_hf_dataset.build(data_dir=data_dir, output_dir=output_dir)

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(["fewshot_exp"], list(manifest["experiments"]))
            self.assertEqual(1, manifest["experiments"]["fewshot_exp"]["n_samples"])
            self.assertEqual(1, manifest["experiments"]["fewshot_exp"]["n_shots"])

            grammar_rows = pq.read_table(
                output_dir / "fewshot_exp" / "grammars.parquet"
            ).to_pylist()
            self.assertEqual(
                "zipf_length", grammar_rows[0]["a_lexical_frequency_profile"]
            )
            self.assertIn("lexical_frequency_metadata_json", grammar_rows[0])

            sample_rows = pq.read_table(
                output_dir / "fewshot_exp" / "samples.parquet"
            ).to_pylist()
            self.assertEqual("samples_abc123.jsonl", sample_rows[0]["source_file"])
            self.assertEqual(0, sample_rows[0]["row_index"])
            self.assertEqual('{"ok": true}', sample_rows[0]["agreement_trace"])
            self.assertIn('"left": "cat go"', sample_rows[0]["raw_json"])

            shot_rows = pq.read_table(
                output_dir / "fewshot_exp" / "shots.parquet"
            ).to_pylist()
            self.assertEqual("shots_abc123.jsonl", shot_rows[0]["source_file"])
            self.assertEqual("dog go", shot_rows[0]["left"])

            readme = (output_dir / "README.md").read_text()
            self.assertIn("split: shots", readme)


if __name__ == "__main__":
    unittest.main()
