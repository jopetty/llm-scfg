import json
import tempfile
import unittest
from pathlib import Path

import main


class AgreementCliTest(unittest.TestCase):
    def test_create_grammar_and_samples_write_agreement_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_data_dir = main.DATA_DIR
            try:
                main.DATA_DIR = data_dir
                grammar_name = main.create_grammar(
                    rng_seed=31,
                    n_verbs=2,
                    n_nouns=2,
                    n_adjectives=1,
                    n_propns=2,
                    n_det_def=1,
                    n_det_indef=1,
                    n_prons=2,
                    n_comps=1,
                    agreement_enabled_a=True,
                    agreement_enabled_b=True,
                )
                grammar_path = data_dir / f"grammar_{grammar_name}.json"
                self.assertTrue(grammar_path.exists())
                with open(grammar_path) as handle:
                    grammar = json.load(handle)
                self.assertTrue(grammar["agreement_metadata"]["enabled"])
                self.assertIn("V -> <", grammar["grammar_str"])
                self.assertIn("1.sg", grammar["grammar_str"])
                self.assertIn("gender=", grammar["grammar_str"])

                main.generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=31,
                    min_depth=0,
                    max_depth=1,
                    n_samples_per_depth=2,
                )
                samples_path = data_dir / f"samples_{grammar_name}.jsonl"
                self.assertTrue(samples_path.exists())
                with open(samples_path) as handle:
                    first = json.loads(handle.readline())
                self.assertIn("agreement_ok", first)
                self.assertIn("subject_features", first)
            finally:
                main.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
