import json
import tempfile
import unittest
from pathlib import Path

import main
from scfg.scfg import CFGParams


class ExperimentCliTest(unittest.TestCase):
    def test_new_orthographies_round_trip(self):
        cases = [
            ("latin_diacritic", lambda text: any(ord(char) > 127 for char in text)),
            (
                "hebrew_unpointed",
                lambda text: any("\u0590" <= char <= "\u05FF" for char in text),
            ),
        ]

        for orthography, predicate in cases:
            with self.subTest(orthography=orthography):
                params = CFGParams(
                    rng_seed=11,
                    orthography=orthography,
                    verbs=2,
                    nouns=2,
                    propns=2,
                    prons=2,
                    adjs=1,
                    det_def=1,
                    det_indef=1,
                    comps=1,
                )
                cloned = CFGParams.from_dict(params.to_dict())
                self.assertEqual(orthography, cloned.orthography)
                self.assertTrue(predicate(cloned.verb_lemmas[0]))
                self.assertTrue(predicate(cloned.noun_lex[0]))

    def test_large_experiments_write_exp_dirs_and_readmes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_data_dir = main.DATA_DIR
            try:
                main.DATA_DIR = data_dir

                main.create_orthography_large_data(
                    grammar_sizes=[25],
                    max_depth=0,
                    n_grammars_per_size=1,
                    n_sentences_per_depth=1,
                )
                orthography_dir = data_dir / "orthography_large_exp"
                self.assertTrue((orthography_dir / "README.md").exists())
                self.assertTrue((orthography_dir / "orthography_large_grammars.txt").exists())
                orthography_readme = (orthography_dir / "README.md").read_text()
                self.assertIn("latin_diacritic", orthography_readme)
                self.assertIn("hebrew_unpointed", orthography_readme)

                with open(orthography_dir / "orthography_large_grammars.txt") as handle:
                    orthography_names = [line.strip() for line in handle if line.strip()]
                self.assertEqual(5, len(orthography_names))

                target_orthographies = set()
                for grammar_name in orthography_names:
                    grammar_path = orthography_dir / f"grammar_{grammar_name}.json"
                    samples_path = orthography_dir / f"samples_{grammar_name}.jsonl"
                    self.assertTrue(grammar_path.exists())
                    self.assertTrue(samples_path.exists())
                    with open(grammar_path) as handle:
                        grammar = json.load(handle)
                    target_orthographies.add(grammar["b"]["orthography"])

                self.assertEqual(
                    {
                        "latin",
                        "latin_diacritic",
                        "cyrillic",
                        "hebrew",
                        "hebrew_unpointed",
                    },
                    target_orthographies,
                )

                main.create_wordorder_large_data(
                    grammar_sizes=[25],
                    max_depth=0,
                    n_grammars_per_size=1,
                    n_sentences_per_depth=1,
                )
                wordorder_dir = data_dir / "wordorder_large_exp"
                self.assertTrue((wordorder_dir / "README.md").exists())
                self.assertTrue((wordorder_dir / "wordorder_large_grammars.txt").exists())
                wordorder_readme = (wordorder_dir / "README.md").read_text()
                self.assertIn("head-final, spec-initial", wordorder_readme)
                self.assertIn("head-final, spec-final", wordorder_readme)

                with open(wordorder_dir / "wordorder_large_grammars.txt") as handle:
                    wordorder_names = [line.strip() for line in handle if line.strip()]
                self.assertEqual(3, len(wordorder_names))

                target_orders = set()
                for grammar_name in wordorder_names:
                    grammar_path = wordorder_dir / f"grammar_{grammar_name}.json"
                    samples_path = wordorder_dir / f"samples_{grammar_name}.jsonl"
                    self.assertTrue(grammar_path.exists())
                    self.assertTrue(samples_path.exists())
                    with open(grammar_path) as handle:
                        grammar = json.load(handle)
                    target_orders.add((grammar["b"]["head_initial"], grammar["b"]["spec_initial"]))

                self.assertEqual(
                    {(True, True), (False, True), (False, False)},
                    target_orders,
                )
            finally:
                main.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
