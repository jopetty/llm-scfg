import json
import tempfile
import unittest
from pathlib import Path

import main


class AgreementCliTest(unittest.TestCase):
    def test_deterministic_seed_is_stable(self):
        seed_one = main.deterministic_seed(
            "agreement", 100, 0, (("agreement_enabled_a", True),)
        )
        seed_two = main.deterministic_seed(
            "agreement", 100, 0, (("agreement_enabled_a", True),)
        )
        seed_three = main.deterministic_seed(
            "agreement", 100, 1, (("agreement_enabled_a", True),)
        )

        self.assertEqual(seed_one, seed_two)
        self.assertNotEqual(seed_one, seed_three)
        self.assertGreaterEqual(seed_one, 42)
        self.assertLess(seed_one, 10_042)

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
                self.assertIn("V1[lemma] -> <", grammar["grammar_str"])
                self.assertIn("1.sg", grammar["grammar_str"])
                self.assertIn("# Agreement metadata:", grammar["grammar_str"])

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
                self.assertIn("possible_right_phonetic", first)
                self.assertGreaterEqual(len(first["possible_right_phonetic"]), 1)
            finally:
                main.DATA_DIR = original_data_dir

    def test_experiment_batchfile_includes_agreement_prompt_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            batch_dir = root / "batches"
            exp_dir = data_dir / "agreement_exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            batch_dir.mkdir(parents=True, exist_ok=True)

            original_project_root = main.PROJECT_ROOT
            original_data_dir = main.DATA_DIR
            original_batch_dir = main.BATCH_DIR
            try:
                main.PROJECT_ROOT = root
                main.DATA_DIR = data_dir
                main.BATCH_DIR = batch_dir

                grammar_name = main.create_grammar(
                    rng_seed=37,
                    n_verbs=2,
                    n_nouns=2,
                    n_adjectives=1,
                    n_propns=2,
                    n_det_def=1,
                    n_det_indef=1,
                    n_prons=2,
                    n_comps=1,
                    exp_name="agreement",
                    agreement_enabled_a=True,
                    agreement_enabled_b=True,
                )
                main.generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=37,
                    min_depth=0,
                    max_depth=0,
                    n_samples_per_depth=1,
                    exp_name="agreement",
                )
                with open(exp_dir / "agreement_grammars.txt", "w") as handle:
                    handle.write(f"{grammar_name}\n")

                main.generate_experiment_batchfile(
                    exp="agreement",
                    model="gpt-5-nano",
                    data_source="local",
                )

                out_files = list((batch_dir / "agreement_exp").glob("*.jsonl"))
                self.assertEqual(1, len(out_files))
                with open(out_files[0]) as handle:
                    payload = json.loads(handle.readline())
                prompt = payload["body"]["messages"][0]["content"]
                self.assertIn("Some lexical items in this grammar inflect", prompt)
                metadata = payload["body"]["metadata"]
                self.assertEqual(grammar_name, metadata["grammar_name"])
                self.assertEqual("0", metadata["sample_id"])
                self.assertEqual("0", metadata["depth"])
                self.assertIn("input_sentence", metadata)
                self.assertIn("output_sentence", metadata)
                self.assertIn("n_words", metadata)
                self.assertIn("n_rules", metadata)
                self.assertIn("prompt_tokens", metadata)
                self.assertTrue(payload["custom_id"].startswith(f"{grammar_name}-"))
                self.assertIn("-sample-0", payload["custom_id"])
            finally:
                main.PROJECT_ROOT = original_project_root
                main.DATA_DIR = original_data_dir
                main.BATCH_DIR = original_batch_dir

    def test_compact_prompt_grammar_is_smaller_than_full_grammar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_data_dir = main.DATA_DIR
            try:
                main.DATA_DIR = data_dir
                grammar_name = main.create_grammar(
                    rng_seed=41,
                    n_verbs=12,
                    n_nouns=12,
                    n_adjectives=2,
                    n_propns=4,
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=6,
                    n_comps=2,
                    agreement_enabled_a=False,
                    agreement_enabled_b=True,
                )
                main.generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=41,
                    min_depth=0,
                    max_depth=0,
                    n_samples_per_depth=1,
                )
                with open(data_dir / f"grammar_{grammar_name}.json") as handle:
                    grammar = json.load(handle)
                with open(data_dir / f"samples_{grammar_name}.jsonl") as handle:
                    sample = json.loads(handle.readline())

                full_prompt = main.basic_prompt(
                    grammar_str=main.prompt_grammar_str(
                        grammar, sample["left_phonetic"], prompt_type="basic"
                    ),
                    sample=sample["left_phonetic"],
                    agreement_metadata=grammar.get("agreement_metadata"),
                )
                compact_prompt = main.basic_prompt(
                    grammar_str=main.prompt_grammar_str(
                        grammar, sample["left_phonetic"], prompt_type="compact"
                    ),
                    sample=sample["left_phonetic"],
                    agreement_metadata=grammar.get("agreement_metadata"),
                )

                self.assertLess(len(compact_prompt), len(full_prompt))
                self.assertIn(sample["left_phonetic"], compact_prompt)
                self.assertIn(" -> <", compact_prompt)
                self.assertIn("1.sg=<'", compact_prompt)
                self.assertNotIn("[1.sg] -> <", compact_prompt)
            finally:
                main.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
