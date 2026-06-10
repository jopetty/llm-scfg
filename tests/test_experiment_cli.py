import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from scfg.scfg import CFGParams


class ExperimentCliTest(unittest.TestCase):
    def test_estimate_prompt_tokens_uses_gemma_chat_template(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3, 4]

        with mock.patch.object(main, "gemma_tokenizer", return_value=tokenizer):
            token_count = main.estimate_prompt_tokens(
                "translate this",
                "google/gemma-3-12b-it",
            )

        self.assertEqual(4, token_count)
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "translate this"}],
            tokenize=True,
            add_generation_prompt=True,
        )

    def test_generate_experiment_batchfile_drops_rows_over_gemma_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            batch_dir = root / "batches"
            exp_dir = data_dir / "wordorder_large_exp"
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
                    rng_seed=17,
                    n_verbs=2,
                    n_nouns=2,
                    n_adjectives=1,
                    n_propns=2,
                    n_det_def=1,
                    n_det_indef=1,
                    n_prons=2,
                    n_comps=1,
                    exp_name="wordorder_large",
                )
                main.generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=17,
                    min_depth=0,
                    max_depth=0,
                    n_samples_per_depth=2,
                    exp_name="wordorder_large",
                )
                with open(exp_dir / "wordorder_large_grammars.txt", "w") as handle:
                    handle.write(f"{grammar_name}\n")

                with (
                    mock.patch.object(main, "model_input_token_limit", return_value=10),
                    mock.patch.object(
                        main,
                        "estimate_prompt_tokens",
                        side_effect=[5, 20],
                    ),
                ):
                    main.generate_experiment_batchfile(
                        exp="wordorder_large",
                        model="google/gemma-3-12b-it",
                    )

                out_files = list((batch_dir / "wordorder_large_exp").glob("*.jsonl"))
                self.assertEqual(1, len(out_files))
                with open(out_files[0]) as handle:
                    payloads = [json.loads(line) for line in handle]

                self.assertEqual(1, len(payloads))
                self.assertIn("-sample-0", payloads[0]["custom_id"])
                metadata = payloads[0]["body"]["metadata"]
                self.assertEqual(grammar_name, metadata["grammar_name"])
                self.assertEqual("5", metadata["prompt_tokens"])
            finally:
                main.PROJECT_ROOT = original_project_root
                main.DATA_DIR = original_data_dir
                main.BATCH_DIR = original_batch_dir

    def test_fewshot_experiment_writes_shots_and_batches_by_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            batch_dir = root / "batches"
            batch_dir.mkdir(parents=True, exist_ok=True)

            original_data_dir = main.DATA_DIR
            original_batch_dir = main.BATCH_DIR
            try:
                main.DATA_DIR = data_dir
                main.BATCH_DIR = batch_dir

                main.create_fewshot_data(
                    grammar_sizes=[25],
                    max_depth=0,
                    n_grammars_per_size=1,
                    n_sentences_per_depth=1,
                    n_shot_examples=1,
                    target_head_spec_params=[(False, True)],
                )
                fewshot_dir = data_dir / "fewshot_exp"
                self.assertTrue((fewshot_dir / "README.md").exists())
                self.assertTrue((fewshot_dir / "fewshot_grammars.txt").exists())
                fewshot_readme = (fewshot_dir / "README.md").read_text()
                self.assertIn("few-shot k values", fewshot_readme)
                self.assertIn("shots_*.jsonl", fewshot_readme)

                with open(fewshot_dir / "fewshot_grammars.txt") as handle:
                    grammar_names = [line.strip() for line in handle if line.strip()]
                self.assertEqual(1, len(grammar_names))
                grammar_name = grammar_names[0]
                self.assertTrue(
                    (fewshot_dir / f"samples_{grammar_name}.jsonl").exists()
                )
                self.assertTrue((fewshot_dir / f"shots_{grammar_name}.jsonl").exists())

                with (
                    mock.patch.object(main, "estimate_prompt_tokens", return_value=5),
                    mock.patch.object(main.secrets, "token_hex", return_value="abc123"),
                ):
                    main.generate_experiment_batchfile(
                        exp="fewshot",
                        model="gpt-5-nano",
                        k_shots=[0, 1],
                    )

                out_files = list((batch_dir / "fewshot_exp").glob("*.jsonl"))
                self.assertEqual(1, len(out_files))
                self.assertIn("inputs_fewshot_kshots_gpt-5-nano", out_files[0].name)
                with open(out_files[0]) as handle:
                    payloads = [json.loads(line) for line in handle]

                self.assertEqual(2, len(payloads))
                payloads_by_k = {
                    payload["body"]["metadata"]["k_shots"]: payload
                    for payload in payloads
                }
                self.assertEqual({"0", "1"}, set(payloads_by_k))
                self.assertIn("-k0-sample-0", payloads_by_k["0"]["custom_id"])
                self.assertIn("-k1-sample-0", payloads_by_k["1"]["custom_id"])
                prompt_0 = payloads_by_k["0"]["body"]["messages"][0]["content"]
                prompt_1 = payloads_by_k["1"]["body"]["messages"][0]["content"]
                self.assertNotIn("Here are example translations", prompt_0)
                self.assertIn("Here are example translations", prompt_1)
                self.assertIn("Source:", prompt_1)
                self.assertIn("Target:", prompt_1)
            finally:
                main.DATA_DIR = original_data_dir
                main.BATCH_DIR = original_batch_dir

    def test_new_orthographies_round_trip(self):
        cases = [
            ("latin_diacritic", lambda text: any(ord(char) > 127 for char in text)),
            (
                "hebrew_unpointed",
                lambda text: any("\u0590" <= char <= "\u05ff" for char in text),
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
                self.assertTrue(
                    (orthography_dir / "orthography_large_grammars.txt").exists()
                )
                orthography_readme = (orthography_dir / "README.md").read_text()
                self.assertIn("latin_diacritic", orthography_readme)
                self.assertIn("hebrew_unpointed", orthography_readme)

                with open(orthography_dir / "orthography_large_grammars.txt") as handle:
                    orthography_names = [
                        line.strip() for line in handle if line.strip()
                    ]
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
                self.assertTrue(
                    (wordorder_dir / "wordorder_large_grammars.txt").exists()
                )
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
                    target_orders.add(
                        (grammar["b"]["head_initial"], grammar["b"]["spec_initial"])
                    )

                self.assertEqual(
                    {(True, True), (False, True), (False, False)},
                    target_orders,
                )
            finally:
                main.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
