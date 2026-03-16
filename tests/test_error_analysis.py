import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "error_analysis.py"
SPEC = importlib.util.spec_from_file_location("notebooks_error_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
error_analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = error_analysis
SPEC.loader.exec_module(error_analysis)


class ErrorAnalysisTest(unittest.TestCase):
    def test_matches_target_orthography_enforces_diacritic_policies(self):
        self.assertTrue(error_analysis.matches_target_orthography("roma", "latin"))
        self.assertFalse(error_analysis.matches_target_orthography("romá", "latin"))
        self.assertTrue(
            error_analysis.matches_target_orthography("romá", "latin_diacritic")
        )
        self.assertTrue(
            error_analysis.matches_target_orthography("אב", "hebrew_unpointed")
        )
        self.assertFalse(
            error_analysis.matches_target_orthography("אָב", "hebrew_unpointed")
        )

    def test_infer_target_orthography_distinguishes_large_hebrew_variants(self):
        self.assertEqual(
            "hebrew",
            error_analysis.infer_target_orthography("אָב", "orthography_large_exp"),
        )
        self.assertEqual(
            "hebrew_unpointed",
            error_analysis.infer_target_orthography("אב", "orthography_large_exp"),
        )
        self.assertEqual(
            "yiddish",
            error_analysis.infer_target_orthography("אָב", "orthography_exp"),
        )

    def test_classify_failure_flags_wrong_script_for_plain_latin_diacritics(self):
        row = pd.Series(
            {
                "exp": "orthography",
                "model_answer": "romá nova tera",
                "output_sentence": "roma nova tera",
                "input_sentence": "luma sora navi",
                "target_orthography": "latin",
                "b_words": ["roma", "nova", "tera"],
            }
        )
        self.assertEqual("wrong_script", error_analysis.classify_failure(row))

    def test_classify_failure_detects_hebrew_diacritic_drop(self):
        row = pd.Series(
            {
                "exp": "orthography",
                "model_answer": "אב גד זח",
                "output_sentence": "אָב גד זח",
                "input_sentence": "ab gd zh",
                "target_orthography": "hebrew",
                "b_words": ["אָב", "גד", "זח"],
            }
        )
        self.assertEqual("diacritic_drop", error_analysis.classify_failure(row))

    def test_merge_outputs_inputs_uses_fuzzy_model_when_custom_ids_repeat(self):
        outputs_df = pd.DataFrame(
            [
                {"custom_id": "dup", "fuzzy_model": "gpt-5", "model_answer": "alpha"},
                {
                    "custom_id": "dup",
                    "fuzzy_model": "gpt-5-mini",
                    "model_answer": "beta",
                },
            ]
        )
        inputs_df = pd.DataFrame(
            [
                {
                    "custom_id": "dup",
                    "fuzzy_model": "gpt-5",
                    "grammar_name": "grammar-a",
                    "sample_id": "0",
                    "input_sentence": "left-a",
                    "output_sentence": "right-a",
                    "depth": 0,
                    "n_words": 10,
                    "n_rules": 5,
                    "input_file": "inputs_a.jsonl",
                },
                {
                    "custom_id": "dup",
                    "fuzzy_model": "gpt-5-mini",
                    "grammar_name": "grammar-b",
                    "sample_id": "1",
                    "input_sentence": "left-b",
                    "output_sentence": "right-b",
                    "depth": 1,
                    "n_words": 12,
                    "n_rules": 6,
                    "input_file": "inputs_b.jsonl",
                },
            ]
        )

        merged_df = error_analysis.merge_outputs_inputs(outputs_df, inputs_df)
        by_model = merged_df.set_index("fuzzy_model")

        self.assertEqual("grammar-a", by_model.loc["gpt-5", "grammar_name"])
        self.assertEqual("left-b", by_model.loc["gpt-5-mini", "input_sentence"])
        self.assertEqual("right-b", by_model.loc["gpt-5-mini", "output_sentence"])

    def test_read_jsonl_prefix_until_keeps_fields_before_possible_right(self):
        content = (
            b'{"left_phonetic": "la", "right_phonetic": "ra", '
            b'"possible_right": ["ra"], "left_tree": "(S)"}\n'
        )
        handle = io.BytesIO(content)

        prefix = error_analysis.read_jsonl_prefix_until(handle, b', "possible_right"')
        line = prefix.decode("utf-8")

        self.assertEqual(
            "la",
            error_analysis.extract_json_field(
                line,
                '"left_phonetic": ',
                [', "right_phonetic":'],
            ),
        )
        self.assertEqual(
            "ra",
            error_analysis.extract_json_field(
                line,
                '"right_phonetic": ',
                [', "possible_right":', ', "left_tree":'],
            ),
        )

    def test_load_sample_sentences_reads_only_requested_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            grammar_name = "grammar123"
            sample_path = data_dir / f"samples_{grammar_name}.jsonl"
            sample_path.write_text(
                "\n".join(
                    [
                        '{"left": "a0", "right": "b0"}',
                        '{"left_phonetic": "a1", "right_phonetic": "b1"}',
                        '{"left": "a2", "right": "b2"}',
                    ]
                )
                + "\n"
            )

            sample_index_df = pd.DataFrame(
                [
                    {"grammar_name": grammar_name, "sample_id": "1"},
                    {"grammar_name": grammar_name, "sample_id": "2"},
                ]
            )

            sample_df = error_analysis.load_sample_sentences(data_dir, sample_index_df)

            self.assertEqual(
                [
                    {
                        "grammar_name": grammar_name,
                        "sample_id": "1",
                        "input_sentence": "a1",
                        "output_sentence": "b1",
                    },
                    {
                        "grammar_name": grammar_name,
                        "sample_id": "2",
                        "input_sentence": "a2",
                        "output_sentence": "b2",
                    },
                ],
                sample_df.to_dict(orient="records"),
            )


if __name__ == "__main__":
    unittest.main()
