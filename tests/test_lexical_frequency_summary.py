import json
import tempfile
import unittest
from pathlib import Path

from scripts import lexical_frequency_summary


class LexicalFrequencySummaryTest(unittest.TestCase):
    def test_summarize_reports_negative_frequency_length_correlation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exp_dir = root / "mini_exp"
            exp_dir.mkdir(parents=True)
            grammar = {
                "name": "abc123",
                "n_rules": 1,
                "n_words": 2,
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
                    "orthography": "latin",
                    "head_initial": True,
                    "spec_initial": True,
                },
                "b": {
                    "orthography": "latin",
                    "head_initial": False,
                    "spec_initial": True,
                },
            }
            (exp_dir / "grammar_abc123.json").write_text(json.dumps(grammar))
            rows = [
                {"left_phonetic": "a", "right_phonetic": "x"},
                {"left_phonetic": "a", "right_phonetic": "x"},
                {"left_phonetic": "a", "right_phonetic": "x"},
                {"left_phonetic": "bbbb", "right_phonetic": "yyyy"},
            ]
            (exp_dir / "samples_abc123.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            summary = lexical_frequency_summary.summarize(
                data_dir=root,
                experiments=["mini"],
                model="cl100k_base",
                max_samples_per_grammar=None,
            )

            source = next(row for row in summary if row["side"] == "source")
            self.assertEqual("zipf_length", source["lexical_frequency_profile"])
            self.assertLess(source["spearman_frequency_char_length"], 0)


if __name__ == "__main__":
    unittest.main()
