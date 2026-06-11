import json
import tempfile
import unittest
from pathlib import Path

from scripts import tokenization_summary


class TokenizationSummaryTest(unittest.TestCase):
    def test_summarize_writes_rows_by_corpus_side_and_condition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            exp_dir = data_dir / "mini_exp"
            exp_dir.mkdir(parents=True)
            grammar = {
                "name": "abc123",
                "n_rules": 18,
                "n_words": 28,
                "grammar_str": "S -> <A B, B A>\nA -> <'foo', 'bar'>",
                "a": {
                    "head_initial": True,
                    "spec_initial": True,
                    "orthography": "latin",
                    "agreement_enabled": False,
                    "verbs": ["foo"],
                    "nouns": ["baz qux"],
                    "propns": [],
                    "prons": [],
                    "adjs": [],
                    "det_def": [],
                    "det_indef": [],
                    "comps": [],
                    "tenses": [],
                    "asps": [],
                },
                "b": {
                    "head_initial": False,
                    "spec_initial": True,
                    "orthography": "cyrillic",
                    "agreement_enabled": True,
                    "verbs": ["бар"],
                    "nouns": ["ква"],
                    "propns": [],
                    "prons": [],
                    "adjs": [],
                    "det_def": [],
                    "det_indef": [],
                    "comps": [],
                    "tenses": [],
                    "asps": [],
                },
            }
            (exp_dir / "grammar_abc123.json").write_text(
                json.dumps(grammar, ensure_ascii=False)
            )
            (exp_dir / "mini_grammars.txt").write_text("abc123\n")
            (exp_dir / "samples_abc123.jsonl").write_text(
                json.dumps(
                    {
                        "left_phonetic": "foo baz",
                        "right_phonetic": "ква бар",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_path = root / "summary.json"
            payload = tokenization_summary.summarize(
                data_dir=data_dir,
                output_path=output_path,
                experiments=["mini"],
                model_names=["cl100k_base"],
                max_samples_per_grammar=None,
                data_source="local",
            )

            self.assertTrue(output_path.exists())
            rows = payload["rows"]
            self.assertEqual(5, len(rows))

            by_corpus_side = {
                (row["corpus"], row["language_side"]): row for row in rows
            }
            source_vocab = by_corpus_side[("vocabulary_words", "a")]
            target_sentences = by_corpus_side[("sample_sentences", "b")]
            grammar_text = by_corpus_side[("grammar_text", "both")]

            self.assertEqual("mini", source_vocab["experiment"])
            self.assertEqual("SOV", source_vocab["target_word_order"])
            self.assertEqual("NoAgr -> Agr", source_vocab["agreement_condition"])
            self.assertEqual("latin", source_vocab["language_orthography"])
            self.assertEqual(3, source_vocab["n_words"])
            self.assertEqual(1, target_sentences["n_samples"])
            self.assertEqual("prompt", grammar_text["language_role"])


if __name__ == "__main__":
    unittest.main()
