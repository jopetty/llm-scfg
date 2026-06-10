import json
import tempfile
import unittest
from pathlib import Path

import prompt_viewer


class PromptViewerTest(unittest.TestCase):
    def test_extract_helpers(self):
        prompt = (
            "Here is the synchronous context-free grammar:\n"
            "    ```\n"
            "    S -> <A, B>\n"
            "    A -> <'foo', 'bar'>\n"
            "    ```\n\n"
            "    Here is the input sentence: `foo baz`.\n"
        )
        self.assertEqual(
            "S -> <A, B>\nA -> <'foo', 'bar'>",
            prompt_viewer.extract_grammar(prompt),
        )
        self.assertEqual("foo baz", prompt_viewer.extract_input_sentence(prompt))
        self.assertEqual(
            "bar baz", prompt_viewer.extract_final_answer("Final answer: bar baz")
        )

    def test_dataset_merges_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            data_dir = batch_dir / "data"
            data_dir.mkdir()
            input_path = batch_dir / "inputs_agreement_compact_gpt-5_part1_abcdef.jsonl"
            output_path = batch_dir / "batch_123_output.jsonl"
            grammar_path = data_dir / "grammar_abcdef1234567890.json"

            input_record = {
                "custom_id": "abcdef1234567890-abcdef-sample-0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Here is the synchronous context-free grammar:\n"
                                "    ```\n"
                                "    S -> <A, B>\n"
                                "    ```\n\n"
                                "    Here is the input sentence: `foo`.\n"
                            ),
                        }
                    ],
                    "metadata": {
                        "grammar_name": "abcdef1234567890",
                        "sample_id": "0",
                        "depth": "2",
                    },
                },
            }
            output_record = {
                "custom_id": "abcdef1234567890-abcdef-sample-0",
                "response": {
                    "status_code": 200,
                    "request_id": "req_123",
                    "body": {
                        "model": "gpt-5-2025-08-07",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Final answer: bar",
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                },
            }
            grammar_record = {
                "name": "abcdef1234567890",
                "agreement_metadata": {
                    "a": {"config": {"enabled": True}},
                    "b": {"config": {"enabled": False}},
                },
            }

            input_path.write_text(json.dumps(input_record) + "\n")
            output_path.write_text(json.dumps(output_record) + "\n")
            grammar_path.write_text(json.dumps(grammar_record) + "\n")

            dataset = prompt_viewer.BatchViewerDataset(batch_dir, data_dir)
            listing = dataset.query_records(search="foo", limit=10)

            self.assertEqual(1, listing["total_records"])
            self.assertEqual(1, listing["filtered_records"])
            self.assertEqual("gpt-5", listing["records"][0]["fuzzy_model"])
            self.assertEqual("bar", listing["records"][0]["final_answer"])
            self.assertEqual(
                "source_on__target_off",
                listing["records"][0]["agreement_condition"],
            )

            filtered_listing = dataset.query_records(
                agreement_condition="source_on__target_off",
                limit=10,
            )
            self.assertEqual(1, filtered_listing["filtered_records"])

            detail = dataset.get_record("abcdef1234567890-abcdef-sample-0")
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual("foo", detail["input_sentence"])
            self.assertEqual("S -> <A, B>", detail["grammar"])
            self.assertEqual("Final answer: bar", detail["response_text"])
            self.assertEqual(30, detail["total_tokens"])
            self.assertEqual(
                "Agr -> NoAgr",
                detail["agreement_condition_label"],
            )


if __name__ == "__main__":
    unittest.main()
