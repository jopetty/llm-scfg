import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parent.parent / "open_weights.py"
SPEC = importlib.util.spec_from_file_location("open_weights_module", MODULE_PATH)
assert SPEC and SPEC.loader
open_weights = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = open_weights
SPEC.loader.exec_module(open_weights)

ERROR_ANALYSIS_PATH = (
    Path(__file__).resolve().parent.parent / "notebooks" / "error_analysis.py"
)
ERROR_SPEC = importlib.util.spec_from_file_location(
    "notebooks_error_analysis_for_open_weights",
    ERROR_ANALYSIS_PATH,
)
assert ERROR_SPEC and ERROR_SPEC.loader
error_analysis = importlib.util.module_from_spec(ERROR_SPEC)
sys.modules[ERROR_SPEC.name] = error_analysis
ERROR_SPEC.loader.exec_module(error_analysis)


class OpenWeightsTest(unittest.TestCase):
    def test_default_wandb_project_uses_repo_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual("llm-scfg-vllm", open_weights.default_wandb_project())

    def test_should_enable_wandb_uses_api_key_by_default(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            self.assertTrue(open_weights.should_enable_wandb())

    def test_should_enable_wandb_honors_disabled_mode(self):
        with mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": "test-key", "WANDB_MODE": "disabled"},
            clear=False,
        ):
            self.assertFalse(open_weights.should_enable_wandb())

    def test_normalize_chat_body_translates_openai_batch_fields(self):
        normalized = open_weights.normalize_chat_body(
            {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 17,
                "metadata": {"grammar_name": "abc"},
                "store": True,
                "n": 1,
            },
            model_override="google/gemma-3-12b-it",
        )

        self.assertEqual("google/gemma-3-12b-it", normalized["model"])
        self.assertEqual(17, normalized["max_tokens"])
        self.assertNotIn("max_completion_tokens", normalized)
        self.assertNotIn("metadata", normalized)
        self.assertNotIn("store", normalized)

    def test_default_output_path_appends_model_override(self):
        output_path = open_weights.default_output_path(
            "batches/wordorder_exp/inputs_wordorder_gpt-5.jsonl",
            model_override="google/gemma-3-12b-it",
        )

        self.assertEqual(
            Path(
                "batches/wordorder_exp/"
                "inputs_wordorder_gpt-5_google_gemma-3-12b-it_output.jsonl"
            ),
            output_path,
        )

    def test_output_record_is_compatible_with_error_analysis_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            output_path = batch_dir / "local_output.jsonl"
            output_path.write_text(
                json.dumps(
                    open_weights.build_success_record(
                        "grammar-a-sample-0",
                        {
                            "model": "google/gemma-3-12b-it",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Final answer: foo bar",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 11,
                                "completion_tokens": 5,
                                "total_tokens": 16,
                            },
                        },
                    )
                )
                + "\n"
            )

            df = error_analysis.load_outputs(batch_dir, "wordorder")

            self.assertEqual("grammar-a-sample-0", df.loc[0, "custom_id"])
            self.assertEqual("google/gemma-3-12b-it", df.loc[0, "model"])
            self.assertEqual("google/gemma-3-12b-it", df.loc[0, "fuzzy_model"])
            self.assertEqual("foo bar", df.loc[0, "model_answer"])
            self.assertEqual(16, df.loc[0, "total_tokens"])


if __name__ == "__main__":
    unittest.main()
