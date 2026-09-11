import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "gemini_batch.py"
SPEC = importlib.util.spec_from_file_location("gemini_batch_module", MODULE_PATH)
assert SPEC and SPEC.loader
gemini_batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gemini_batch
SPEC.loader.exec_module(gemini_batch)

ERROR_ANALYSIS_PATH = ROOT / "notebooks" / "error_analysis.py"
ERROR_SPEC = importlib.util.spec_from_file_location(
    "notebooks_error_analysis_for_gemini_batch",
    ERROR_ANALYSIS_PATH,
)
assert ERROR_SPEC and ERROR_SPEC.loader
error_analysis = importlib.util.module_from_spec(ERROR_SPEC)
sys.modules[ERROR_SPEC.name] = error_analysis
ERROR_SPEC.loader.exec_module(error_analysis)


def sample_gemini_response(text: str = "Final answer: foo bar") -> dict:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 150,
        },
        "modelVersion": "gemini-3.8-flash",
        "responseId": "abc123",
    }


class RequestConversionTest(unittest.TestCase):
    def test_user_message_becomes_gemini_contents(self):
        body = {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "translate this"}],
            "max_completion_tokens": None,
            "n": 1,
        }
        request = gemini_batch.openai_body_to_gemini_request(body)
        self.assertEqual(
            {"contents": [{"role": "user", "parts": [{"text": "translate this"}]}]},
            request,
        )

    def test_system_and_assistant_messages_and_generation_config(self):
        body = {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
            "max_completion_tokens": 256,
            "n": 2,
            "temperature": 0.5,
        }
        request = gemini_batch.openai_body_to_gemini_request(body)
        self.assertEqual(
            {"parts": [{"text": "be terse"}]}, request["systemInstruction"]
        )
        self.assertEqual(
            ["user", "model", "user"], [c["role"] for c in request["contents"]]
        )
        self.assertEqual(
            {"maxOutputTokens": 256, "candidateCount": 2, "temperature": 0.5},
            request["generationConfig"],
        )

    def test_vertex_input_lines_carry_custom_id(self):
        requests = [
            gemini_batch.BatchRequest(
                custom_id="g-sample-0",
                body={"messages": [{"role": "user", "content": "x"}]},
            )
        ]
        (line,) = gemini_batch.build_vertex_input_lines(requests)
        self.assertIn('"custom_id": "g-sample-0"', line)
        self.assertIn('"request": {"contents"', line)


class ResponseConversionTest(unittest.TestCase):
    def test_success_record_matches_analysis_expectations(self):
        line = {"custom_id": "g-sample-3", "response": sample_gemini_response()}
        record = gemini_batch.convert_prediction_line(
            line, model="google/gemini-3.8-flash"
        )
        self.assertEqual("g-sample-3", record["custom_id"])
        self.assertEqual(200, record["response"]["status_code"])
        body = record["response"]["body"]
        self.assertEqual("google/gemini-3.8-flash", body["model"])
        self.assertEqual("stop", body["choices"][0]["finish_reason"])
        self.assertEqual(
            "Final answer: foo bar", body["choices"][0]["message"]["content"]
        )
        # Matches the shape produced by the online OpenAI-compatible endpoint.
        self.assertEqual((100, 10, 150), error_analysis.usage_tuple(body))
        self.assertEqual(
            40, body["usage"]["completion_tokens_details"]["reasoning_tokens"]
        )
        self.assertEqual(
            "foo bar",
            error_analysis.extract_answer(body["choices"][0]["message"]["content"]),
        )

    def test_thought_parts_are_excluded_from_content(self):
        response = sample_gemini_response()
        response["candidates"][0]["content"]["parts"].insert(
            0, {"text": "thinking...", "thought": True}
        )
        body = gemini_batch.gemini_response_to_openai_body(response, model="m")
        self.assertEqual(
            "Final answer: foo bar", body["choices"][0]["message"]["content"]
        )

    def test_status_string_becomes_error_record(self):
        line = {"custom_id": "g-sample-1", "status": "Internal error", "response": None}
        record = gemini_batch.convert_prediction_line(line, model="m")
        self.assertEqual(500, record["response"]["status_code"])
        self.assertEqual([], record["response"]["body"]["choices"])
        self.assertIn("Internal error", record["response"]["body"]["error"]["message"])

    def test_missing_candidates_becomes_error_record(self):
        response = sample_gemini_response()
        response["candidates"] = []
        response["promptFeedback"] = {"blockReason": "SAFETY"}
        record = gemini_batch.convert_prediction_line(
            {"custom_id": "g-sample-2", "response": response}, model="m"
        )
        self.assertEqual(400, record["response"]["status_code"])
        self.assertIn("SAFETY", record["response"]["body"]["error"]["message"])

    def test_custom_id_falls_back_to_request_text(self):
        request = gemini_batch.openai_body_to_gemini_request(
            {"messages": [{"role": "user", "content": "unique prompt"}]}
        )
        lookup = {gemini_batch.request_text_key(request): "g-sample-9"}
        record = gemini_batch.convert_prediction_line(
            {"request": request, "response": sample_gemini_response()},
            model="m",
            custom_id_by_text=lookup,
        )
        self.assertEqual("g-sample-9", record["custom_id"])


if __name__ == "__main__":
    unittest.main()
