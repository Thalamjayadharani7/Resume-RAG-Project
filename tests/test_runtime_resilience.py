import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.rag.llm import GeminiClient
import main


class DummyModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if "timeout" in kwargs:
            raise AssertionError("timeout should not be passed to generate_content")
        return type("Response", (), {"text": "ok"})()


class RuntimeResilienceTests(unittest.TestCase):
    def test_gemini_client_uses_sdk_supported_kwargs(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model_name = "gemini-2.5-flash"
        client.timeout_seconds = 30
        client.api_key = "fake-key"
        client._client = type("Client", (), {"models": DummyModels()})()

        response = client.generate_response("hello")

        self.assertEqual(response, "ok")
        self.assertEqual(client._client.models.calls[0].get("model"), "gemini-2.5-flash")
        self.assertNotIn("timeout", client._client.models.calls[0])

    def test_main_exits_cleanly_when_input_is_missing(self):
        class DummyPipeline:
            def build_index(self, data_dir):
                return []

            def list_resume_files(self, data_dir):
                return []

            def answer_question(self, question, resume_name=None):
                return {"answer": "ignored"}

        stdout = io.StringIO()
        with patch.object(main, "RAGPipeline", return_value=DummyPipeline()), patch("builtins.input", side_effect=EOFError("No input received")), patch("main.configure_logging"), redirect_stdout(stdout):
            main.main()

        output = stdout.getvalue()
        self.assertIn("No input received. Exiting.", output)


if __name__ == "__main__":
    unittest.main()
