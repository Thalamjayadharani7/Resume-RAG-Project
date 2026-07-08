import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.data_processing.embedding import EmbeddedChunk
from src.data_processing.vector_store import VectorStore
from src.rag.llm import GeminiClient
from src.rag.retriever import Retriever
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

    def test_reset_collection_clears_previous_embeddings(self):
        class DummyCollection:
            def __init__(self):
                self.count_calls = 0
                self.docs = []

            def count(self):
                self.count_calls += 1
                return len(self.docs)

            def add(self, **kwargs):
                self.docs.extend(kwargs.get("documents", []))

            def get(self, include=None):
                return {"metadatas": []}

        class DummyClient:
            def __init__(self):
                self.collections = {}

            def delete_collection(self, name):
                self.collections.pop(name, None)

            def get_or_create_collection(self, name, metadata=None):
                if name not in self.collections:
                    self.collections[name] = DummyCollection()
                return self.collections[name]

        class DummyChroma:
            @staticmethod
            def PersistentClient(path=None):
                return DummyClient()

            @staticmethod
            def Client():
                return DummyClient()

        with patch("src.data_processing.vector_store.chromadb", DummyChroma()):
            store = VectorStore(collection_name="test_resume_chunks", persist_directory=".")
            store.create_collection()
            store.add_documents(
                [
                    EmbeddedChunk(
                        filename="sample.pdf",
                        chunk_id="sample__0",
                        chunk_text="sample chunk",
                        embedding_vector=[0.1, 0.2],
                        candidate_name="Dharani",
                        page_number=1,
                    )
                ]
            )

            self.assertEqual(store.collection.count(), 1)

            store.reset_collection()

            self.assertEqual(store.collection.count(), 0)

    def test_retriever_filters_by_candidate_name_or_filename(self):
        class FakeVectorStore:
            def __init__(self):
                self.last_where = None

            def create_collection(self):
                return None

            def get_all_metadata(self):
                return []

            def similarity_search(self, query_embedding, top_k=5, where=None):
                self.last_where = where
                return []

        class FakeEmbeddingGenerator:
            def load_model(self):
                return type("Model", (), {"encode": lambda self, texts, convert_to_numpy, normalize_embeddings: [[0.1, 0.2] for _ in texts]})()

        fake_store = FakeVectorStore()
        retriever = Retriever(embedding_generator=FakeEmbeddingGenerator(), vector_store=fake_store)

        retriever.retrieve_context("What are Dharani's skills?", resume_name="Dharani")

        self.assertIsNotNone(fake_store.last_where)
        self.assertIn("$or", str(fake_store.last_where))

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
