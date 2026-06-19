from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path

from langchain_core.documents import Document

from modsa_rag.config import Settings
from modsa_rag import rag
from modsa_rag import ingest


class RagFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            LLM_BASE_URL="http://example.com/v1",
            LLM_API_KEY="test-key",
            LLM_MODEL="test-model",
            EMBEDDING_BASE_URL="http://localhost:11434",
            EMBEDDING_MODEL="test-embedding",
            CHROMA_DIR="chroma_db",
            RETRIEVAL_K=4,
            RETRIEVAL_MIN_RELEVANCE=0.35,
        )

    def test_empty_retrieval_skips_llm(self) -> None:
        original_retrieve = rag.retrieve_documents
        original_build_llm = rag.build_llm
        try:
            rag.retrieve_documents = lambda settings, question: []
            rag.build_llm = lambda settings: self.fail("LLM should not be called when retrieval is empty")

            result = rag.answer_question(self.settings, "วันไหว้ครูคือวันไหน")

            self.assertEqual(result["sources"], [])
            self.assertIn("ไม่พบข้อมูลยืนยัน", result["answer"])
        finally:
            rag.retrieve_documents = original_retrieve
            rag.build_llm = original_build_llm

    def test_retrieved_documents_are_returned_as_sources(self) -> None:
        original_retrieve = rag.retrieve_documents
        original_build_llm = rag.build_llm
        doc = Document(page_content="sample content", metadata={"source": "knowledge/sample.pdf", "page": 0})

        class FakeLLM:
            def invoke(self, messages):
                return SimpleNamespace(content="ตอบจากบริบท")

        try:
            rag.retrieve_documents = lambda settings, question: [doc]
            rag.build_llm = lambda settings: FakeLLM()

            result = rag.answer_question(self.settings, "sample question")

            self.assertEqual(result["answer"], "ตอบจากบริบท")
            self.assertEqual(result["sources"], [{"source": "knowledge/sample.pdf", "page": 1}])
        finally:
            rag.retrieve_documents = original_retrieve
            rag.build_llm = original_build_llm

    def test_empty_collection_forces_reindex_even_when_manifest_matches(self) -> None:
        original_discover_source_files = ingest.discover_source_files
        original_build_manifest = ingest.build_manifest
        original_load_manifest = ingest.load_manifest
        original_collection_document_count = ingest.collection_document_count
        original_reset_collection = ingest.reset_collection
        original_load_documents = ingest.load_documents
        original_split_documents = ingest.split_documents
        original_get_vector_store = ingest.get_vector_store
        original_save_manifest = ingest.save_manifest

        calls: dict[str, int] = {"added": 0, "reset": 0, "saved": 0}
        doc = Document(page_content="sample content", metadata={"source": "knowledge/sample.pdf", "page": 0})

        class FakeStore:
            def add_documents(self, chunks):
                calls["added"] = len(chunks)

        try:
            ingest.discover_source_files = lambda paths: [Path("knowledge/sample.pdf")]
            ingest.build_manifest = lambda files: {"files": [{"path": "knowledge/sample.pdf"}]}
            ingest.load_manifest = lambda settings: {"files": [{"path": "knowledge/sample.pdf"}]}
            ingest.collection_document_count = lambda settings: 0
            ingest.reset_collection = lambda settings: calls.__setitem__("reset", calls["reset"] + 1)
            ingest.load_documents = lambda files: ([doc], [])
            ingest.split_documents = lambda settings, documents: documents
            ingest.get_vector_store = lambda settings: FakeStore()
            ingest.save_manifest = lambda settings, manifest: calls.__setitem__("saved", calls["saved"] + 1)

            result = ingest.ingest_sources(self.settings)

            self.assertEqual(result["status"], "indexed")
            self.assertEqual(calls["reset"], 1)
            self.assertEqual(calls["added"], 1)
            self.assertEqual(calls["saved"], 1)
        finally:
            ingest.discover_source_files = original_discover_source_files
            ingest.build_manifest = original_build_manifest
            ingest.load_manifest = original_load_manifest
            ingest.collection_document_count = original_collection_document_count
            ingest.reset_collection = original_reset_collection
            ingest.load_documents = original_load_documents
            ingest.split_documents = original_split_documents
            ingest.get_vector_store = original_get_vector_store
            ingest.save_manifest = original_save_manifest


if __name__ == "__main__":
    unittest.main()
