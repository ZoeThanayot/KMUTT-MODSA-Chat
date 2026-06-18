from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.errors import NotFoundError
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modsa_rag.config import Settings


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
MANIFEST_FILE = "source_manifest.json"


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_uses_ollama:
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.resolved_embedding_base_url,
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.resolved_embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def get_vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_dir),
        embedding_function=build_embeddings(settings),
    )


def discover_source_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(child)
    return sorted(files)


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def build_manifest(files: list[Path]) -> dict[str, object]:
    return {
        "files": [file_fingerprint(path) for path in files],
    }


def manifest_path(settings: Settings) -> Path:
    return settings.chroma_dir / MANIFEST_FILE


def load_manifest(settings: Settings) -> dict[str, object] | None:
    path = manifest_path(settings)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(settings: Settings, manifest: dict[str, object]) -> None:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(settings).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_documents(files: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = str(path)
            documents.append(document)
    return documents


def split_documents(settings: Settings, documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def reset_collection(settings: Settings) -> None:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.chroma_collection)
    except (ValueError, NotFoundError):
        pass


def ingest_sources(settings: Settings, force: bool = False) -> dict[str, object]:
    files = discover_source_files(settings.source_paths)
    current_manifest = build_manifest(files)
    previous_manifest = load_manifest(settings)

    if not force and previous_manifest == current_manifest:
        return {
            "status": "skipped",
            "reason": "source files unchanged",
            "files": len(files),
        }

    reset_collection(settings)

    if not files:
        save_manifest(settings, current_manifest)
        return {
            "status": "empty",
            "reason": "no supported source files found",
            "files": 0,
            "chunks": 0,
        }

    documents = load_documents(files)
    chunks = split_documents(settings, documents)
    vector_store = get_vector_store(settings)
    vector_store.add_documents(chunks)
    save_manifest(settings, current_manifest)

    return {
        "status": "indexed",
        "files": len(files),
        "documents": len(documents),
        "chunks": len(chunks),
    }
