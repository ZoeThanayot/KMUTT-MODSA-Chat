from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modsa_rag.config import get_settings
from modsa_rag.ingest import ingest_sources
from modsa_rag.rag import answer_question


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, object]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ingestion = ingest_sources(settings)
    yield


app = FastAPI(
    title="MOD-SA RAG API",
    description="Simple LangChain + Chroma RAG API for the KMUTT MOD-SA chatbot.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "ingestion": getattr(app.state, "ingestion", None),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, object]:
    settings = get_settings()
    try:
        app.state.ingestion = ingest_sources(settings)
        return answer_question(settings, request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reindex")
def reindex() -> dict[str, object]:
    settings = get_settings()
    result = ingest_sources(settings, force=True)
    app.state.ingestion = result
    return result
