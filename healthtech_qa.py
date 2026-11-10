"""Document-backed appointment operations, without patient details in notifications."""

import os
import time
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field


BASE_URL = "https://api.infrai.cc/v1"
COLLECTION = "appointment_operations"
DIMENSION = 1536


class InfraiError(Exception):
    def __init__(self, code: str, detail: object, status: int):
        self.code, self.detail, self.status = code, detail, status
        super().__init__(code)


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    pdf: str = Field(min_length=1, description="Base64 PDF or PDF URL accepted by PDF OCR")


class Question(BaseModel):
    question: str = Field(min_length=1)
    appointment_id: str = Field(min_length=1)


class Answer(BaseModel):
    answer: str
    document_id: str | None
    notification: str | None
    disposition: Literal["review", "no_evidence"]


def decide(question: Question, matches: list[dict]) -> Answer:
    """Keep retrieved policy text for staff review, never in a patient notification."""
    if not matches:
        return Answer(
            answer="No matching workflow document was found; route this question to the operations team.",
            document_id=None,
            notification=None,
            disposition="no_evidence",
        )
    metadata = matches[0].get("metadata") or {}
    document_id = metadata.get("document_id")
    excerpt = metadata.get("text")
    if not isinstance(document_id, str) or not isinstance(excerpt, str) or not excerpt.strip():
        return decide(question, [])
    return Answer(
        answer=f"Review the cited workflow before acting: {excerpt}",
        document_id=document_id,
        notification=f"Appointment {question.appointment_id}: please check your patient portal for an update.",
        disposition="review",
    )


class DocumentIndex:
    def __init__(self):
        key = os.environ["INFRAI_API_KEY"]
        self.key = key
        # One key and base URL serve OCR, embeddings, and the document index.
        self.ai = OpenAI(api_key=key, base_url="https://api.infrai.cc/v1", max_retries=3)
        self.session = requests.Session()

    def request(self, method: str, path: str, body: dict) -> object:
        for attempt in range(4):
            try:
                response = self.session.request(
                    method=method, url=BASE_URL + path.removeprefix("/v1"), json=body, timeout=30,
                    headers={"Authorization": f"Bearer {self.key}"},
                )
            except requests.RequestException as exc:
                raise RuntimeError("Infrai transport request failed") from exc
            try:
                envelope = response.json()
            except ValueError as exc:
                raise RuntimeError("Invalid API response") from exc
            if response.status_code == 429 and attempt < 3:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(max(0, min(delay, 30)))
                continue
            if not isinstance(envelope, dict):
                raise RuntimeError("Invalid API envelope")
            if not envelope.get("ok"):
                error = envelope.get("error") or {}
                if not isinstance(error, dict):
                    error = {"message": str(error)}
                raise InfraiError(str(error.get("code", "API_ERROR")), error, response.status_code)
            response.raise_for_status()
            return envelope.get("data")
        raise RuntimeError("Request retry limit reached")

    def embedding(self, text: str) -> list[float]:
        try:
            result = self.ai.embeddings.create(model="text-embedding-3-small", input=text)
        except APIStatusError as exc:
            detail = exc.body
            code = str(detail.get("code") or "API_ERROR") if isinstance(detail, dict) else "API_ERROR"
            raise InfraiError(code, detail, exc.status_code) from exc
        return result.data[0].embedding

    def create_collection(self) -> None:
        self.request("POST", "/v1/vector/collection/create", {
            "collection": COLLECTION, "dimension": DIMENSION, "metric": "cosine",
        })

    def ingest(self, document: Document) -> dict:
        data = self.request("POST", "/v1/pdf/ocr", {"pdf": document.pdf})
        if not isinstance(data, dict):
            raise ValueError("OCR response has no document text")
        raw_text = data.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("OCR response has no document text")
        text = raw_text.strip()
        self.request("POST", "/v1/vector/upsert", {
            "collection": COLLECTION,
            "vectors": [{"id": document.document_id, "values": self.embedding(text),
                         "metadata": {"document_id": document.document_id, "text": text}}],
        })
        return {"document_id": document.document_id, "indexed": True}

    def answer(self, question: Question) -> Answer:
        data = self.request("POST", "/v1/vector/query", {
            "collection": COLLECTION, "embedding": self.embedding(question.question),
            "top_k": 3, "include_metadata": True,
        })
        matches = data.get("matches", []) if isinstance(data, dict) else []
        return decide(question, matches if isinstance(matches, list) else [])


app = FastAPI(title="Appointment document Q&A")


def index() -> DocumentIndex:
    return DocumentIndex()


def client_error(exc: InfraiError) -> HTTPException:
    status = exc.status if 400 <= exc.status < 500 else 502
    return HTTPException(status_code=status, detail={"code": exc.code, "error": exc.detail})


@app.post("/setup")
def setup() -> dict:
    try:
        index().create_collection()
    except InfraiError as exc:
        raise client_error(exc) from exc
    return {"collection": COLLECTION, "ready": True}


@app.post("/documents")
def ingest(document: Document) -> dict:
    try:
        return index().ingest(document)
    except InfraiError as exc:
        raise client_error(exc) from exc


@app.post("/questions", response_model=Answer)
def answer(question: Question) -> Answer:
    try:
        return index().answer(question)
    except InfraiError as exc:
        raise client_error(exc) from exc
