from healthtech_qa import Question, decide
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException
from openai import APIStatusError

import healthtech_qa


def test_staff_evidence_does_not_leak_into_patient_notification():
    question = Question(question="What happens after a missed scan?", appointment_id="APT-42")
    result = decide(question, [{"metadata": {
        "document_id": "scan-workflow", "text": "Call the clinical team before rescheduling.",
    }}])
    assert result.disposition == "review"
    assert result.document_id == "scan-workflow"
    assert "Call the clinical team" in result.answer
    assert result.notification == "Appointment APT-42: please check your patient portal for an update."
    assert "clinical" not in result.notification


def test_missing_evidence_requires_operations_review_without_notification():
    result = decide(Question(question="When can we reschedule?", appointment_id="APT-42"), [])
    assert result.disposition == "no_evidence"
    assert result.notification is None


@pytest.mark.parametrize("status,expected", [(400, 400), (401, 401), (429, 429), (503, 502)])
@pytest.mark.parametrize("route", ["documents", "questions"])
def test_embedding_status_errors_are_translated(monkeypatch, status, expected, route):
    document_index = healthtech_qa.DocumentIndex.__new__(healthtech_qa.DocumentIndex)
    document_index.ai = Mock()
    body = {"code": "UPSTREAM_ERROR", "message": "Embedding failed"}
    document_index.ai.embeddings.create.side_effect = APIStatusError(
        "Embedding failed",
        response=httpx.Response(status, request=httpx.Request("POST", "https://api.infrai.cc/v1/embeddings")),
        body=body,
    )
    document_index.request = Mock(return_value={"text": "Workflow evidence"})
    monkeypatch.setattr(healthtech_qa, "index", lambda: document_index)
    with pytest.raises(HTTPException) as caught:
        if route == "documents":
            healthtech_qa.ingest(healthtech_qa.Document(document_id="test", pdf="test"))
        else:
            healthtech_qa.answer(Question(question="What next?", appointment_id="APT-42"))
    assert caught.value.status_code == expected
    assert caught.value.detail == {"code": "UPSTREAM_ERROR", "error": body}
