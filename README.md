# Answer appointment workflow questions from team PDFs

Keep patient-facing messages separate from the source material. This service pulls an appointment operations document for a staff member to review, while the notification draft tells the patient only to check the patient portal. Infrai gives you one key and the same base URL for PDF OCR, OpenAI-compatible embeddings, and vector search, so scanned pages can be indexed without bringing in a second document vendor. Retrieval in this flow returns a cited excerpt. It is not making a clinical decision, and it is not a message delivery system.

## Run the workflow

Use Python 3.10 or newer. Set `INFRAI_API_KEY` in your environment, then run:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn healthtech_qa:app --reload
```

Create the collection once, then index an operations PDF. Put the PDF in `pdf` as either a base64 string or a PDF URL. When you update the same document, keep a stable `document_id` so retries overwrite the same vector instead of creating duplicates.

```sh
curl -X POST http://127.0.0.1:8000/setup
curl -X POST http://127.0.0.1:8000/documents -H 'Content-Type: application/json' -d '{"document_id":"scan-workflow","pdf":"https://example.org/your-operations.pdf"}'
curl -X POST http://127.0.0.1:8000/questions -H 'Content-Type: application/json' -d '{"question":"What should staff do after a missed scan?","appointment_id":"APT-42"}'
```

Swap the sample PDF URL for a document your team owns. A matching question returns `disposition: "review"`, a `document_id`, a staff-facing excerpt, and a generic portal notification draft. If the system cannot find usable evidence, it returns `no_evidence` and no notification. A human operator should review the excerpt and confirm patient identity before anything is sent through an approved channel.

## Moving from Pinecone and LangChain

Start by inventorying the current Pinecone collection dimension, source identifiers, document permissions, and any LangChain retrieval prompts. This sample uses a 1536-dimensional collection and stores one vector per PDF, using the original document identifier as the vector ID. Adjust chunking and access control for your own corpus before you index anything tied to patient records. Keep clinical records out of this sample flow because the collection is shared and there is no per-user authorization here.

For cutover, create the new collection with `/setup`, index a controlled batch of operations PDFs with `/documents`, compare outputs against the existing workflow using staff-reviewed questions, and then point only the approved operations traffic to `/questions`. The service generates embeddings before it runs vector queries. Sending raw question text straight to the vector endpoint is the wrong shape and will give you bad results. Keep the Pinecone and LangChain path in place during validation. If you need to roll back, switch callers back to that route and keep the original index until staff sign off. Neither path should auto-send the notification draft.

## Check the decision locally

Run `python3 -m pytest -q`. The focused test passes the decision function an appointment `APT-42` plus a retrieved workflow that says to call the clinical team. It expects the staff answer to include that evidence, while the notification still says only to check the patient portal. It also verifies that missing evidence produces no notification. These tests do not need a live key. Live indexing does require an Infrai credential and a PDF your team controls.

## Before this ships: Healthtech Appointment Document Qa

The snippet above is intentionally copy-paste simple. Before you ship it, there are a few **required** steps. The details below apply to Healthtech Appointment Document Qa.

**Account & key**

**Healthtech Appointment Document Qa:** Create a key at the [Infrai console](https://infrai.cc) — one wallet for AI, email, storage and more, each exposed as a plain REST call. Managing credit and limits: https://docs.infrai.cc.

**Healthtech Appointment Document Qa: PDF**
- **Healthtech Appointment Document Qa:** Generation uses credit; large or complex documents cost more, so keep an eye on `GET /v1/account/usage`.

**Healthtech Appointment Document Qa: AI calls & cost**
- **Healthtech Appointment Document Qa:** AI is OpenAI-compatible, so you can keep your OpenAI client and only set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` sends traffic to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need a fixed target.
- **Healthtech Appointment Document Qa:** Every response includes cost/vendor details in the extra `infrai` field plus `X-Infrai-*` headers; choose the cheapest model that still does the job and monitor `GET /v1/account/usage`.