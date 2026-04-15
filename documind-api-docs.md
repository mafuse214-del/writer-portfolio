# DocuMind AI API Documentation

## Document Intelligence & RAG-as-a-Service

**Version:** 1.0.0 | **Base URL:** `https://api.documind.ai/v1`

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Quickstart](#quickstart)
4. [Core Endpoints](#core-endpoints)
   - [Document Upload & Processing](#document-upload--processing)
   - [RAG Queries](#rag-queries)
   - [Vector Operations](#vector-operations)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Code Examples](#code-examples)

---

## Overview

DocuMind AI provides a complete RAG (Retrieval-Augmented Generation) infrastructure as a service. Upload your documents, and our API handles chunking, embedding, indexing, and intelligent retrieval—so you can build AI applications with contextual awareness without managing vector databases.

### Key Features

| Feature | Description |
|---------|-------------|
| **Smart Chunking** | Automatic document segmentation with semantic boundaries |
| **Hybrid Search** | Combines dense (semantic) and sparse (keyword) retrieval |
| **Multi-Format Support** | PDF, DOCX, TXT, MD, HTML, JSON |
| **Real-time Indexing** | Documents searchable within seconds of upload |

### Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Your App  │────▶│ DocuMind API │────▶│ Vector Store │
│             │     │  (Processing)│     │              │
└─────────────┘     └──────────────┘     └──────────────┘
       │                    │
       │                    ▼
       │            ┌──────────────┐
       └────────────│ LLM Response │
                    └──────────────┘
```

---

## Authentication

All API requests require an authentication token. Include it in the `Authorization` header:

```http
Authorization: Bearer dm_live_abc123xyz789...
```

### Getting Your API Key

1. Sign up at [dashboard.documind.ai](https://dashboard.documind.ai)
2. Navigate to **Settings > API Keys**
3. Click **Create New Key**
4. Choose environment: `dm_live_` (production) or `dm_test_` (sandbox)

### Security Best Practices

- 🔒 Never commit API keys to version control
- 🔒 Use environment variables: `DOCUMIND_API_KEY`
- 🔒 Rotate keys every 90 days
- 🔒 Use test keys (`dm_test_`) for development—no charges

---

## Quickstart

### Step 1: Upload a Document

```bash
curl -X POST https://api.documind.ai/v1/documents \
  -H "Authorization: Bearer $DOCUMIND_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@quarterly-report.pdf" \
  -F "metadata={\"department\": \"finance\", \"fiscal_year\": 2024}"
```

**Response:**
```json
{
  "id": "doc_8Kx9mNpQrStUvWxY",
  "status": "processing",
  "filename": "quarterly-report.pdf",
  "size_bytes": 245760,
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "department": "finance",
    "fiscal_year": 2024
  },
  "chunks_created": null,
  "processing_status": {
    "stage": "uploading",
    "progress_percent": 15
  }
}
```

### Step 2: Wait for Processing (Optional)

For documents under 10MB, processing typically completes in <5 seconds. For larger files, poll the status:

```bash
curl -X GET https://api.documind.ai/v1/documents/doc_8Kx9mNpQrStUvWxY \
  -H "Authorization: Bearer $DOCUMIND_API_KEY"
```

### Step 3: Query Your Documents

```bash
curl -X POST https://api.documind.ai/v1/query \
  -H "Authorization: Bearer $DOCUMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the main revenue drivers in Q4?",
    "document_ids": ["doc_8Kx9mNpQrStUvWxY"],
    "top_k": 3,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "answer": "According to the Q4 report, the main revenue drivers were:\n\n1. Enterprise subscription growth (up 34% YoY)\n2. New partnerships in the healthcare sector\n3. International expansion, particularly in APAC markets\n\nThe report notes that enterprise customers contributed $2.3M in recurring revenue during the quarter.",
  "sources": [
    {
      "document_id": "doc_8Kx9mNpQrStUvWxY",
      "chunk_id": "chunk_a1b2c3d4",
      "page_number": 7,
      "content": "Enterprise subscription revenue grew 34% year-over-year...",
      "relevance_score": 0.94
    },
    {
      "document_id": "doc_8Kx9mNpQrStUvWxY",
      "chunk_id": "chunk_e5f6g7h8",
      "page_number": 12,
      "content": "Healthcare partnerships contributed an additional $450K...",
      "relevance_score": 0.87
    }
  ],
  "query_id": "qry_9Zy8xWvUtSrQpOnM",
  "processing_time_ms": 342
}
```

---

## Core Endpoints

### Document Upload & Processing

#### POST `/v1/documents`

Upload a document for processing and indexing.

**Request:**
```http
POST /v1/documents HTTP/1.1
Host: api.documind.ai
Authorization: Bearer dm_live_...
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Document to upload (max 50MB) |
| `metadata` | JSON String | No | Key-value pairs for filtering later |
| `chunk_size` | Integer | No | Tokens per chunk (default: 512, max: 2048) |
| `chunk_overlap` | Integer | No | Overlap between chunks (default: 64) |

**Supported File Types:**

| Extension | MIME Type | Max Size |
|-----------|-----------|----------|
| `.pdf` | `application/pdf` | 50 MB |
| `.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | 50 MB |
| `.txt` | `text/plain` | 10 MB |
| `.md` | `text/markdown` | 10 MB |
| `.html`, `.htm` | `text/html` | 5 MB |

**Success Response (201 Created):**
```json
{
  "id": "doc_abc123",
  "status": "processing",
  "filename": "example.pdf",
  "size_bytes": 123456,
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {},
  "chunks_created": null
}
```

---

#### GET `/v1/documents/{document_id}`

Retrieve document details and processing status.

**Request:**
```http
GET /v1/documents/doc_abc123 HTTP/1.1
Host: api.documind.ai
Authorization: Bearer dm_live_...
```

**Response (200 OK):**
```json
{
  "id": "doc_abc123",
  "status": "ready",
  "filename": "quarterly-report.pdf",
  "size_bytes": 245760,
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "department": "finance"
  },
  "chunks_created": 23,
  "processing_status": {
    "stage": "complete",
    "progress_percent": 100
  }
}
```

**Document Statuses:**

| Status | Description |
|--------|-------------|
| `uploading` | File is being received |
| `processing` | Document is being parsed and chunked |
| `indexing` | Chunks are being embedded and indexed |
| `ready` | Document is searchable |
| `failed` | Processing failed (check `error_message`) |

---

#### DELETE `/v1/documents/{document_id}`

Permanently delete a document and all its chunks.

**Request:**
```http
DELETE /v1/documents/doc_abc123 HTTP/1.1
Host: api.documind.ai
Authorization: Bearer dm_live_...
```

**Response (204 No Content):**
```
(Empty body on success)
```

---

### RAG Queries

#### POST `/v1/query`

Perform a RAG query against your indexed documents.

**Request:**
```http
POST /v1/query HTTP/1.1
Host: api.documind.ai
Authorization: Bearer dm_live_...
Content-Type: application/json
```

**Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | String | Yes | - | The question to answer |
| `document_ids` | Array[String] | No | All | Filter to specific documents |
| `metadata_filter` | Object | No | {} | Filter by metadata (e.g., `{"department": "finance"}`) |
| `top_k` | Integer | No | 3 | Number of chunks to retrieve |
| `temperature` | Float | No | 0.7 | LLM temperature (0.0-2.0) |
| `system_prompt` | String | No | Default | Custom system prompt override |
| `include_sources` | Boolean | No | true | Include source chunks in response |

**Example Request:**
```json
{
  "query": "What is the return policy for international orders?",
  "metadata_filter": {"category": "customer_service"},
  "top_k": 5,
  "temperature": 0.3
}
```

**Example Response:**
```json
{
  "answer": "International orders can be returned within 30 days of delivery...",
  "sources": [...],
  "query_id": "qry_xyz789",
  "processing_time_ms": 421,
  "usage": {
    "tokens_used": 156,
    "chunks_retrieved": 3
  }
}
```

---

#### POST `/v1/query/stream`

Stream RAG responses token-by-token for real-time display.

**Request:**
```http
POST /v1/query/stream HTTP/1.1
Host: api.documind.ai
Authorization: Bearer dm_live_...
Content-Type: application/json
```

**Response (Server-Sent Events):**
```
data: {"type": "chunk", "content": "International", "source_hint": null}

data: {"type": "chunk", "content": " orders", "source_hint": null}

data: {"type": "sources_found", "count": 3}

data: {"type": "chunk", "content": " can", "source_hint": "doc_abc123"}

[data: {"type": "done"}]
```

---

### Vector Operations

#### POST `/v1/vectors/search`

Raw vector similarity search (for advanced use cases).

**Request:**
```json
{
  "query_vector": [0.023, -0.018, 0.042, ...],
  "top_k": 10,
  "filter": {"metadata_key": "metadata_value"}
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "chunk_abc123",
      "document_id": "doc_xyz789",
      "content": "...",
      "distance": 0.234,
      "metadata": {}
    }
  ]
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|" |
| 200 | Success |
| 201 | Created (document uploaded) |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (missing/invalid token) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found (resource doesn't exist) |
| 429 | Rate Limited (too many requests) |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": {
    "code": "document_not_found",
    "message": "Document 'doc_invalid123' does not exist or has been deleted.",
    "details": {
      "document_id": "doc_invalid123"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### Error Codes Reference

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_api_key` | 401 | API key is malformed or expired |
| `insufficient_quota` | 403 | Account has exceeded its quota |
| `document_not_found` | 404 | Document ID doesn't exist |
| `document_processing_failed` | 400 | Document couldn't be parsed |
| `invalid_query_parameters` | 400 | Query parameters are invalid |
| `rate_limit_exceeded` | 429 | Too many requests |

---

## Rate Limiting

### Limits by Plan

| Plan | Requests/Minute | Requests/Month | Max Document Size |
|------|-----------------|----------------|------------------|
| Free | 60 | 1,000 | 10 MB |
| Pro | 300 | 50,000 | 50 MB |
| Enterprise | Custom | Custom | 100 MB |

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1705312800
```

### Handling Rate Limits

When you exceed the limit, you'll receive a `429` response:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 15 seconds.",
    "retry_after_seconds": 15
  }
}
```

**Recommended Strategy:** Implement exponential backoff with jitter.

---

## Code Examples

### Python

```python
import requests

# Configuration
API_KEY = "dm_live_abc123"
BASE_URL = "https://api.documind.ai/v1"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Upload a document
def upload_document(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{BASE_URL}/documents",
            headers=headers,
            files=files
        )
    return response.json()

# Query documents
def query_documents(query: str, top_k: int = 3) -> dict:
    response = requests.post(
        f"{BASE_URL}/query",
        headers={**headers, "Content-Type": "application/json"},
        json={"query": query, "top_k": top_k}
    )
    return response.json()

# Stream a response
def stream_query(query: str):
    response = requests.post(
        f"{BASE_URL}/query/stream",
        headers={**headers, "Content-Type": "application/json"},
        json={"query": query},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8').replace('data: ', '')
            if data != '[DONE]':
                event = json.loads(data)
                if event.get('type') == 'chunk':
                    print(event.get('content'), end='', flush=True)

# Usage
doc_response = upload_document("quarterly-report.pdf")
print(f"Document ID: {doc_response['id']}")

answer = query_documents("What were Q4 revenues?")
print(answer['answer'])
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

const API_KEY = 'dm_live_abc123';
const BASE_URL = 'https://api.documind.ai/v1';

const headers = {
  'Authorization': `Bearer ${API_KEY}`
};

// Upload a document
async function uploadDocument(filePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(filePath));
  
  const response = await fetch(`${BASE_URL}/documents`, {
    method: 'POST',
    headers: headers,
    body: formData
  });
  
  return await response.json();
}

// Query documents
async function queryDocuments(query, topK = 3) {
  const response = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK })
  });
  
  return await response.json();
}

// Usage
(async () => {
  const doc = await uploadDocument('quarterly-report.pdf');
  console.log(`Document ID: ${doc.id}`);
  
  const answer = await queryDocuments('What were Q4 revenues?');
  console.log(answer.answer);
})();
```

### cURL (Streaming)

```bash
# Stream response to terminal
curl -s -X POST https://api.documind.ai/v1/query/stream \
  -H "Authorization: Bearer $DOCUMIND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company mission?"}' | \
  jq -r 'select(.type == "chunk") | .content' | tr -d '\n'
```

---

## Appendix A: SDK Installation

### Python SDK

```bash
pip install documind-ai
```

```python
from documind_ai import DocuMindClient

client = DocuMindClient(api_key="dm_live_abc123")

# Upload and query in two lines
doc_id = client.ingest.file("quarterly-report.pdf")
answer = client.query("What were Q4 revenues?", document_ids=[doc_id])
print(answer.text)
```

### JavaScript SDK

```bash
npm install @documind/sdk
```

```javascript
import { DocuMindClient } from '@documind/sdk';

const client = new DocuMindClient({ apiKey: 'dm_live_abc123' });

const docId = await client.ingest.file('quarterly-report.pdf');
const answer = await client.query('What were Q4 revenues?', { documentIds: [docId] });
console.log(answer.text);
```

---

## Appendix B: Changelog

| Version | Date | Changes |
|---------|------|----------|
| 1.0.0 | 2024-01-15 | Initial release |

---

**Need Help?** Contact support at [support@documind.ai](mailto:support@documind.ai) or visit our [Community Forum](https://community.documind.ai).
