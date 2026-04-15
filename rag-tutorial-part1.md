# Building a RAG System from Scratch

## Part 1: Understanding the Fundamentals

*A hands-on guide to Retrieval-Augmented Generation — bringing your own data into LLM applications.*

---

## TL;DR Summary

**RAG (Retrieval-Augmented Generation)** is an architecture pattern that combines:
1. **Retrieval** — Finding relevant information from your knowledge base
2. **Augmentation** — Adding that context to the user's query
3. **Generation** — Having the LLM answer using both the query AND the retrieved context

This allows you to build AI applications with access to private, proprietary, or domain-specific data without fine-tuning a model.

---

## Table of Contents

1. [What Problem Does RAG Solve?](#what-problem-does-rag-solve)
2. [The RAG Architecture at a Glance](#the-rag-architecture-at-a-glance)
3. [Key Components Deep Dive](#key-components-deep-dive)
4. [When to Use RAG vs Fine-Tuning](#when-to-use-rag-vs-fine-tuning)
5. [System Design Decisions](#system-design-decisions)
6. [Summary & Next Steps](#summary--next-steps)

---

## What Problem Does RAG Solve?

### The Four Challenges of Using LLMs in Production

#### Challenge 1: Knowledge Cutoff

LLMs have a training cutoff date. They don't know:
- Your company's internal documents
- Recent events after their training
- Proprietary information unique to your domain

**Example:**
```python
# GPT-4 (trained on data until ~2023)
question = "What was our Q4 2024 revenue?"
response = llm(question)  # ❌ "I don't have access to that information."
```

#### Challenge 2: Hallucinations

LLMs can confidently generate incorrect information:

```python
question = "What does the --quantum-flag do in our CLI tool?"
response = llm(question)  # ❌ Generates plausible-sounding but fake answer
```

#### Challenge 3: Context Window Limits

Even with large context windows (128K+ tokens), you can't fit everything:
- Your entire codebase?
- All customer support tickets?
- Every research paper in your field?

**You need retrieval to find the RELEVANT pieces.**

#### Challenge 4: Cost & Latency

Sending massive amounts of context on every request is expensive and slow.

---

### The RAG Solution

RAG addresses all four challenges:

| Challenge | RAG Solution |
|-----------|-------------|
| Knowledge Cutoff | Query your own up-to-date data |
| Hallucinations | Ground answers in retrieved sources |
| Context Limits | Retrieve only relevant chunks |
| Cost/Latency | Smaller context = cheaper, faster |

---

## The RAG Architecture at a Glance

### High-Level Diagram

```
                    ┌─────────────────────────────────────┐
                    │         INDEXING PIPELINE           │
                    │                                     │
    Your Documents  │  ┌──────────┐   ┌──────────┐       │
    (PDF, TXT, etc) │  │  Chunk   │──▶│ Embed    │──▶    │
         │          │  │  ing     │   │ ding     │       │
         ▼          │  └──────────┘   └──────────┘       │
                    │         │            │              │
                    │         ▼            ▼              │
                    │    ┌─────────────────────┐         │
                    │    │   Vector Database   │◀────────┘
                    │    │   (Chroma, Pinecone)│
                    │    └─────────────────────┘         │
                    └─────────────────────────────────────┘
                                   ▲
                                   │ Pre-computed
                                   │
                    ┌───────────────┴───────────────────────┐
                    │        QUERY PIPELINE                 │
                    │                                       │
    User Query      │  ┌──────────┐   ┌──────────┐        │
    "What's our     │  │ Embed    │──▶│ Retrieve │        │
    refund policy?" │  │ Query    │   │ Top-K    │        │
         │          │  │          │   │ Chunks   │        │
         └──────────┼──┴──────────┘   └─────┬────┘        │
                    │                       │              │
                    │                       ▼              │
                    │            ┌────────────────┐       │
                    │            │  Construct     │       │
                    │            │  Prompt with   │       │
                    │            │  Context       │       │
                    │            └────────┬───────┘       │
                    │                     │               │
                    │                     ▼               │
                    │            ┌────────────────┐      │
                    │            │    LLM Call    │      │
                    │            │                │      │
                    │            └────────┬───────┘      │
                    │                     │               │
                    └─────────────────────┼───────────────┘
                                          ▼
                                   Final Answer
```

### The Two Phases of RAG

#### Phase 1: Indexing (One-Time Setup)

```python
# Simplified indexing pipeline

documents = load_documents("./knowledge_base/")  # Load PDFs, TXT, etc.
chunks = chunk_documents(documents)               # Split into pieces
embeddings = embed(chunks)                        # Convert to vectors
index = vector_db.create_collection(embeddings)   # Store in vector DB
```

#### Phase 2: Query (Every User Request)

```python
# Simplified query pipeline

def answer_question(query: str) -> str:
    # 1. Embed the query
    query_vector = embed(query)
    
    # 2. Retrieve relevant chunks
    context_chunks = vector_db.search(query_vector, top_k=3)
    
    # 3. Build prompt with context
    prompt = f"""
    Using ONLY the following context, answer the question.
    If the answer isn't in the context, say "I don't know.".
    
    Context:
    {context_chunks}
    
    Question: {query}
    Answer:
    """
    
    # 4. Get LLM response
    return llm.generate(prompt)
```

---

## Key Components Deep Dive

### Component 1: Document Chunking

**What is chunking?** Breaking documents into smaller, manageable pieces for retrieval.

#### Why Not Just Index Whole Documents?

| Issue | Explanation |
|-------|-------------|
| **Lost Precision** | Query about "refund policy" retrieves entire 200-page manual |
| **Context Waste** | Only 1 paragraph is relevant, but you pay for all 50K tokens |
| **Poor Retrieval** | Embedding of whole doc dilutes specific information |

#### Chunking Strategies

**Strategy A: Fixed-Size Chunking (Simplest)**
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap for context continuity
)
chunks = text_splitter.split_text(long_document)
```

**Strategy B: Semantic Chunking (Better)**
```python
# Split by meaningful boundaries:
# - Paragraph breaks
# - Section headers  
# - Sentence boundaries
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]  # Try these in order
)
```

**Strategy C: Document-Aware Chunking (Best for PDFs)**
```python
# Respects document structure:
# - Keeps tables intact
# - Preserves headers with content
# - Doesn't split mid-sentence
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import SemanticChunker

loader = PyPDFLoader("document.pdf")
docs = loader.load()

semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    buffer_words=10,
    min_chunks=5
)
chunks = semantic_splitter.create_documents([doc.page_content for doc in docs])
```

#### Chunk Size Tradeoffs

| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (256-512 tokens) | Precise retrieval, cheaper queries | May lose context, more chunks to manage |
| Medium (512-1024 tokens) | Good balance | — |
| Large (1024+ tokens) | More self-contained context | Expensive, less precise retrieval |

**Recommendation:** Start with 512-768 tokens with 50-100 token overlap.

---

### Component 2: Embeddings

**What are embeddings?** Numerical representations of text that capture semantic meaning.

#### How Embeddings Enable Semantic Search

```python
# Traditional keyword search (fails)
query = "How do I get my money back?"
document = "Our refund policy allows returns within 30 days."
keyword_match(query, document)  # ❌ No common words!

# Semantic search with embeddings (works)
query_vector = embed("How do I get my money back?")
doc_vector = embed("Our refund policy allows returns within 30 days.")
similarity = cosine_similarity(query_vector, doc_vector)  # ✅ 0.87!
```

#### Popular Embedding Models

| Model | Dimensions | Context Window | Best For |
|-------|------------|----------------|----------|
| `text-embedding-3-small` (OpenAI) | 1536 | 8K tokens | General purpose, cheap |
| `text-embedding-3-large` (OpenAI) | 3072 | 8K tokens | High quality needs |
| `all-MiniLM-L6-v2` (Sentence Transformers) | 384 | 512 tokens | Local, free, fast |
| `bge-small-en-v1.5` | 384 | 512 tokens | Local, good quality |

#### Embedding Code Example

```python
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Create embeddings
client = openai.OpenAI()

query_embedding = client.embeddings.create(
    input="How do I reset my password?",
    model="text-embedding-3-small"
).data[0].embedding

doc_embedding = client.embeddings.create(
    input="Users can reset their password by clicking 'Forgot Password' on the login page.",
    model="text-embedding-3-small"
).data[0].embedding

# Calculate similarity (0.0 to 1.0)
similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
print(f"Similarity: {similarity:.3f}")  # Output: Similarity: 0.823
```

---

### Component 3: Vector Database

**What is a vector database?** A specialized database optimized for storing and searching embeddings.

#### Why Not Just Use Python Lists?

```python
# Naive approach (doesn't scale)
def naive_search(query_vector, document_vectors, top_k=5):
    similarities = []
    for i, doc_vector in enumerate(document_vectors):  # O(n) per query!
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((i, sim))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

# With 100K documents:
# - Each comparison takes ~1ms
# - Total time: 100K × 1ms = 100 seconds ❌
```

**Vector databases use Approximate Nearest Neighbor (ANN) algorithms:**
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- PQ (Product Quantization)

These trade a small amount of accuracy for massive speedups.

#### Vector Database Options

| Database | Type | Best For | Pricing |
|----------|------|----------|---------|
| **Chroma** | Local/Embedded | Development, <100K vectors | Free |
| **FAISS** (Facebook) | Library | Custom solutions | Free |
| **Pinecone** | Managed Service | Production, scalability | Paid |
| **Weaviate** | Hybrid | Flexible deployment | Free/Paid |
| **Qdrant** | Managed/Self-hosted | Performance-focused | Free/Paid |

#### Using Chroma (Simplest for Getting Started)

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection("knowledge_base")

# Add documents with embeddings
for i, chunk in enumerate(chunks):
    embedding = embed(chunk)  # Your embedding function
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": "handbook.pdf", "page": 5}]
    )

# Query the collection
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "distances"]
)

print(results["documents"][0])  # Top 3 relevant chunks
```

---

### Component 4: Retrieval Strategies

#### Strategy A: Simple Vector Similarity (Baseline)

```python
results = vector_db.search(query_embedding, top_k=5)
```

Good for: General semantic similarity

#### Strategy B: Hybrid Search (Better)

Combines dense (semantic) + sparse (keyword) retrieval:

```python
from langchain.retrievers import EnsembleRetriever

# Vector retriever (semantic)
vector_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Keyword retriever (BM25)
keyword_retriever = BM25Retriever.from_documents(documents, k=5)

# Ensemble (combines both)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)

results = ensemble_retriever.get_relevant_documents(query)
```

Good for: When exact matches matter (names, codes, specific terms)

#### Strategy C: Query Expansion (Advanced)

Rewrite the query to capture different angles:

```python
# Original query
query = "How do I refund?"

# Expanded queries
expanded_queries = [
    "How do I get a refund?",
    "What is the refund policy?",
    "How can I return a product?"
]

# Retrieve for each, merge results
all_results = []
for expanded_query in expanded_queries:
    results = retrieve(embed(expanded_query), top_k=3)
    all_results.extend(results)

final_results = deduplicate_and_rerank(all_results, top_k=5)
```

Good for: When users might phrase questions differently

---

### Component 5: Prompt Construction

The final step combines retrieved context with the user query:

```python
def build_rag_prompt(query: str, context_chunks: list) -> str:
    # Format context strings
    context_text = "\n\n".join(
        f"[Source {i+1}]: \n{chunk.page_content}" 
        for i, chunk in enumerate(context_chunks)
    )
    
    prompt = f"""
    You are a helpful assistant answering questions based ONLY on the provided context.
    
    <context>
    {context_text}
    </context>
    
    <instructions>
    1. Use ONLY the information in the context above
    2. If the answer isn't clearly in the context, say "I don't have enough information to answer that question based on the provided sources."
    3. Cite your sources using [Source N] format
    4. Be concise but thorough
    </instructions>
    
    <question>
    {query}
    </question>
    
    <answer>
    """
    return prompt
```

**Example Output:**
```
Based on the provided context, our refund policy allows returns within 30 days of 
purchase for a full refund [Source 1]. Items must be in original condition with tags 
attached [Source 2]. Refunds are processed within 5-7 business days after receipt 
of the returned item [Source 1].
```

---

## When to Use RAG vs Fine-Tuning

### Decision Framework

| Factor | Choose RAG | Choose Fine-Tuning |
|--------|------------|-------------------|
| **Data Updates** | Frequent changes | Static knowledge |
| **Reasoning Style** | Standard Q&A | Specialized reasoning needed |
| **Budget** | Lower upfront cost | Higher training cost |
| **Latency Requirements** | Can tolerate retrieval latency | Need fast responses |
| **Data Sensitivity** | Want data isolated | OK with model learning it |
| **Evaluation Needed** | Easy to test/update | Harder to evaluate changes |

### When to Combine Both

```python
# Fine-tuned model + RAG = Best of both worlds
# - Model knows your domain style and terminology (fine-tuning)
# - Model has access to current data (RAG)
```

---

## System Design Decisions

### Decision 1: Batch vs Real-Time Indexing

**Batch Indexing:**
```python
# Run once per day/hour
schedule.every().day.at("02:00").do(index_all_documents)
```
- Pros: Simpler, predictable load
- Cons: Data not immediately searchable

**Real-Time Indexing:**
```python
# Index on upload
@app.post("/documents")
def upload_document(file):
    content = extract_content(file)
    chunks = chunk(content)
    index_chunks(chunks)  # Blocks until done
    return {"status": "indexed"}
```
- Pros: Immediate availability
- Cons: Slower uploads, variable latency

### Decision 2: Re-ranking Layer

Should you re-rank retrieved results before sending to LLM?

```python
# Without re-ranking (fast)
top_k = vector_db.search(query, k=10)  # Take top 10 directly

# With re-ranking (more accurate)
candidates = vector_db.search(query, k=50)  # Get more candidates
reranked = cross_encoder_reranker.rerank(query, candidates, top_n=10)
```

**Tradeoff:** +200-500ms latency for potentially better relevance.

### Decision 3: Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_answer(query_hash: str, context_hash: str) -> str:
    """Cache LLM responses for identical queries."""
    return llm.generate(build_prompt(context))
```

**Savings:** 60-80% of questions in many systems are repeats.

---

## Summary & Next Steps

### What We Covered

1. ✅ **The Problem**: LLMs lack your data, hallucinate, and have context limits
2. ✅ **The Solution**: RAG retrieves relevant info before answering
3. ✅ **Key Components**:
   - Chunking: Breaking documents into pieces
   - Embeddings: Converting text to vectors
   - Vector DB: Storing and searching embeddings
   - Retrieval: Finding relevant chunks
   - Prompt Construction: Combining context + query
4. ✅ **Design Decisions**: Indexing strategy, re-ranking, caching

### What's Next (Part 2)

In Part 2, we'll build a complete RAG system from scratch:
- Setting up the environment
- Loading and chunking real documents
- Building the vector index
- Creating a query interface
- Evaluating performance

---

## References & Further Reading

1. **Original RAG Paper**: [Lewis et al., 2020](https://arxiv.org/abs/2005.11401)
2. **LangChain Documentation**: [RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
3. **Chroma Docs**: [Getting Started](https://docs.trychroma.com/)

---

*This is Part 1 of a 3-part series. [Continue to Part 2: Building the System →]*
