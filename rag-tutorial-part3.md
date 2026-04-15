# Building a RAG System from Scratch

## Part 3: Evaluation, Optimization & Deployment

*How to measure performance, tune your system, and deploy to production.*

---

## TL;DR Summary

In this final part, we'll cover:
1. **Evaluation Metrics** — How to measure RAG quality
2. **Optimization Techniques** — Re-ranking, query expansion, caching
3. **Common Pitfalls** — What goes wrong and how to fix it
4. **Deployment Considerations** — Scaling and monitoring

---

## Part 1: Evaluating Your RAG System

### Why Evaluation Matters

A RAG system can fail in multiple ways:

```
┌─────────────────────────────────────────────────────────┐
│              RAG FAILURE MODES                          │
├─────────────────────────────────────────────────────────┤
│ 1. Retrieval Failure: Wrong chunks retrieved            │
│ 2. Context Failure: Answer not in retrieved chunks      │
│ 3. Generation Failure: Hallucination despite context    │
│ 4. Relevance Failure: Correct but unhelpful answer      │
└─────────────────────────────────────────────────────────┘
```

You need to measure each component separately.

---

### Metric 1: Retrieval Quality (Recall@K)

**Question:** Does retrieval find the right chunks?

```python
"""
Evaluate retrieval quality using a gold standard dataset
"""

def evaluate_retrieval(retriever, test_cases: List[Dict]) -> Dict:
    """
    Test cases format:
    {
        'query': 'What is the refund policy?',
        'gold_chunk_id': 'chunk_123',  # The chunk that should be retrieved
        'gold_document_id': 'doc_456'   # Or just document level
    }
    """
    results = {'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0, 'total': len(test_cases)}
    
    for case in test_cases:
        retrieved = retriever.retrieve(case['query'], top_k=5)
        retrieved_ids = [r.get('id') for r in retrieved]
        
        gold_id = case['gold_chunk_id']
        
        # Check at different K values
        if gold_id in retrieved_ids[:1]:
            results['hit_at_1'] += 1
        if gold_id in retrieved_ids[:3]:
            results['hit_at_3'] += 1
        if gold_id in retrieved_ids[:5]:
            results['hit_at_5'] += 1
    
    # Calculate recall@K
    total = results['total']
    return {
        'recall@1': results['hit_at_1'] / total,
        'recall@3': results['hit_at_3'] / total,
        'recall@5': results['hit_at_5'] / total
    }

# Example usage
test_cases = [
    {
        'query': 'What is the refund policy?',
        'gold_chunk_id': 'chunk_42'  # The chunk containing refund info
    },
    # Add more test cases...
]

metrics = evaluate_retrieval(retriever, test_cases)
print(f"Recall@1: {metrics['recall@1']:.2f}")  # e.g., 0.75
print(f"Recall@3: {metrics['recall@3']:.2f}")  # e.g., 0.89
print(f"Recall@5: {metrics['recall@5']:.2f}")  # e.g., 0.94
```

**Target Metrics:**
- Recall@1 > 0.7 (good)
- Recall@3 > 0.85 (acceptable)
- Recall@5 > 0.9 (minimum)

---

### Metric 2: Answer Correctness (with LLM Judge)

**Question:** Are the generated answers correct?

```python
"""
Use an LLM to judge answer quality (LLM-as-a-Judge pattern)
"""

from langchain.evaluation import load_evaluator
from langchain_core.pydantic_v1 import BaseModel, Field

class AnswerEvaluation(BaseModel):
    correctness_score: float = Field(description="0-1 scale of factual correctness")
    reasoning: str = Field(description="Explanation of the score")
    issues: List[str] = Field(description="Specific problems if any")

def evaluate_answer_quality(test_cases: List[Dict], rag_system) -> List[Dict]:
    """
    Test cases format:
    {
        'query': 'What is the refund policy?',
        'expected_answer': '30 days for full refund...',  # Gold answer
        'context': [...]  # Ground truth context
    }
    """
    evaluator = load_evaler("labeled")
    
    all_evaluations = []
    
    for case in test_cases:
        # Get RAG answer
        rag_answer = rag_system.answer_question(case['query'])
        
        # Evaluate
        evaluation = evaluator.evaluate_strings(
            prediction=rag_answer,
            reference=case['expected_answer'],
            question=case['query']
        )
        
        all_evaluations.append({
            'query': case['query'],
            'rag_answer': rag_answer,
            'expected_answer': case['expected_answer'],
            'score': evaluation.get('score', 0),
            'feedback': evaluation.get('reasoning', '')
        })
    
    return all_evaluations

# Simpler custom evaluator
def simple_llm_judge(query: str, rag_answer: str, gold_answer: str) -> Dict:
    """
    Custom LLM-based evaluation with detailed feedback.
    """
    prompt = f"""
    Evaluate the following RAG answer against the gold standard.
    
    Query: {query}
    
    Gold Answer: {gold_answer}
    
    RAG Answer: {rag_answer}
    
    Provide evaluation in JSON format:
    {{
        "correctness": 0-1 score (1 = fully correct)
        "completeness": 0-1 score (did it cover all key points?)
        "hallucination_detected": boolean
        "key_points_missing": [list of missing points]
        "factual_errors": [list of errors if any]
    }}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Run evaluation
test_cases = [
    {
        'query': 'What is the refund policy?',
        'gold_answer': '30 days for full refund, 60 days for partial...'
    },
    # More cases...
]

for case in test_cases:
    rag_answer = rag_system.answer_question(case['query'])
    evaluation = simple_llm_judge(case['query'], rag_answer, case['gold_answer'])
    print(f"Query: {case['query']}")
    print(f"Correctness: {evaluation['correctness']:.2f}")
    print(f"Hallucination: {evaluation['hallucination_detected']}")
```

---

### Metric 3: Context Relevance & Faithfulness

**Question:** Does the answer use only the provided context?

```python
"""
Faithfulness evaluation - does answer stick to context?
"""

def evaluate_faithfulness(query: str, answer: str, context_chunks: List[str]) -> float:
    """
    Measures if each claim in the answer can be traced to context.
    Returns 0-1 score (1 = fully faithful to context).
    """
    from langchain.evaluation import Faithfulness
    
    faithfulness_evaluator = Faithfulness()
    
    result = faithfulness_evaluator.evaluate_prediction(
        prediction=answer,
        question=query,
        context=context_chunks
    )
    
    return result.get('score', 0)

# Or manual evaluation with LLM
def detailed_faithfulness_check(query: str, answer: str, context: str) -> Dict:
    prompt = f"""
    Analyze the answer for faithfulness to the provided context.
    
    Context:\n{context}
    
    Query: {query}
    
    Answer: {answer}
    
    For each claim in the answer, determine if it is:
    - SUPPORTED: Clearly stated or implied in context
    - PARTIALLY_SUPPORTED: Partially supported but some inference needed  
    - NOT_SUPPORTED: Not found in context (potential hallucination)
    
    Return JSON with analysis of each claim.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)
```

---

### Complete Evaluation Suite

```python
"""
Complete RAG evaluation pipeline
"""

import numpy as np
from typing import List, Dict

class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.results = {}
    
    def run_full_evaluation(self, test_cases: List[Dict]) -> Dict:
        """
        Run all evaluation metrics on test set.
        """
        print("Running comprehensive RAG evaluation...\n")
        
        # 1. Retrieval quality
        print("1/4: Evaluating retrieval quality...")
        retrieval_metrics = self._evaluate_retrieval(test_cases)
        
        # 2. Answer correctness
        print("2/4: Evaluating answer correctness...")
        correctness_metrics = self._evaluate_correctness(test_cases)
        
        # 3. Faithfulness
        print("3/4: Evaluating faithfulness...")
        faithfulness_metrics = self._evaluate_faithfulness(test_cases)
        
        # 4. Latency
        print("4/4: Measuring latency...")
        latency_metrics = self._measure_latency(test_cases)
        
        all_metrics = {
            'retrieval': retrieval_metrics,
            'correctness': correctness_metrics,
            'faithfulness': faithfulness_metrics,
            'latency': latency_metrics,
            'overall_score': self._calculate_overall_score(
                retrieval_metrics, correctness_metrics, 
                faithfulness_metrics, latency_metrics
            )
        }
        
        # Print summary
        self._print_summary(all_metrics)
        return all_metrics
    
    def _evaluate_retrieval(self, test_cases: List[Dict]) -> Dict:
        """Evaluate retrieval recall@K."""
        # Implementation as shown above
        pass
    
    def _evaluate_correctness(self, test_cases: List[Dict]) -> Dict:
        """Evaluate answer correctness using LLM judge."""
        scores = []
        for case in test_cases:
            answer = self.rag_system.answer_question(case['query'])
            eval_result = simple_llm_judge(
                case['query'], answer, case['gold_answer']
            )
            scores.append(eval_result['correctness'])
        
        return {
            'mean_correctness': np.mean(scores),
            'std_correctness': np.std(scores),
            'min_correctness': np.min(scores)
        }
    
    def _evaluate_faithfulness(self, test_cases: List[Dict]) -> Dict:
        """Evaluate answer faithfulness to context."""
        scores = []
        for case in test_cases:
            result = self.rag_system.answer_question(case['query'])
            context = '\n'.join([c['content'] for c in result['sources']])
            faith_score = evaluate_faithfulness(
                case['query'], result['answer'], [context]
            )
            scores.append(faith_score)
        
        return {
            'mean_faithfulness': np.mean(scores),
            'hallucination_rate': sum(1 for s in scores if s < 0.8) / len(scores)
        }
    
    def _measure_latency(self, test_cases: List[Dict], samples: int = 5) -> Dict:
        """Measure system latency."""
        import time
        
        latencies = []
        for case in test_cases[:samples]:  # Sample for speed
            start = time.time()
            self.rag_system.answer_question(case['query'])
            elapsed = time.time() - start
            latencies.append(elapsed)
        
        return {
            'mean_latency_ms': np.mean(latencies) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'max_latency_ms': np.max(latencies) * 1000
        }
    
    def _calculate_overall_score(self, retrieval: Dict, correctness: Dict,
                                 faithfulness: Dict, latency: Dict) -> float:
        """
        Calculate weighted overall score.
        Weights can be adjusted based on priorities.
        """
        weights = {
            'correctness': 0.4,
            'faithfulness': 0.3,
            'retrieval': 0.2,
            'latency': 0.1
        }
        
        # Normalize each metric to 0-1 scale
        correctness_score = correctness['mean_correctness']
        faithfulness_score = faithfulness['mean_faithfulness']
        retrieval_score = retrieval['recall@3']
        latency_score = max(0, 1 - (latency['mean_latency_ms'] / 5000))  # Penalize >5s
        
        overall = (
            weights['correctness'] * correctness_score +
            weights['faithfulness'] * faithfulness_score +
            weights['retrieval'] * retrieval_score +
            weights['latency'] * latency_score
        )
        
        return overall
    
    def _print_summary(self, metrics: Dict):
        """Print formatted evaluation summary."""
        print("\n" + "="*60)
        print("📊 RAG EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n🎯 Overall Score: {metrics['overall_score']:.3f}")
        
        print(f"\n📚 Retrieval Quality:")
        print(f"   Recall@1:  {metrics['retrieval']['recall@1']:.3f}")
        print(f"   Recall@3:  {metrics['retrieval']['recall@3']:.3f}")
        
        print(f"\n✅ Answer Quality:")
        print(f"   Correctness: {metrics['correctness']['mean_correctness']:.3f}")
        print(f"   Faithfulness: {metrics['faithfulness']['mean_faithfulness']:.3f}")
        
        print(f"\n⚡ Performance:")
        print(f"   Mean Latency: {metrics['latency']['mean_latency_ms']:.0f}ms")
        print(f"   P95 Latency:  {metrics['latency']['p95_latency_ms']:.0f}ms")


# Run evaluation
evaluator = RAGEvaluator(rag_system)
test_cases = load_test_dataset()  # Your test cases
results = evaluator.run_full_evaluation(test_cases)
```

---

## Part 2: Optimization Techniques

### Technique 1: Re-Ranking (Improves Retrieval Quality)

**Problem:** Vector similarity doesn't always match relevance.

**Solution:** Use a cross-encoder to re-rank retrieved results.

```python
"""
Re-ranking with Cross-Encoder
"""

from sentence_transformers import CrossEncoder

class RerankingRetriever:
    """Retriever with cross-encoder re-ranking."""
    
    def __init__(self, base_retriever, top_k_initial: int = 20,
                 top_k_final: int = 5):
        self.base_retriever = base_retriever
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        
        # Load cross-encoder model
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-minilm-l12-v2",
            max_length=512,
            device="cpu"  # Or "cuda" if available
        )
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve with re-ranking.
        
        Process:
        1. Get top-K candidates from vector search (fast, approximate)
        2. Re-rank using cross-encoder (slower, more accurate)
        3. Return top-N after re-ranking
        """
        # Step 1: Initial retrieval (get more candidates)
        candidates = self.base_retriever.retrieve(query, top_k=self.top_k_initial)
        
        if len(candidates) <= self.top_k_final:
            return candidates
        
        # Step 2: Create pairs for cross-encoder
        pairs = [
            (query, candidate['content'])
            for candidate in candidates
        ]
        
        # Step 3: Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Step 4: Sort by re-rank score and take top-N
        scored_candidates = [
            {**candidate, 'rerank_score': float(score)}
            for candidate, score in zip(candidates, rerank_scores)
        ]
        
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return sorted_candidates[:self.top_k_final]

# Usage
base_retriever = DocumentRetriever()
reranking_retriever = RerankingRetriever(
    base_retriever,
    top_k_initial=20,  # Get 20 candidates
    top_k_final=5      # Return best 5 after re-ranking
)

results = reranking_retriever.retrieve("What is the refund policy?")
for r in results:
    print(f"Score: {r['rerank_score']:.3f} - {r['content'][:50]}...")
```

**Tradeoff:** +200-500ms latency for potentially 10-20% better retrieval.

---

### Technique 2: Query Expansion (Improves Recall)

**Problem:** Users phrase queries differently than documents are written.

**Solution:** Expand query into multiple related queries.

```python
"""
Query expansion with LLM
"""

def expand_query(query: str, num_expansions: int = 3) -> List[str]:
    """
    Generate related query variations using LLM.
    """
    prompt = f"""
    A user asked: "{query}"
    
    Generate {num_expansions} different ways they might phrase this question
    or related questions that would lead to the same answer.
    
    Return ONLY a JSON array of strings, no other text.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

# Multi-query retrieval with fusion
def multi_query_retrieval(query: str, retriever, num_expansions: int = 3,
                         top_k: int = 5) -> List[Dict]:
    """
    Retrieve using multiple query expansions and fuse results.
    """
    # Expand query
    expanded_queries = expand_query(query, num_expansions)
    print(f"Original: {query}")
    print(f"Expanded: {expanded_queries}\n")
    
    # Retrieve for each query
    all_results = {}
    for exp_query in expanded_queries:
        results = retriever.retrieve(exp_query, top_k=top_k * 2)
        for result in results:
            chunk_id = result['content'][:50]  # Simple ID (use real IDs in production)
            if chunk_id not in all_results:
                all_results[chunk_id] = result
            # Update with highest score
            elif result['similarity'] > all_results[chunk_id]['similarity']:
                all_results[chunk_id] = result
    
    # Sort by max similarity and return top-k
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['similarity'],
        reverse=True
    )
    
    return sorted_results[:top_k]

# Usage
results = multi_query_retrieval("How do I get a refund?", retriever)
```

---

### Technique 3: Caching (Improves Latency & Cost)

**Problem:** Many users ask similar questions.

**Solution:** Cache responses for identical/similar queries.

```python
"""
Caching layer for RAG system
"""

import hashlib
from functools import lru_cache
import redis

class RAGCache:
    """Multi-level caching for RAG system."""
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        if use_redis:
            self.redis_client = redis.Redis(host='localhost', port=6379)
    
    def _get_cache_key(self, query: str, context_hash: str) -> str:
        """Generate cache key from query and context."""
        combined = f"{query}::{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context_chunks: List[Dict]) -> str | None:
        """
        Try to retrieve cached answer.
        Returns cached answer or None if miss.
        """
        # Create hash of context (order-independent)
        context_hash = hashlib.md5(
            ''.join(sorted([c['content'] for c in context_chunks])).encode()
        ).hexdigest()[:16]
        
        cache_key = self._get_cache_key(query, context_hash)
        
        if self.use_redis:
            cached = self.redis_client.get(cache_key)
            return cached.decode() if cached else None
        else:
            # Simple in-memory cache (lost on restart)
            if hasattr(self, '_cache') and cache_key in self._cache:
                return self._cache[cache_key]
            return None
    
    def set(self, query: str, context_chunks: List[Dict], answer: str,
           ttl: int = 3600):
        """
        Store answer in cache.
        TTL in seconds (default: 1 hour).
        """
        context_hash = hashlib.md5(
            ''.join(sorted([c['content'] for c in context_chunks])).encode()
        ).hexdigest()[:16]
        
        cache_key = self._get_cache_key(query, context_hash)
        
        if self.use_redis:
            self.redis_client.setex(cache_key, ttl, answer)
        else:
            if not hasattr(self, '_cache'):
                self._cache = {}
            self._cache[cache_key] = answer
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if self.use_redis:
            info = self.redis_client.info('memory')
            return {
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'keys_count': self.redis_client.dbsize()
            }
        else:
            return {'cache_size': len(getattr(self, '_cache', {}))}

# Integration with RAG system
class CachedRAGSystem(RAGApplication):
    """RAG system with caching layer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = RAGCache(use_redis=True)  # Or False for in-memory
        self.cache_hits = 0
        self.cache_misses = 0
    
    def answer_question(self, query: str, **kwargs) -> Dict:
        """Answer with caching."""
        # First retrieve to get context
        context_chunks = self.retriever.retrieve(query, top_k=kwargs.get('top_k', 3))
        
        # Try cache
        cached_answer = self.cache.get(query, context_chunks)
        if cached_answer:
            self.cache_hits += 1
            print("[CACHE HIT]")
            return {
                'answer': cached_answer,
                'sources': context_chunks,
                'query': query,
                'cached': True
            }
        
        # Cache miss - generate answer
        self.cache_misses += 1
        print("[CACHE MISS]")
        prompt = self.build_prompt(query, context_chunks)
        answer = self.generate_answer(prompt)
        
        # Store in cache
        self.cache.set(query, context_chunks, answer)
        
        return {
            'answer': answer,
            'sources': context_chunks,
            'query': query,
            'cached': False
        }
    
    def get_cache_stats(self) -> Dict:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / total if total > 0 else 0
        }
```

---

## Part 3: Common Pitfalls & Solutions

### Pitfall 1: Garbage In, Garbage Out

**Problem:** Poor quality source documents lead to poor answers.

```python
"""
Document quality check before indexing
"""

def assess_document_quality(file_path: str) -> Dict:
    """Assess if document is suitable for RAG."""
    # Check 1: File size
    size = os.path.getsize(file_path)
    if size > 50 * 1024 * 1024:  # 50MB
        return {'valid': False, 'reason': 'File too large'}
    
    # Check 2: Readability (for text files)
    if file_path.endswith('.txt'):
        with open(file_path) as f:
            content = f.read()
        avg_word_length = sum(len(w) for w in content.split()) / max(1, len(content.split()))
        if avg_word_length > 15:  # Likely binary/gibberish
            return {'valid': False, 'reason': 'Low readability score'}
    
    # Check 3: PDF text extractability
    if file_path.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ''.join(p.extract_text() or '' for p in reader.pages[:5])
        if len(text.strip()) < 100:  # Scanned PDF without OCR?
            return {'valid': False, 'reason': 'No extractable text (scanned?)'}
    
    return {'valid': True, 'size_bytes': size}

# Filter documents before indexing
all_files = find_documents('./source/')
valid_files = [
    f for f in all_files 
    if assess_document_quality(f)['valid']
]
print(f"Validated {len(valid_files)}/{len(all_files)} documents")
```

---

### Pitfall 2: Chunk Size Misconfiguration

**Problem:** Chunks too small lose context; chunks too large waste tokens.

```python
"""
Experiment with chunk sizes
"""

def benchmark_chunk_sizes(documents: List, queries: List[str]) -> Dict:
    """Test different chunk sizes and measure impact."""
    chunk_sizes = [256, 512, 768, 1024, 2048]
    results = {}
    
    for chunk_size in chunk_sizes:
        # Index with this chunk size
        chunks = chunk_documents(documents, chunk_size=chunk_size)
        index_chunks(chunks)
        
        # Evaluate on queries
        total_relevance = 0
        total_tokens_used = 0
        
        for query in queries:
            results_list = retrieve(query, top_k=3)
            context_tokens = sum(len(c['content'].split()) for c in results_list)
            total_tokens_used += context_tokens
            # Simple relevance proxy: average similarity
            total_relevance += sum(c['similarity'] for c in results_list)
        
        avg_similarity = total_relevance / len(queries) / 3
        avg_tokens = total_tokens_used / len(queries)
        
        results[chunk_size] = {
            'avg_similarity': avg_similarity,
            'avg_context_tokens': avg_tokens,
            'num_chunks': len(chunks)
        }
    
    return results

# Run benchmark
results = benchmark_chunk_sizes(documents, test_queries)
for size, metrics in results.items():
    print(f"Chunk size {size}: similarity={metrics['avg_similarity']:.3f}, "
          f"tokens={metrics['avg_context_tokens']:.0f}")
```

---

### Pitfall 3: No Fallback for Unknown Questions

**Problem:** System hallucinates answers to questions it can't answer.

```python
"""
Confidence-based fallback mechanism
"""

def answer_with_confidence(query: str, context_chunks: List[Dict], 
                          threshold: float = 0.6) -> Dict:
    """
    Answer only if confidence is above threshold.
    Otherwise return fallback response.
    """
    # Check max similarity of retrieved chunks
    max_similarity = max(c['similarity'] for c in context_chunks)
    
    if max_similarity < threshold:
        return {
            'answer': f"I couldn't find relevant information about '{query}' in my knowledge base. "
                     f"The most similar content had a relevance score of {max_similarity:.2f} "
                     f"(threshold: {threshold}).\n\n" 
                     f"Could you rephrase your question or try a different topic?",
            'sources': [],
            'confidence': max_similarity,
            'fallback_triggered': True
        }
    
    # Normal RAG answer
    prompt = build_prompt(query, context_chunks)
    answer = generate_answer(prompt)
    
    return {
        'answer': answer,
        'sources': context_chunks,
        'confidence': max_similarity,
        'fallback_triggered': False
    }
```

---

## Part 4: Deployment Considerations

### Scaling Architecture

```python
"""
Production-ready RAG architecture overview
"""

# Recommended stack for production:
#
# ┌─────────────────────────────────────────────────────┐
# │                  LOAD BALANCER                       │
# │                (nginx / cloud LB)                    │
# └──────────────────────┬──────────────────────────────┘
#                        │
#        ┌───────────────┴───────────────┐
#        ▼                               ▼
# ┌──────────────┐              ┌──────────────┐
# │  API Server  │              │  API Server  │
# │  (FastAPI)   │              │  (FastAPI)   │
# └──────┬───────┘              └──────┬───────┘
#        │                            │
#        └────────────┬───────────────┘
#                     ▼
#         ┌──────────────────┐
#         │   Message Queue  │
#         │    (Redis/Celery)│
#         └────────┬─────────┘
#                  │
#     ┌────────────┴────────────┐
#     ▼                         ▼
# ┌──────────┐            ┌──────────┐
# │ Worker   │            │ Worker   │
# │ (Index)  │            │(Query)   │
# └─────┬────┘            └─────┬────┘
#       │                      │
#       ▼                      ▼
# ┌──────────────┐      ┌──────────────┐
# │ Vector DB    │      │   LLM API    │
# │ (Pinecone/   │      │  (OpenAI/    │
# │  Qdrant)     │      │   Anthropic) │
# └──────────────┘      └──────────────┘
```

### FastAPI Production Server

```python
"""
Production RAG API with FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="RAG API", version="1.0.0")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    temperature: float = 0.7
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None
    confidence: float
    processing_time_ms: int

# Initialize RAG system globally (outside requests)
rag_system = CachedRAGSystem(collection_name="production_kb")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint."""
    import time
    start_time = time.time()
    
    try:
        result = rag_system.answer_question(
            query=request.query,
            top_k=request.top_k
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'] if request.include_sources else None,
            confidence=result.get('confidence', 1.0),
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_endpoint(directory: str):
    """Trigger indexing job."""
    rag_system.indexer.index_from_directory(directory)
    return {"status": "indexed"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cache_hit_rate": rag_system.get_cache_stats()['hit_rate']
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Summary

### What We've Built Across This Series

| Part | Focus | Key Takeaways |
|------|-------|---------------|
| **Part 1** | Fundamentals | RAG architecture, components, tradeoffs |
| **Part 2** | Implementation | Working code for indexing and querying |
| **Part 3** | Optimization | Evaluation, re-ranking, caching, deployment |

### Quick Reference: RAG Checklist

```markdown
## Before You Start
□ Define your use case and success metrics
□ Gather and clean source documents
□ Choose embedding model (quality vs cost tradeoff)

## Implementation
□ Chunk size: Start with 512-768 tokens
□ Overlap: 50-100 tokens
□ Top-k retrieval: Start with 3-5 chunks
□ Embedding model: text-embedding-3-small (good balance)

## Optimization (in order of impact)
1. Improve chunking strategy
2. Add re-ranking layer
3. Implement caching
4. Try query expansion

## Evaluation
□ Recall@K > 0.85 for retrieval
□ Correctness > 0.7 on test set
□ Faithfulness > 0.8 (low hallucination)
□ Latency < 2s for end-to-end
```

---

*This completes the 3-part series on Building a RAG System from Scratch.*

**Resources:**
- [GitHub Repository with all code](#)
- [Part 1: Fundamentals](#) | [Part 2: Implementation](#)
