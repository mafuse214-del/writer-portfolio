# Building a RAG System from Scratch

## Part 2: Hands-On Implementation

*Let's build a working RAG system step-by-step, with real code you can run.*

---

## TL;DR Summary

In this part, we'll build a complete RAG system that:
1. Loads documents from a folder (PDFs, TXT, MD)
2. Chunks and embeds them automatically
3. Stores embeddings in ChromaDB
4. Answers questions via a simple CLI interface

**Final Result:** A working knowledge base you can query about your own documents.

---

## Prerequisites & Setup

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate

# Install packages
pip install langchain chromadb openai pypdf python-dotenv
```

### Set Up API Key

Create a `.env` file:

```bash
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
EOF
```

**Free Alternative:** Use Hugging Face embeddings (no API key needed):
```bash
pip install sentence-transformers
```

---

## Step 1: Project Structure

Create this folder structure:

```
rag-system/
├── .env                    # API keys
├── documents/              # Your source documents
│   ├── employee-handbook.pdf
│   ├── product-specs.md
│   └── faq.txt
├── rag_app.py             # Main application
├── indexer.py             # Document indexing logic
├── retriever.py           # Query retrieval logic
└── requirements.txt       # Dependencies
```

---

## Step 2: Build the Indexer

The indexer loads, chunks, and stores your documents.

### Complete Indexer Code (`indexer.py`)

```python
"""
Indexer Module - Loads and indexes documents into vector database
"""

import os
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# Alternative (free): from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings


class DocumentIndexer:
    """Handles document loading, chunking, and indexing."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the indexer with ChromaDB client."""
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Or use free alternative:
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
    
    def load_documents(self, document_paths: List[str]) -> List:
        """
        Load documents from various file formats.
        
        Args:
            document_paths: List of file paths to load
            
        Returns:
            List of LangChain Document objects
        """
        all_documents = []
        
        for path in document_paths:
            print(f"Loading: {path}")
            
            try:
                # Determine loader based on file extension
                if path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif path.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(path)
                elif path.endswith('.txt'):
                    loader = TextLoader(path)
                else:
                    print(f"  ⚠ Skipping unsupported format: {path}")
                    continue
                
                # Load document
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  ✓ Loaded {len(docs)} pages/chunks")
                
            except Exception as e:
                print(f"  ✗ Error loading {path}: {str(e)}")
        
        return all_documents
    
    def chunk_documents(self, documents: List, chunk_size: int = 1000,
                       chunk_overlap: int = 200) -> List:
        """
        Split documents into overlapping chunks.
        
        Args:
            documents: List of Document objects
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks for context continuity
            
        Returns:
            List of smaller Document chunks
        """
        print(f"\nChunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,  # Track original position
            separators=["\n\n", "\n", ". ", " "]
        )
        
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                all_chunks.append({
                    'content': chunk,
                    'source': doc.metadata.get('source', 'unknown')
                })
        
        print(f"  ✓ Created {len(all_chunks)} chunks")
        return all_chunks
    
    def index_chunks(self, chunks: List[Dict], collection_name: str = "knowledge_base"):
        """
        Embed and store chunks in ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'source'
            collection_name: Name for the ChromaDB collection
        """
        print(f"\nIndexing {len(chunks)} chunks...")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embeddings
            )
            # Clear existing data for fresh index
            self.client.delete_collection(name=collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Prepare data for insertion
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [{"source": chunk['source']} for chunk in chunks]
        
        # Add to collection (embeddings computed automatically)
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"  ✓ Indexed into collection '{collection_name}'")
        print(f"  ✓ Total vectors in DB: {self.collection.count()}")
    
    def index_from_directory(self, directory: str, collection_name: str = "knowledge_base",
                            recursive: bool = False):
        """
        Convenience method to index all documents in a directory.
        
        Args:
            directory: Path to directory containing documents
            collection_name: Name for ChromaDB collection
            recursive: Whether to search subdirectories
        """
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        supported_extensions = ('.pdf', '.txt', '.md')
        
        document_paths = [
            str(path) for path in Path(directory).glob(pattern)
            if path.is_file() and path.suffix.lower() in supported_extensions
        ]
        
        if not document_paths:
            print(f"⚠ No supported documents found in {directory}")
            return
        
        print(f"Found {len(document_paths)} documents to index:\n")
        for path in document_paths:
            print(f"  - {path}")
        print()
        
        # Process pipeline
        documents = self.load_documents(document_paths)
        if not documents:
            print("⚠ No documents were successfully loaded")
            return
            
        chunks = self.chunk_documents(documents)
        self.index_chunks(chunks, collection_name)
        
        print(f"\n✅ Indexing complete!")


# Example usage
if __name__ == "__main__":
    # Initialize indexer
    indexer = DocumentIndexer(persist_directory="./chroma_db")
    
    # Index all documents from a directory
    indexer.index_from_directory("./documents", collection_name="my_knowledge_base")
```

---

## Step 3: Build the Retriever

The retriever handles queries and returns relevant context.

### Complete Retriever Code (`retriever.py`)

```python
"""
Retriever Module - Handles querying the indexed documents
"""

import os
from typing import List, Dict, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import openai


class DocumentRetriever:
    """Handles querying and retrieving relevant documents."""
    
    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "knowledge_base"):
        """Initialize the retriever with ChromaDB client."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Or free alternative:
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Collection '{collection_name}' not found. Run indexer first.")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with 'content', 'source', and 'score'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            # Convert distance to similarity (1 - normalized_distance)
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)  # Simple conversion
            
            formatted_results.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "similarity": similarity,
                "raw_distance": distance
            })
        
        return formatted_results
    
    def retrieve_with_explanation(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve results with human-readable explanation.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Formatted string showing retrieved context
        """
        results = self.retrieve(query, top_k)
        
        output = f"\n🔍 Query: \"{query}\"\n"
        output += "=" * 60 + "\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"[Source {i}] (Similarity: {result['similarity']:.3f})\n"
            output += f"From: {result['source']}\n"
            output += "-" * 40 + "\n"
            # Truncate long content for display
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            output += f"{content}\n\n"
        
        return output
    
    def search_with_filters(self, query: str, source_filter: str = None,
                           top_k: int = 3) -> List[Dict]:
        """
        Retrieve with metadata filtering.
        
        Args:
            query: The search query
            source_filter: Only search documents from this source
            top_k: Number of results to return
            
        Returns:
            Filtered list of results
        """
        where_filter = {"source": source_filter} if source_filter else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "distances", "metadatas"]
        )
        
        formatted_results = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)
            
            formatted_results.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "similarity": similarity
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the indexed collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get unique sources
        all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
        sources = set(m.get("source", "unknown") for m in all_metadata)
        
        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "sources_list": list(sources)
        }


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = DocumentRetriever(
        persist_directory="./chroma_db",
        collection_name="my_knowledge_base"
    )
    
    # Show stats
    stats = retriever.get_collection_stats()
    print(f"Collection Stats: {stats}")
    
    # Test query
    query = "What is the refund policy?"
    results = retriever.retrieve_with_explanation(query)
    print(results)
```

---

## Step 4: Build the Main Application

Combine indexing and retrieval with LLM integration.

### Complete RAG App (`rag_app.py`)

```python
"""
RAG Application - Main entry point for the question-answering system
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
from indexer import DocumentIndexer
from retriever import DocumentRetriever
import openai

# Load environment variables
load_dotenv()


class RAGApplication:
    """Main RAG application combining indexing, retrieval, and generation."""
    
    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "knowledge_base"):
        """Initialize the RAG application."""
        self.indexer = DocumentIndexer(persist_directory=persist_directory)
        self.retriever = None  # Will be initialized after indexing
        self.collection_name = collection_name
        
        # Configure OpenAI client
        self.client = openai.OpenAI()
    
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build the final prompt combining context and query.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Complete prompt string for LLM
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"<source id='{i}'>\n{chunk['content']}\n</source>"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a helpful research assistant. Answer the following question 
using ONLY the information provided in the sources below.

IMPORTANT GUIDELINES:
1. Use only information from the provided sources
2. If the answer is not clearly in the sources, state "I cannot find a clear answer 
to this question in the provided sources."
3. Cite sources using [Source N] format at the end of relevant sentences
4. Be concise but thorough
5. Do not make up information or speculate

<sources>
{context_text}
</sources>

<question>
{query}
</question>

<answer>
"""
        return prompt
    
    def generate_answer(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate answer using OpenAI's GPT model.
        
        Args:
            prompt: The complete prompt with context and query
            temperature: Creativity/randomness (0.0-2.0)
            
        Returns:
            Generated answer string
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective for this use case
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    def answer_question(self, query: str, top_k: int = 3, 
                       show_reasoning: bool = False) -> Dict:
        """
        Main method to answer a question using RAG.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            show_reasoning: Whether to include retrieved context in output
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant chunks
        context_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        if not context_chunks:
            return {
                "answer": "I don't have any indexed documents to search. Please run the indexer first.",
                "sources": [],
                "query": query
            }
        
        # Build prompt with context
        prompt = self.build_prompt(query, context_chunks)
        
        # Generate answer
        answer = self.generate_answer(prompt)
        
        result = {
            "answer": answer,
            "sources": context_chunks,
            "query": query,
            "num_sources_used": len(context_chunks)
        }
        
        return result
    
    def run_interactive_session(self):
        """Run an interactive question-answering session."""
        print("\n" + "="*60)
        print("🤖 RAG Question-Answering System")
        print("="*60)
        print("\nEnter your questions below. Type 'quit' or 'exit' to end.")
        print("Type 'stats' to see collection information.\n")
        
        while True:
            try:
                query = input("📝 Your question: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = self.retriever.get_collection_stats()
                    print(f"\n📊 Collection Statistics:")
                    print(f"   Total chunks indexed: {stats['total_chunks']}")
                    print(f"   Unique sources: {stats['unique_sources']}")
                    print(f"   Sources: {', '.join(stats['sources_list'])}\n")
                    continue
                
                # Process query
                print("\n⏳ Thinking...")
                result = self.answer_question(query, show_reasoning=True)
                
                # Display results
                print("\n" + "-"*60)
                print(f"💡 Answer:")
                print("-"*60)
                print(result['answer'])
                print("\n📚 Sources used:")
                for i, source in enumerate(result['sources'], 1):
                    similarity = source['similarity']
                    src_name = source['source'][:30] + "..." if len(source['source']) > 30 else source['source']
                    print(f"   [{i}] {src_name} (similarity: {similarity:.3f})")
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Question-Answering System")
    parser.add_argument("--index", type=str, help="Index documents from directory")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    parser.add_argument("--collection", type=str, default="knowledge_base",
                       help="Collection name in ChromaDB")
    args = parser.parse_args()
    
    # Initialize application
    app = RAGApplication(collection_name=args.collection)
    
    # Index mode
    if args.index:
        print(f"📁 Indexing documents from: {args.index}")
        app.indexer.index_from_directory(args.index, collection_name=args.collection)
        print("\n✅ Indexing complete! You can now query the system.")
        return
    
    # Initialize retriever for querying
    try:
        app.retriever = DocumentRetriever(collection_name=args.collection)
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Tip: Run with --index ./your_documents_folder first")
        return
    
    # Single query mode
    if args.query:
        result = app.answer_question(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {result['answer']}")
        return
    
    # Interactive mode (default)
    app.run_interactive_session()


if __name__ == "__main__":
    main()
```

---

## Step 5: Run Your RAG System!

### Index Your Documents

```bash
# Create a documents folder and add some files
mkdir -p documents
cp your-file.pdf documents/
cp your-file.txt documents/

# Run the indexer
python rag_app.py --index ./documents --collection my_docs
```

**Output:**
```
Found 3 documents to index:

  - documents/employee-handbook.pdf
  - documents/product-specs.md
  - documents/faq.txt

Loading: documents/employee-handbook.pdf
  ✓ Loaded 25 pages/chunks
...

Chunking 25 documents...
  ✓ Created 87 chunks

Indexing 87 chunks...
  ✓ Indexed into collection 'my_docs'
  ✓ Total vectors in DB: 87

✅ Indexing complete!
```

### Query Your Knowledge Base

```bash
# Interactive mode
python rag_app.py --collection my_docs
```

**Sample Session:**
```
📝 Your question: What is the employee vacation policy?

⏳ Thinking...

------------------------------------------------------------
💡 Answer:
------------------------------------------------------------
According to the employee handbook, full-time employees are entitled to 
15 days of paid vacation per year [Source 1]. Vacation accrues monthly at 
a rate of 1.25 days per month [Source 2]. Employees must request vacation 
at least 2 weeks in advance through the HR portal [Source 1].

📚 Sources used:
   [1] employee-handbook.pdf (similarity: 0.847)
   [2] employee-handbook.pdf (similarity: 0.792)
   [3] hr-policies.md (similarity: 0.756)
```

### Single Query Mode

```bash
python rag_app.py --query "What are the system requirements?" --collection my_docs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `API key error` | Check `.env` file and verify key at platform.openai.com |
| `Collection not found` | Run indexer first with `--index` flag |
| `Poor answer quality` | Increase `top_k`, check chunk size, verify documents loaded |
| `Slow responses` | Use smaller chunks, reduce `top_k`, or cache results |

---

## Summary & Next Steps

### What We Built

✅ **Document Indexer** - Loads PDFs/TXT/MD, chunks, embeds, stores in ChromaDB  
✅ **Document Retriever** - Semantic search with similarity scoring  
✅ **RAG Application** - Combines retrieval + LLM for answers  
✅ **CLI Interface** - Interactive and single-query modes

### What's Next (Part 3)

In Part 3, we'll cover:
- Evaluating RAG performance (metrics, testing)
- Advanced techniques (re-ranking, query expansion)
- Deployment considerations
- Common pitfalls and how to avoid them

---

*This is Part 2 of a 3-part series. [← Back to Part 1] | [Continue to Part 3 →]*
