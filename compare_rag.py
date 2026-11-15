"""CLI script to verify embeddings, persistence, and run the Groq-powered RAG vs. non-RAG comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import groq
from google import genai

from config import load_config
from src.document_loader import load_documents_from_directory
from src.embeddings import GeminiEmbedder
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.text_splitter import chunk_documents, inspect_sample_chunks
from src.vector_store import ChromaVectorStore


def main() -> int:
    cfg = load_config()
    cfg.ensure_directories()

    is_valid, validation_msg = cfg.validate_api_key()
    if not is_valid:
        print(f"CONFIGURATION ERROR: {validation_msg}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("VALENTIUS RAG KNOWLEDGE ASSISTANT - HYBRID ARCHITECTURE VERIFICATION")
    print("=" * 70)
    print(f"Gemini Embeddings Model : {cfg.embedding_model} ({cfg.embedding_dimensions}d)")
    print(f"Groq Generation Model   : {cfg.groq_model}")
    print(f"Groq Base URL           : {cfg.groq_base_url}")
    print(f"ChromaDB Storage        : {cfg.chroma_db_dir}")
    print("=" * 70)

    # 1. Verify Gemini API connectivity (for embeddings)
    print("\n[1/5] Verifying Google Gemini Embeddings API connectivity...")
    try:
        genai_client = genai.Client(api_key=cfg.gemini_api_key)
        # Check that client can query models
        models = {
            m.name.removeprefix("models/")
            for m in genai_client.models.list(config={"page_size": 50})
            if getattr(m, "name", None)
        }
        print(f"  ✓ Connected to Gemini API ({len(models)} models available)")
    except Exception as exc:
        print(f"  Note on Gemini listing: {exc}")

    # 2. Verify Groq API connectivity (for generation)
    print("\n[2/5] Verifying Groq Chat Completion API connectivity...")
    try:
        norm_url = cfg.groq_base_url.rstrip("/")
        if norm_url.endswith("/openai/v1"):
            norm_url = norm_url[:-len("/openai/v1")].rstrip("/")
        groq_client = groq.Groq(api_key=cfg.groq_api_key, base_url=norm_url or None)
        test_chat = groq_client.chat.completions.create(
            model=cfg.groq_model,
            messages=[{"role": "user", "content": "Respond with 'Groq online'."}],
            max_tokens=10,
        )
        print(f"  ✓ Connected to Groq API ({cfg.groq_model}): '{test_chat.choices[0].message.content.strip()}'")
    except Exception as exc:
        print(f"  ✗ Groq API check failed: {exc}", file=sys.stderr)
        return 1

    # 3. Verify ChromaDB documents & chunks
    print("\n[3/5] Inspecting ChromaDB Vector Store...")
    embedder = GeminiEmbedder(
        api_key=cfg.gemini_api_key,
        model_name=cfg.embedding_model,
        output_dimensionality=cfg.embedding_dimensions,
    )
    store = ChromaVectorStore(
        persist_directory=cfg.chroma_db_dir,
        collection_name=cfg.chroma_collection_name,
    )
    doc_summary = store.list_documents()
    print(f"  ✓ Persistent collection: '{cfg.chroma_collection_name}'")
    print(f"  ✓ Total Indexed Documents: {doc_summary['total_documents']}")
    print(f"  ✓ Total Indexed Chunks   : {doc_summary['total_chunks']}")

    # If ChromaDB is empty, ingest from documents directory
    if doc_summary["total_chunks"] == 0:
        print(f"\n  In-memory collection empty. Ingesting from {cfg.documents_dir}...")
        report = load_documents_from_directory(cfg.documents_dir, cfg.allowed_extensions)
        chunks = chunk_documents(
            report.pages,
            chunk_size_tokens=cfg.chunk_size_tokens,
            chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        )
        stats = store.add_chunks(chunks, embedder=embedder, force=False)
        print(f"  ✓ Ingested {len(stats.files_ingested)} files ({stats.chunks_added} chunks).")

    # 4. Initialize Retriever and Groq RAG Pipeline
    print("\n[4/5] Initializing Hybrid RAG Pipeline...")
    retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=cfg.top_k_results)
    pipeline = RAGPipeline(
        retriever=retriever,
        groq_api_key=cfg.groq_api_key,
        generation_model=cfg.groq_model,
        base_url=cfg.groq_base_url,
    )
    print("  ✓ RAG Pipeline ready with Groq generation and Gemini embeddings.")

    # 5. Run Section 10 RAG vs. Plain LLM Comparison
    question = (
        "According to the Valentius Kryptix Internal RAG Engineering Standard (VK-RAG-2026-v1), "
        "what are the required target chunk size, chunk overlap, embedding dimensionality, and maximum P95 retrieval latency?"
    )
    print("\n[5/5] Running RAG vs. Plain LLM Performance Comparison...")
    print(f"Question: \"{question}\"\n")

    comparison = pipeline.compare(question, top_k=5)

    print("=" * 70)
    print(f"MODE A: PLAIN GROQ ({cfg.groq_model}) - NO RETRIEVAL")
    print("=" * 70)
    print(comparison["plain_llm_answer"])
    print(f"\n⏱ Latency: {comparison['timing']['plain_llm']['total_time_ms']} ms")
    print("=" * 70)

    print(f"\nMODE B: HYBRID RAG (GEMINI EMBEDDINGS + CHROMADB + GROQ {cfg.groq_model})")
    print("=" * 70)
    print(comparison["rag_answer"])
    print("\n--- Provenance & Citations ---")
    for cit in comparison["citations"]:
        p_str = f", Page {cit['page']}" if cit["page"] is not None else ""
        print(f"  • [Source: {cit['source']}{p_str}] (Relevance: {cit['relevance_score']:.4f})")
    print(f"\n--- Latency Telemetry Breakdown ---")
    rag_timing = comparison["timing"]["rag"]
    print(f"  • Gemini Query Embedding Latency : {rag_timing['embed_time_ms']} ms")
    print(f"  • ChromaDB Vector Retrieval Latency: {rag_timing['retrieval_time_ms']} ms")
    print(f"  • Groq Generation Latency        : {rag_timing['generation_time_ms']} ms")
    print(f"  • Total End-to-End Latency       : {rag_timing['total_time_ms']} ms")
    print("=" * 70)

    out_path = Path("comparison_output.json")
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved full comparison output to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
