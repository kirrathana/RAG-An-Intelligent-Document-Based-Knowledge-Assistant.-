"""Flask backend application for the Valentius RAG Knowledge Assistant."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config, load_config
from src.document_loader import load_document, load_documents_from_directory
from src.embeddings import EmbeddingError, GeminiEmbedder
from src.rag_pipeline import GenerationError, RAGPipeline
from src.retriever import Retriever
from src.text_splitter import chunk_documents, inspect_sample_chunks
from src.vector_store import ChromaVectorStore, VectorStoreError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("valentius.app")


def _ingest_directory_on_startup(
    cfg: Config,
    vector_store: ChromaVectorStore,
    embedder: GeminiEmbedder,
) -> None:
    """Index any unindexed documents present in data/documents on startup if API key is valid."""
    is_valid, validation_msg = cfg.validate_api_key()
    if not is_valid:
        logger.info("Startup ingestion skipped: %s", validation_msg)
        return

    report = load_documents_from_directory(cfg.documents_dir, cfg.allowed_extensions)
    if not report.pages:
        logger.info("No documents found in %s.", cfg.documents_dir)
        return

    chunks = chunk_documents(
        report.pages,
        chunk_size_tokens=cfg.chunk_size_tokens,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
    )
    if not chunks:
        return

    try:
        stats = vector_store.add_chunks(chunks, embedder=embedder, force=False)
        logger.info(
            "Startup ingestion complete: %d ingested, %d skipped, %d total chunks in ChromaDB.",
            len(stats.files_ingested),
            len(stats.files_skipped),
            stats.total_chunks_in_store,
        )
    except EmbeddingError as exc:
        logger.warning("Startup ingestion could not complete: %s", exc)
    except Exception as exc:
        logger.warning("Startup ingestion encountered an error: %s", exc)


def create_app(
    cfg: Optional[Config] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    embedder: Optional[GeminiEmbedder] = None,
    retriever: Optional[Retriever] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    auto_ingest_on_startup: bool = False,
) -> Flask:
    """Create and configure the Flask application."""
    app_cfg = cfg or load_config()
    app_cfg.ensure_directories()

    store = vector_store or ChromaVectorStore(
        persist_directory=app_cfg.chroma_db_dir,
        collection_name=app_cfg.chroma_collection_name,
    )
    emb = embedder or GeminiEmbedder(
        api_key=app_cfg.gemini_api_key,
        model_name=app_cfg.embedding_model,
        output_dimensionality=app_cfg.embedding_dimensions,
    )
    ret = retriever or Retriever(
        embedder=emb,
        vector_store=store,
        default_top_k=app_cfg.top_k_results,
    )
    pipeline = rag_pipeline or RAGPipeline(
        retriever=ret,
        groq_api_key=app_cfg.groq_api_key,
        generation_model=app_cfg.groq_model,
        base_url=app_cfg.groq_base_url,
    )

    if auto_ingest_on_startup:
        _ingest_directory_on_startup(app_cfg, store, emb)

    app = Flask(
        __name__,
        static_folder=str(app_cfg.frontend_dir),
        static_url_path="",
    )
    app.config["MAX_CONTENT_LENGTH"] = app_cfg.max_upload_size_mb * 1024 * 1024
    app.config["VALENTIUS_CFG"] = app_cfg
    app.config["VECTOR_STORE"] = store
    app.config["EMBEDDER"] = emb
    app.config["RETRIEVER"] = ret
    app.config["RAG_PIPELINE"] = pipeline

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err: RequestEntityTooLarge):
        return (
            jsonify(
                {
                    "error": (
                        f"Uploaded file exceeds the maximum allowed size of "
                        f"{app_cfg.max_upload_size_mb} MB."
                    )
                }
            ),
            413,
        )

    @app.get("/")
    def serve_frontend():
        return send_from_directory(str(app_cfg.frontend_dir), "index.html")

    @app.get("/api/health")
    def health_check():
        doc_summary = store.list_documents()
        is_valid, validation_msg = app_cfg.validate_api_key()
        status = "ok" if is_valid else "degraded_invalid_api_key"
        return jsonify(
            {
                "status": status,
                "status_message": validation_msg,
                "config": app_cfg.to_public_dict(),
                "vector_store": {
                    "total_documents": doc_summary["total_documents"],
                    "total_chunks": doc_summary["total_chunks"],
                },
            }
        )

    @app.get("/api/documents")
    def list_documents():
        try:
            summary = store.list_documents()
            disk_files = sorted(
                p.name
                for p in app_cfg.documents_dir.iterdir()
                if p.is_file() and p.suffix.lower() in app_cfg.allowed_extensions
            )
            summary["files_on_disk"] = disk_files
            return jsonify(summary)
        except Exception as exc:
            logger.exception("Failed to list documents")
            return jsonify({"error": f"Failed to list documents: {exc}"}), 500

    @app.get("/api/chunks/sample")
    def sample_chunks():
        raw_limit = request.args.get("limit", "5")
        source = request.args.get("source")
        try:
            limit = max(1, min(int(raw_limit), 20))
        except ValueError:
            return jsonify({"error": "Query parameter 'limit' must be an integer."}), 400

        try:
            samples = store.get_sample_chunks(limit=limit, source=source)
            return jsonify({"count": len(samples), "samples": samples})
        except Exception as exc:
            logger.exception("Failed to fetch sample chunks")
            return jsonify({"error": f"Failed to fetch sample chunks: {exc}"}), 500

    def _parse_top_k(payload: dict[str, Any]) -> int:
        raw_top_k = payload.get("top_k", app_cfg.top_k_results)
        if isinstance(raw_top_k, bool):
            raise ValueError("top_k must be an integer between 1 and 20.")
        try:
            k = int(raw_top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer between 1 and 20.") from exc
        if k < 1 or k > 20:
            raise ValueError("top_k must be between 1 and 20.")
        return k

    @app.post("/api/retrieve")
    def retrieve_chunks():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        query = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({"error": "Field 'query' is required and cannot be empty."}), 400

        try:
            top_k = _parse_top_k(payload)
            chunks = ret.retrieve(query=query, top_k=top_k)
            return jsonify(
                {
                    "query": query,
                    "top_k": top_k,
                    "count": len(chunks),
                    "chunks": [c.to_dict() for c in chunks],
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except EmbeddingError as exc:
            status_code = 401 if "authentication failed" in str(exc).lower() else 503
            return jsonify({"error": str(exc)}), status_code
        except VectorStoreError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Field 'question' is required and cannot be empty."}), 400

        try:
            top_k = _parse_top_k(payload)
            result = pipeline.answer(question=question, top_k=top_k)
            return jsonify(result.to_dict())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except (EmbeddingError, GenerationError) as exc:
            status_code = 401 if "authentication failed" in str(exc).lower() else 503
            return jsonify({"error": str(exc)}), status_code
        except VectorStoreError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/compare")
    def compare_rag():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Field 'question' is required and cannot be empty."}), 400

        try:
            top_k = _parse_top_k(payload)
            comparison = pipeline.compare(question=question, top_k=top_k)
            return jsonify(comparison)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except (EmbeddingError, GenerationError) as exc:
            status_code = 401 if "authentication failed" in str(exc).lower() else 503
            return jsonify({"error": str(exc)}), status_code
        except VectorStoreError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/upload")
    def upload_documents():
        is_valid, validation_msg = app_cfg.validate_gemini_key()
        if not is_valid:
            return (
                jsonify(
                    {
                        "error": f"Cannot index document: {validation_msg}",
                        "details": [{"filename": "all", "error": validation_msg}],
                    }
                ),
                401,
            )

        if "file" not in request.files and "files" not in request.files:
            return jsonify({"error": "No file part provided in multipart form data."}), 400

        uploaded_files = request.files.getlist("file") + request.files.getlist("files")
        uploaded_files = [f for f in uploaded_files if f and f.filename]
        if not uploaded_files:
            return jsonify({"error": "No selected file to upload."}), 400

        force_raw = str(request.form.get("force", "false")).strip().lower()
        force = force_raw in {"1", "true", "yes"}

        saved_paths: list[Path] = []
        errors: list[dict[str, str]] = []

        for file_storage in uploaded_files:
            safe_name = secure_filename(file_storage.filename or "")
            if not safe_name:
                errors.append({"filename": str(file_storage.filename), "error": "Invalid filename."})
                continue

            ext = Path(safe_name).suffix.lower()
            if ext not in app_cfg.allowed_extensions:
                errors.append(
                    {
                        "filename": safe_name,
                        "error": f"Unsupported file extension '{ext}'. Allowed: {list(app_cfg.allowed_extensions)}",
                    }
                )
                continue

            dest_path = app_cfg.documents_dir / safe_name
            file_storage.save(str(dest_path))
            saved_paths.append(dest_path)

        if not saved_paths and errors:
            return jsonify({"error": "All uploaded files failed validation.", "details": errors}), 400

        all_chunks = []
        for path in saved_paths:
            try:
                pages = load_document(path)
                doc_chunks = chunk_documents(
                    pages,
                    chunk_size_tokens=app_cfg.chunk_size_tokens,
                    chunk_overlap_tokens=app_cfg.chunk_overlap_tokens,
                )
                all_chunks.extend(doc_chunks)
            except Exception as exc:
                errors.append({"filename": path.name, "error": str(exc)})
                # Clean up empty or unreadable uploaded file.
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        if not all_chunks and errors:
            return jsonify({"error": "No extractable text found in uploaded files.", "details": errors}), 400

        try:
            stats = store.add_chunks(all_chunks, embedder=emb, force=force)
        except EmbeddingError as exc:
            for p in saved_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            status_code = 401 if "authentication failed" in str(exc).lower() else 503
            return jsonify({"error": str(exc), "details": errors}), status_code
        except VectorStoreError as exc:
            for p in saved_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            return jsonify({"error": str(exc), "details": errors}), 500

        return (
            jsonify(
                {
                    "message": "Document upload and ingestion completed.",
                    "ingestion": stats.to_dict(),
                    "sample_chunks": inspect_sample_chunks(all_chunks, limit=3),
                    "errors": errors,
                }
            ),
            200,
        )

    @app.post("/api/ingest")
    def ingest_existing_documents():
        is_valid, validation_msg = app_cfg.validate_gemini_key()
        if not is_valid:
            return (
                jsonify(
                    {
                        "error": f"Cannot scan and index folder: {validation_msg}",
                        "ingestion": {
                            "files_ingested": [],
                            "files_skipped": [],
                            "chunks_added": 0,
                            "total_chunks_in_store": store.count(),
                        },
                    }
                ),
                401,
            )

        payload = request.get_json(silent=True) or {}
        force = bool(payload.get("force", False))

        report = load_documents_from_directory(
            app_cfg.documents_dir, app_cfg.allowed_extensions
        )
        if not report.pages:
            return jsonify(
                {
                    "message": "No supported documents found to ingest.",
                    "files_processed": report.files_processed,
                    "skipped_files": report.skipped_files,
                    "ingestion": {
                        "files_ingested": [],
                        "files_skipped": [],
                        "chunks_added": 0,
                        "total_chunks_in_store": store.count(),
                    },
                }
            )

        chunks = chunk_documents(
            report.pages,
            chunk_size_tokens=app_cfg.chunk_size_tokens,
            chunk_overlap_tokens=app_cfg.chunk_overlap_tokens,
        )

        try:
            stats = store.add_chunks(chunks, embedder=emb, force=force)
        except EmbeddingError as exc:
            status_code = 401 if "authentication failed" in str(exc).lower() else 503
            return jsonify({"error": str(exc)}), status_code
        except VectorStoreError as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(
            {
                "message": "Corpus ingestion completed.",
                "files_processed": report.files_processed,
                "load_errors": report.skipped_files,
                "ingestion": stats.to_dict(),
                "sample_chunks": inspect_sample_chunks(chunks, limit=3),
            }
        )

    return app


if __name__ == "__main__":
    config = load_config()
    application = create_app(cfg=config, auto_ingest_on_startup=True)
    logger.info(
        "Starting Valentius RAG Knowledge Assistant on http://%s:%d",
        config.flask_host,
        config.flask_port,
    )
    application.run(
        host=config.flask_host,
        port=config.flask_port,
        debug=config.flask_debug,
    )

