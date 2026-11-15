"""Centralized configuration for the Valentius RAG Knowledge Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)


def _clean_key(raw: Optional[str]) -> str:
    """Strip whitespace, quotes, and placeholder values from API keys."""
    if not raw:
        return ""
    cleaned = raw.strip().strip("\"'").strip()
    placeholders = (
        "your_gemini_api_key",
        "your_actual_gemini_api_key",
        "your_groq_api_key",
        "your_actual_groq_api_key",
        "placeholder",
        "insert_key_here",
    )
    if any(p in cleaned.lower() for p in placeholders):
        return ""
    return cleaned


def _get_gemini_api_key() -> str:
    """Read Gemini API key with fallback to GOOGLE_API_KEY, cleaned of quotes and placeholders."""
    key = _clean_key(os.getenv("GEMINI_API_KEY"))
    if not key:
        key = _clean_key(os.getenv("GOOGLE_API_KEY"))
    return key


def _get_groq_api_key() -> str:
    """Read Groq API key, cleaned of quotes and placeholders."""
    return _clean_key(os.getenv("GROQ_API_KEY"))


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    base_dir: Path = BASE_DIR
    gemini_api_key: str = _get_gemini_api_key()
    groq_api_key: str = _get_groq_api_key()
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
    generation_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    embedding_dimensions: int = _get_int_env("EMBEDDING_DIMENSIONS", 768)

    chunk_size_tokens: int = _get_int_env("CHUNK_SIZE_TOKENS", 500)
    chunk_overlap_tokens: int = _get_int_env("CHUNK_OVERLAP_TOKENS", 75)
    top_k_results: int = _get_int_env("TOP_K_RESULTS", 5)

    documents_dir: Path = BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")
    chroma_db_dir: Path = BASE_DIR / os.getenv("CHROMA_DB_DIR", "data/chroma_db")
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "valentius_knowledge_base"
    ).strip()

    frontend_dir: Path = BASE_DIR / "frontend"

    flask_host: str = os.getenv("FLASK_HOST", "127.0.0.1").strip()
    flask_port: int = _get_int_env("FLASK_PORT", 5000)
    flask_debug: bool = _get_bool_env("FLASK_DEBUG", False)
    max_upload_size_mb: int = _get_int_env("MAX_UPLOAD_SIZE_MB", 20)
    request_timeout_seconds: int = _get_int_env("REQUEST_TIMEOUT_SECONDS", 60)

    allowed_extensions: tuple[str, ...] = (".pdf", ".md", ".txt")

    def ensure_directories(self) -> None:
        """Create required data directories if they do not exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

    def validate_gemini_key(self) -> tuple[bool, str]:
        """Validate Gemini API key for embeddings without exposing secrets."""
        if not self.gemini_api_key:
            return (
                False,
                "GEMINI_API_KEY is not set or is still a placeholder in .env. "
                "Please configure a valid Google AI Studio Gemini API key for embeddings.",
            )
        if len(self.gemini_api_key) < 20:
            return (
                False,
                "The configured GEMINI_API_KEY appears too short. "
                "Please check your API key in .env.",
            )
        return (True, "Gemini API key configured.")

    def validate_groq_key(self) -> tuple[bool, str]:
        """Validate Groq API key for generation without exposing secrets."""
        if not self.groq_api_key:
            return (
                False,
                "GROQ_API_KEY is not set or is still a placeholder in .env. "
                "Please configure a valid Groq API key for LLM generation.",
            )
        if len(self.groq_api_key) < 15:
            return (
                False,
                "The configured GROQ_API_KEY appears too short. "
                "Please check your Groq API key in .env.",
            )
        return (True, "Groq API key configured.")

    def validate_api_key(self) -> tuple[bool, str]:
        """Validate both Gemini and Groq API keys without exposing secrets."""
        is_gemini_valid, gemini_msg = self.validate_gemini_key()
        if not is_gemini_valid:
            return (False, gemini_msg)

        is_groq_valid, groq_msg = self.validate_groq_key()
        if not is_groq_valid:
            return (False, groq_msg)

        return (True, "API keys configured.")

    def to_public_dict(self) -> dict:
        """Return safe configuration metadata without exposing credentials."""
        is_gemini_valid, _ = self.validate_gemini_key()
        is_groq_valid, _ = self.validate_groq_key()
        is_both_valid = is_gemini_valid and is_groq_valid

        return {
            "api_key_configured": is_both_valid,
            "gemini_api_key_configured": is_gemini_valid,
            "groq_api_key_configured": is_groq_valid,
            "generation_model": self.groq_model,
            "groq_model": self.groq_model,
            "groq_base_url": self.groq_base_url,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "top_k_results": self.top_k_results,
            "collection_name": self.chroma_collection_name,
            "allowed_extensions": list(self.allowed_extensions),
            "max_upload_size_mb": self.max_upload_size_mb,
        }


def load_config() -> Config:
    """Reload environment variables and return a fresh Config instance."""
    load_dotenv(BASE_DIR / ".env", override=True)
    return Config(
        gemini_api_key=_get_gemini_api_key(),
        groq_api_key=_get_groq_api_key(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        generation_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        embedding_dimensions=_get_int_env("EMBEDDING_DIMENSIONS", 768),
        chunk_size_tokens=_get_int_env("CHUNK_SIZE_TOKENS", 500),
        chunk_overlap_tokens=_get_int_env("CHUNK_OVERLAP_TOKENS", 75),
        top_k_results=_get_int_env("TOP_K_RESULTS", 5),
        documents_dir=BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents"),
        chroma_db_dir=BASE_DIR / os.getenv("CHROMA_DB_DIR", "data/chroma_db"),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "valentius_knowledge_base"
        ).strip(),
        flask_host=os.getenv("FLASK_HOST", "127.0.0.1").strip(),
        flask_port=_get_int_env("FLASK_PORT", 5000),
        flask_debug=_get_bool_env("FLASK_DEBUG", False),
        max_upload_size_mb=_get_int_env("MAX_UPLOAD_SIZE_MB", 20),
        request_timeout_seconds=_get_int_env("REQUEST_TIMEOUT_SECONDS", 60),
    )
