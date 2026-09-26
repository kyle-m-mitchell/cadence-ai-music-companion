"""Session-safe construction of the stateless Cadence engine for Streamlit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.companion import MusicCompanion
from src.contracts import CatalogDescriptor
from src.embeddings import GeminiEmbedder
from src.factory import CompanionConfig, CompanionDeps, build_companion
from src.generation import GeminiTextGenerator
from src.research import (
    TrackResearchAgent,
    build_optional_research_agent,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeBundle:
    """Stateless recommendation and opt-in research services for one catalog."""

    companion: MusicCompanion
    provider_configured: bool
    research_agent: TrackResearchAgent
    provider_free_research_agent: TrackResearchAgent

    @property
    def catalog_id(self) -> str:
        return self.companion.catalog_id

    @property
    def catalog_descriptor(self) -> CatalogDescriptor | None:
        return self.companion.catalog_descriptor

    @property
    def catalog_artifact_source(self) -> str | None:
        return getattr(self.companion, "catalog_artifact_source", None)

    @property
    def catalog_warnings(self) -> tuple[str, ...]:
        return tuple(getattr(self.companion, "catalog_warnings", ()))


def _dotenv_key() -> str | None:
    """Read only GEMINI_API_KEY from the ignored local file; never mutate env."""
    path = ROOT / ".env"
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "GEMINI_API_KEY":
            candidate = value.strip().strip('"').strip("'")
            return candidate or None
    return None


def _provider_key() -> str | None:
    """Resolve a secret from Cloud settings, environment, or ignored local file."""
    if os.environ.get("CADENCE_DISABLE_PROVIDER") == "1":
        return None  # operational/test kill switch; guarantees no provider objects
    try:
        secret = st.secrets.get("GEMINI_API_KEY")
        if secret:
            return str(secret)
    except Exception:  # no secrets file is the normal local/offline case
        pass
    return os.environ.get("GEMINI_API_KEY") or _dotenv_key()


@st.cache_resource(show_spinner=False)
def get_runtime(catalog_id: str = "fma") -> RuntimeBundle:
    """Build one audited stateless engine per selected catalog.

    The companion stores immutable indexes and provider clients only.  Per-user
    intents, turns, history, receipts, and feedback live in ``st.session_state``;
    the event sink remains the default ``NullEventSink`` so no shared file is read
    or written by the UI.
    """
    if catalog_id not in {"fma", "fictional"}:
        raise ValueError("catalog_id must be 'fma' or 'fictional'")

    key = _provider_key()
    embedder = None
    generator = None
    if key:
        try:
            embedder = GeminiEmbedder(api_key=key, timeout=8.0, max_retries=0)
        except Exception:
            embedder = None
        try:
            generator = GeminiTextGenerator(api_key=key, timeout=8.0, max_retries=0)
        except Exception:
            generator = None

    release_manifest = ROOT / "data" / "catalogs" / "fma-release.manifest.json"
    release_url = os.environ.get("CADENCE_FMA_RELEASE_URL")
    release_enabled = bool(release_url and release_manifest.is_file())
    config = CompanionConfig(
        catalog_id=catalog_id,
        catalog_path=str(ROOT / "data" / "songs.csv"),
        guides_dir=str(ROOT / "data" / "context_guides"),
        catalog_cache_path=str(ROOT / "data" / "embeddings" / "catalog.json"),
        query_cache_path=str(ROOT / "data" / "embeddings" / "queries.json"),
        fma_local_full_path=str(ROOT / "artifacts" / "fma-full.sqlite"),
        fma_local_full_manifest_path=str(
            ROOT / "artifacts" / "fma-full.manifest.json"
        ),
        fma_lite_path=str(ROOT / "data" / "catalogs" / "fma-lite.sqlite"),
        fma_lite_manifest_path=str(
            ROOT / "data" / "catalogs" / "fma-lite.manifest.json"
        ),
        fma_release_url=release_url if release_enabled else None,
        fma_release_manifest_path=(
            str(release_manifest) if release_enabled else None
        ),
        fma_release_cache_path=(
            str(ROOT / ".cache" / "catalogs" / "fma-full.sqlite")
            if release_enabled
            else None
        ),
        use_live_embedder=embedder is not None,
        use_generator=generator is not None,
        event_log_path=None,
    )
    companion = build_companion(
        config,
        CompanionDeps(live_embedder=embedder, generator=generator),
    )
    return RuntimeBundle(
        companion=companion,
        provider_configured=embedder is not None or generator is not None,
        research_agent=build_optional_research_agent(key),
        # Local-only forbids the Gemini research leg. An explicit research click
        # may still resolve the public recording identity through MusicBrainz;
        # the UI discloses that only title and artist leave the session.
                provider_free_research_agent=TrackResearchAgent(
            offline_reason=(
                "Web research is off for this session to protect your privacy. "
                "Showing local FMA evidence."
            )
        ),
    )
