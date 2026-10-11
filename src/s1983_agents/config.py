"""Runtime settings, read from the environment (mise loads .env + mise.toml [env])."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["research", "plaintiff", "defense", "judge", "drafter"]

# Courts whose opinions are binding or directly relevant in E.D. Mich. (Bay City).
# CourtListener court ids: U.S. Supreme Court, Sixth Circuit, E.D. Mich., W.D. Mich.
DEFAULT_COURTS = ("scotus", "ca6", "mied", "miwd")

# Authorities every run must resolve, regardless of what the research planner asks for.
DEFAULT_SEED_CITATIONS = (
    "Ross v. Blake, 578 U.S. 632 (2016)",
    "Jones v. Bock, 549 U.S. 199 (2007)",
    "Ashcroft v. Iqbal, 556 U.S. 662 (2009)",
    "Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007)",
    "Will v. Michigan Dept. of State Police, 491 U.S. 58 (1989)",
    "Haines v. Kerner, 404 U.S. 519 (1972)",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="S1983_", extra="ignore")

    # Per-role models in langchain `init_chat_model` form: "provider:model".
    model_research: str = "anthropic:claude-sonnet-5-5"
    model_plaintiff: str = "anthropic:claude-opus-5-5"
    model_defense: str = "anthropic:claude-opus-5-5"
    model_judge: str = "anthropic:claude-opus-5-5"
    model_drafter: str = "anthropic:claude-opus-5-5"
    temperature: float = 0.0

    # Loop limits.
    max_judge_revisions: int = 3
    max_verify_retries: int = 2

    # Retrieval.
    courts: tuple[str, ...] = DEFAULT_COURTS
    seed_citations: tuple[str, ...] = DEFAULT_SEED_CITATIONS
    results_per_query: int = 8
    max_research_queries: int = 6
    corpus_dir: Path = Path("data/corpus")

    # CourtListener. Token is read WITHOUT the S1983_ prefix.
    courtlistener_api_token: SecretStr | None = Field(default=None, validation_alias="COURTLISTENER_API_TOKEN")
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4"

    # Persistence for the human-review interrupt (resume a thread later).
    checkpoint_db: Path = Path(".state/checkpoints.sqlite")

    def model_for(self, role: Role) -> str:
        return getattr(self, f"model_{role}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
