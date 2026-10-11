"""Graph wiring.

START
  │
  ▼
research (Agent 1) ──┬──► plaintiff_advocate (Agent 3) ──┐
                     └──► defense_advocate   (Agent 4) ──┤  (parallel)
                                                         ▼
                    ┌──────────────────────────────► drafter (Agent 6)
                    │                                    │
                    │ revise (≤ max_judge_revisions)     ▼
                    ├──────────────────────────────── judge (Agent 2)
                    │                                    │ pass / budget spent
                    │ fail (≤ max_verify_retries)        ▼
                    ├──────────────────────────────── verifier (Agent 5: eyecite + CourtListener)
                    │                                    │ pass / budget spent
                    │ revise                             ▼
                    └──────────────────────────────── human_review (interrupt)
                                                         │ approve        │ reject
                                                         ▼                ▼
                                                      finalize ──► END   END
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from . import state as state_types
from .config import Settings, get_settings
from .corpus import Corpus
from .courtlistener import CourtListener
from .llm import LLM, LangChainLLM
from .state import GraphState

# Pydantic types stored in checkpoints. LangGraph only deserializes allowlisted
# types, so keep this list in sync with the models used in GraphState.
CHECKPOINT_TYPES = [
    (state_types.__name__, name)
    for name in (
        "Authority",
        "MemoPoint",
        "ResearchMemo",
        "AdvocateBrief",
        "JudgeIssue",
        "JudgeRuling",
        "ComplaintDraft",
        "CitationCheck",
        "VerificationReport",
        "HumanDecision",
    )
]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """Durable checkpoints so a run paused at human review survives restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=serializer())
    finally:
        conn.close()


def route_after_judge(state: GraphState, max_revisions: int) -> Literal["drafter", "verifier"]:
    ruling = state["judge_ruling"]
    if ruling.verdict == "revise" and state.get("judge_revisions", 0) <= max_revisions:
        return "drafter"
    # Passed, or out of budget: go verify; unresolved issues reach the human.
    return "verifier"


def route_after_verify(state: GraphState, max_retries: int) -> Literal["drafter", "human_review"]:
    report = state["verification"]
    if not report.passed and state.get("verify_attempts", 0) <= max_retries:
        return "drafter"
    return "human_review"


def route_after_human(state: GraphState) -> Literal["drafter", "finalize", "__end__"]:
    action = state["human_decision"].action
    return {"approve": "finalize", "revise": "drafter"}.get(action, END)


def build_graph(
    llm: LLM,
    cl: CourtListener | None,
    corpus: Corpus,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
):
    g = StateGraph(GraphState)

    g.add_node("research", nodes.make_research(llm, cl, corpus, settings))
    g.add_node("plaintiff_advocate", nodes.make_advocate(llm, "plaintiff"))
    g.add_node("defense_advocate", nodes.make_advocate(llm, "defense"))
    g.add_node("drafter", nodes.make_drafter(llm))
    g.add_node("judge", nodes.make_judge(llm))
    g.add_node("verifier", nodes.make_verifier(cl, corpus))
    g.add_node("human_review", nodes.human_review)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "research")
    # Fan out to both advocates, then join: drafter waits for both.
    g.add_edge("research", "plaintiff_advocate")
    g.add_edge("research", "defense_advocate")
    g.add_edge(["plaintiff_advocate", "defense_advocate"], "drafter")

    g.add_edge("drafter", "judge")
    g.add_conditional_edges(
        "judge",
        lambda s: route_after_judge(s, settings.max_judge_revisions),
        ["drafter", "verifier"],
    )
    g.add_conditional_edges(
        "verifier",
        lambda s: route_after_verify(s, settings.max_verify_retries),
        ["drafter", "human_review"],
    )
    g.add_conditional_edges("human_review", route_after_human, ["drafter", "finalize", END])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


def default_dependencies(settings: Settings | None = None) -> tuple[LLM, CourtListener | None, Corpus, Settings]:
    s = settings or get_settings()
    token = s.courtlistener_api_token.get_secret_value() if s.courtlistener_api_token else None
    cl = CourtListener(token, s.courtlistener_base_url) if token else None
    return LangChainLLM(s), cl, Corpus.load(s.corpus_dir), s


def make_graph():
    """Entry point for `langgraph dev` (LangGraph Studio supplies its own checkpointer)."""
    llm, cl, corpus, s = default_dependencies()
    return build_graph(llm, cl, corpus, s)
