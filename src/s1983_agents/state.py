"""Typed graph state and the structured outputs each agent must return.

Design rule: citation strings that reach the complaint are never authored by a
model. Agent 1 *selects* authorities by id from candidates that came back from
CourtListener or the local primary-source corpus; the text of each citation is
copied from that source record. Agent 5 then re-checks everything in the draft.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# --------------------------------------------------------------- source records


class Authority(BaseModel):
    """A retrieved, source-backed authority. Built by code, not by a model."""

    id: str = Field(description="Stable id, e.g. 'cl:2812209' or 'corpus:mdoc-pd-03.02.130'")
    kind: Literal["case", "statute", "regulation", "court_rule", "policy"]
    title: str = Field(description="Case name or document title")
    citation: str = Field(description="Citation string exactly as the source reports it")
    court: str | None = None
    date: str | None = None
    url: str | None = None
    excerpt: str = Field(default="", description="Snippet/passage the source returned")
    as_of: str | None = Field(default=None, description="Version date for statutes/policies")


# ------------------------------------------------------------ agent outputs


class ResearchPlan(BaseModel):
    queries: list[str] = Field(description="Short CourtListener full-text queries")


class MemoPoint(BaseModel):
    proposition: str = Field(description="One legal proposition, stated plainly")
    authority_ids: list[str] = Field(description="Ids of candidate authorities that support it")
    notes: str = ""


class ResearchMemoDraft(BaseModel):
    """What the research model returns: propositions keyed to candidate ids."""

    claims_and_elements: list[MemoPoint]
    exhaustion_and_availability: list[MemoPoint]
    defenses_to_anticipate: list[MemoPoint]
    open_questions: list[str] = Field(default_factory=list)


class ResearchMemo(BaseModel):
    """ResearchMemoDraft after code has dropped any id that wasn't retrieved."""

    claims_and_elements: list[MemoPoint]
    exhaustion_and_availability: list[MemoPoint]
    defenses_to_anticipate: list[MemoPoint]
    open_questions: list[str] = Field(default_factory=list)
    authorities: dict[str, Authority] = Field(default_factory=dict)
    dropped_ids: list[str] = Field(default_factory=list)


class AdvocateBrief(BaseModel):
    side: Literal["plaintiff", "defense"]
    strongest_points: list[str]
    weaknesses_in_opponent: list[str]
    facts_needed: list[str] = Field(description="Facts that would change the outcome if pleaded")
    authority_ids: list[str] = Field(default_factory=list)


class JudgeIssue(BaseModel):
    category: Literal[
        "plausibility",
        "personal_involvement",
        "qualified_immunity",
        "eleventh_amendment",
        "exhaustion_framing",
        "screening_frivolous",
        "form_and_local_rules",
        "other",
    ]
    defendant: str | None = None
    problem: str
    fix: str


class JudgeRuling(BaseModel):
    verdict: Literal["pass", "revise"]
    standard_applied: str = Field(description="e.g. '28 U.S.C. § 1915A screening / Rule 12(b)(6) plausibility'")
    issues: list[JudgeIssue] = Field(default_factory=list)
    summary: str


class ComplaintDraft(BaseModel):
    markdown: str = Field(description="Full complaint text in Markdown")
    authority_ids: list[str] = Field(description="Ids of every authority cited in the draft")
    change_log: str = Field(default="", description="What changed vs the previous draft")


# ------------------------------------------------------------ verification


class CitationCheck(BaseModel):
    text: str
    kind: Literal["case", "statute", "regulation", "court_rule", "policy", "short_form"]
    status: Literal[
        "verified",  # resolved in CourtListener and grounded in the memo
        "not_found",  # valid format, not in CourtListener: probable hallucination
        "ambiguous",  # resolves to several clusters
        "name_mismatch",  # reporter cite is real but belongs to a different case
        "ungrounded",  # real or not, it was never retrieved by Agent 1
        "skipped",  # short forms (Id., supra) that point at earlier cites
        "error",
    ]
    detail: str = ""
    matched_authority_id: str | None = None
    url: str | None = None


class VerificationReport(BaseModel):
    passed: bool
    checks: list[CitationCheck]
    blocking: list[str] = Field(description="Human-readable reasons the gate failed")
    warnings: list[str] = Field(default_factory=list)


class HumanDecision(BaseModel):
    action: Literal["approve", "revise", "reject"]
    feedback: str = ""


# ------------------------------------------------------------------ graph state


class GraphState(TypedDict, total=False):
    # Inputs
    case_facts: str
    # Agent 1
    research_memo: ResearchMemo
    # Agents 3 & 4 (run in parallel; distinct keys so no reducer is needed)
    plaintiff_brief: AdvocateBrief
    defense_brief: AdvocateBrief
    # Agent 6
    draft: ComplaintDraft
    # Agent 2
    judge_ruling: JudgeRuling
    judge_revisions: int
    # Agent 5
    verification: VerificationReport
    verify_attempts: int
    # Human gate
    human_decision: HumanDecision
    final_complaint: str
    # Append-only audit trail of what each node did
    log: Annotated[list[str], operator.add]
