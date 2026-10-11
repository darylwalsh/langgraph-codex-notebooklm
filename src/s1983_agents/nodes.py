"""Node factories for the six agents plus the human review gate.

Each factory closes over its dependencies (LLM, CourtListener, corpus) and
returns a plain `state -> partial state` function, so the graph wiring stays
declarative and tests can inject fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langgraph.types import interrupt

from . import prompts
from .config import Settings
from .corpus import Corpus
from .courtlistener import CourtListener, CourtListenerError
from .llm import LLM
from .state import (
    AdvocateBrief,
    Authority,
    ComplaintDraft,
    GraphState,
    HumanDecision,
    JudgeRuling,
    MemoPoint,
    ResearchMemo,
    ResearchMemoDraft,
    ResearchPlan,
)
from .verifier import verify_draft

log = logging.getLogger(__name__)
Node = Callable[[GraphState], dict]


# --------------------------------------------------------------- rendering


def render_authorities(auths: dict[str, Authority], with_excerpts: bool = True) -> str:
    lines = []
    for a in auths.values():
        head = f"[{a.id}] {a.title} | {a.citation} | {a.kind}"
        head += f" | {a.court or ''} {a.date or ''}".rstrip()
        if a.as_of:
            head += f" | text as of {a.as_of}"
        lines.append(head)
        if with_excerpts and a.excerpt:
            lines.append(f"    excerpt: {a.excerpt[:800]}")
    return "\n".join(lines) or "(none)"


def render_memo(memo: ResearchMemo) -> str:
    def block(title: str, points: list[MemoPoint]) -> str:
        body = "\n".join(
            f"- {p.proposition} [{', '.join(p.authority_ids) or 'NO SUPPORT'}]" + (f" ({p.notes})" if p.notes else "")
            for p in points
        )
        return f"## {title}\n{body or '(none)'}"

    return "\n\n".join(
        [
            block("Claims and elements", memo.claims_and_elements),
            block("Exhaustion and availability", memo.exhaustion_and_availability),
            block("Defenses to anticipate", memo.defenses_to_anticipate),
            "## Open questions\n" + ("\n".join(f"- {q}" for q in memo.open_questions) or "(none)"),
            "## AUTHORITIES\n" + render_authorities(memo.authorities),
        ]
    )


def render_brief(b: AdvocateBrief) -> str:
    def lst(xs: list[str]) -> str:
        return "\n".join(f"- {x}" for x in xs) or "(none)"

    return (
        f"### {b.side.title()} advocate\nStrongest points:\n{lst(b.strongest_points)}\n"
        f"Weaknesses in opponent:\n{lst(b.weaknesses_in_opponent)}\n"
        f"Facts needed:\n{lst(b.facts_needed)}"
    )


# --------------------------------------------------------------- grounding


def _ground_points(points: list[MemoPoint], allowed: set[str], dropped: set[str]) -> list[MemoPoint]:
    out = []
    for p in points:
        bad = [i for i in p.authority_ids if i not in allowed]
        dropped.update(bad)
        out.append(p.model_copy(update={"authority_ids": [i for i in p.authority_ids if i in allowed]}))
    return out


def ground_memo(draft: ResearchMemoDraft, candidates: dict[str, Authority], seed_ids: set[str]) -> ResearchMemo:
    """Drop any authority id the model invented; keep only retrieved records."""
    allowed = set(candidates)
    dropped: set[str] = set()
    claims = _ground_points(draft.claims_and_elements, allowed, dropped)
    exhaust = _ground_points(draft.exhaustion_and_availability, allowed, dropped)
    defenses = _ground_points(draft.defenses_to_anticipate, allowed, dropped)
    used = {i for p in (*claims, *exhaust, *defenses) for i in p.authority_ids} | seed_ids
    return ResearchMemo(
        claims_and_elements=claims,
        exhaustion_and_availability=exhaust,
        defenses_to_anticipate=defenses,
        open_questions=draft.open_questions,
        authorities={i: candidates[i] for i in candidates if i in used},
        dropped_ids=sorted(dropped),
    )


# --------------------------------------------------------------- Agent 1


def make_research(llm: LLM, cl: CourtListener | None, corpus: Corpus, s: Settings) -> Node:
    def research(state: GraphState) -> dict:
        facts = state["case_facts"]
        plan = llm.structured(
            "research",
            ResearchPlan,
            prompts.RESEARCH_PLANNER.format(max_queries=s.max_research_queries),
            f"CASE FACTS:\n{facts}",
        )
        candidates: dict[str, Authority] = {}
        seed_ids: set[str] = set()
        notes: list[str] = []

        if cl is None:
            notes.append("CourtListener not configured: case law retrieval skipped.")
        else:
            try:
                for a in cl.resolve_citations(list(s.seed_citations)):
                    candidates[a.id] = a
                    seed_ids.add(a.id)
                for q in plan.queries[: s.max_research_queries]:
                    for a in cl.search_opinions(q, s.courts, s.results_per_query):
                        candidates.setdefault(a.id, a)
            except CourtListenerError as e:
                notes.append(f"CourtListener error during research: {e}")

        for a in corpus.search(facts + "\n" + "\n".join(plan.queries), k=8):
            candidates.setdefault(a.id, a)

        draft = llm.structured(
            "research",
            ResearchMemoDraft,
            prompts.RESEARCH_SYNTHESIS,
            f"CASE FACTS:\n{facts}\n\nCANDIDATE AUTHORITIES:\n{render_authorities(candidates)}",
        )
        memo = ground_memo(draft, candidates, seed_ids)
        if memo.dropped_ids:
            notes.append(f"dropped {len(memo.dropped_ids)} invented authority id(s): {memo.dropped_ids}")
        msg = (
            f"research: {len(plan.queries)} queries, {len(candidates)} candidates, "
            f"{len(memo.authorities)} authorities kept"
        )
        return {"research_memo": memo, "log": [msg, *(f"research: {n}" for n in notes)]}

    return research


# --------------------------------------------------------------- Agents 3 & 4


def _filter_ids(ids: list[str], memo: ResearchMemo) -> list[str]:
    return [i for i in ids if i in memo.authorities]


def make_advocate(llm: LLM, side: str) -> Node:
    role, system, key = (
        ("plaintiff", prompts.PLAINTIFF, "plaintiff_brief")
        if side == "plaintiff"
        else ("defense", prompts.DEFENSE, "defense_brief")
    )

    def advocate(state: GraphState) -> dict:
        memo = state["research_memo"]
        brief = llm.structured(
            role,
            AdvocateBrief,
            system,
            f"CASE FACTS:\n{state['case_facts']}\n\nRESEARCH MEMO:\n{render_memo(memo)}",
        )
        brief = brief.model_copy(update={"side": side, "authority_ids": _filter_ids(brief.authority_ids, memo)})
        return {key: brief, "log": [f"{side}: {len(brief.strongest_points)} points"]}

    advocate.__name__ = f"{side}_advocate"
    return advocate


# --------------------------------------------------------------- Agent 6


def make_drafter(llm: LLM) -> Node:
    def drafter(state: GraphState) -> dict:
        memo = state["research_memo"]
        parts = [
            f"CASE FACTS:\n{state['case_facts']}",
            f"RESEARCH MEMO:\n{render_memo(memo)}",
            render_brief(state["plaintiff_brief"]),
            render_brief(state["defense_brief"]),
        ]
        if prev := state.get("draft"):
            parts.append(f"PREVIOUS DRAFT:\n{prev.markdown}")
        ruling = state.get("judge_ruling")
        if ruling and ruling.verdict == "revise":
            issues = "\n".join(
                f"- [{i.category}{'/' + i.defendant if i.defendant else ''}] {i.problem} -> FIX: {i.fix}"
                for i in ruling.issues
            )
            parts.append(f"JUDGE ISSUES:\n{issues}")
        ver = state.get("verification")
        if ver and not ver.passed:
            parts.append("VERIFICATION FAILURES:\n" + "\n".join(f"- {b}" for b in ver.blocking))
        hd = state.get("human_decision")
        if hd and hd.action == "revise":
            parts.append(f"HUMAN FEEDBACK:\n{hd.feedback}")

        draft = llm.structured("drafter", ComplaintDraft, prompts.DRAFTER, "\n\n".join(parts))
        draft = draft.model_copy(update={"authority_ids": _filter_ids(draft.authority_ids, memo)})
        return {"draft": draft, "log": [f"drafter: {len(draft.markdown)} chars; {draft.change_log[:120]}"]}

    return drafter


# --------------------------------------------------------------- Agent 2


def make_judge(llm: LLM) -> Node:
    def judge(state: GraphState) -> dict:
        ruling = llm.structured(
            "judge",
            JudgeRuling,
            prompts.JUDGE,
            "\n\n".join(
                [
                    f"CASE FACTS:\n{state['case_facts']}",
                    f"AUTHORITIES:\n{render_authorities(state['research_memo'].authorities, with_excerpts=False)}",
                    render_brief(state["plaintiff_brief"]),
                    render_brief(state["defense_brief"]),
                    f"DRAFT COMPLAINT:\n{state['draft'].markdown}",
                ]
            ),
        )
        # A "pass" with issues listed is incoherent; treat it as revise.
        if ruling.verdict == "pass" and ruling.issues:
            ruling = ruling.model_copy(update={"verdict": "revise"})
        n = state.get("judge_revisions", 0) + (1 if ruling.verdict == "revise" else 0)
        return {
            "judge_ruling": ruling,
            "judge_revisions": n,
            "log": [f"judge: {ruling.verdict} ({len(ruling.issues)} issues)"],
        }

    return judge


# --------------------------------------------------------------- Agent 5


def make_verifier(cl: CourtListener | None, corpus: Corpus) -> Node:
    def verifier(state: GraphState) -> dict:
        report = verify_draft(state["draft"].markdown, state.get("research_memo"), corpus.citation_keys, cl)
        attempts = state.get("verify_attempts", 0) + (0 if report.passed else 1)
        return {
            "verification": report,
            "verify_attempts": attempts,
            "log": [f"verifier: {'PASS' if report.passed else 'FAIL'} ({len(report.blocking)} blocking)"],
        }

    return verifier


# --------------------------------------------------------------- human gate


def human_review(state: GraphState) -> dict:
    """Pause the graph. Resume with Command(resume={"action": ..., "feedback": ...})."""
    ruling = state.get("judge_ruling")
    ver = state.get("verification")
    answer = interrupt(
        {
            "draft": state["draft"].markdown,
            "judge_verdict": ruling.verdict if ruling else None,
            "unresolved_judge_issues": [i.model_dump() for i in ruling.issues] if ruling else [],
            "verification_passed": ver.passed if ver else False,
            "verification_blocking": ver.blocking if ver else [],
            "verification_warnings": ver.warnings if ver else [],
            "instructions": "Reply with action approve | revise | reject, plus feedback for revise.",
        }
    )
    decision = HumanDecision.model_validate(answer)
    update: dict = {"human_decision": decision, "log": [f"human: {decision.action}"]}
    if decision.action == "revise":
        # Fresh loop budget for the new round of human-directed changes.
        update |= {"judge_revisions": 0, "verify_attempts": 0}
    return update


def finalize(state: GraphState) -> dict:
    """Assemble the approved complaint. Unresolved problems are stamped on top."""
    ver = state.get("verification")
    ruling = state.get("judge_ruling")
    banner = []
    if ver and not ver.passed:
        banner += [f"> VERIFICATION FAILED: {b}" for b in ver.blocking]
    if ruling and ruling.verdict == "revise":
        banner += [f"> JUDGE ISSUE ({i.category}): {i.problem}" for i in ruling.issues]
    text = state["draft"].markdown
    if banner:
        text = (
            "> **Approved with unresolved issues. Do not file until these are cleared.**\n"
            + "\n".join(banner)
            + "\n\n"
            + text
        )
    return {"final_complaint": text, "log": ["finalize: done"]}
