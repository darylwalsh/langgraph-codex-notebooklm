"""End-to-end graph runs with a scripted LLM and mock CourtListener (fully offline)."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from s1983_agents.graph import build_graph, serializer

from .conftest import FakeLLM

CLEAN = (
    "# Complaint\n\n1. Defendant Officer A destroyed Plaintiff's grievance on [FACT NEEDED: date].\n\n"
    "Remedies were unavailable. Ross v. Blake, 578 U.S. 632, 643-44 (2016); 42 U.S.C. § 1997e(a).\n\n"
    "Count I arises under 42 U.S.C. § 1983."
)
WITH_FAKE_CITE = CLEAN + "\n\nSee also Smith v. Warden, 999 F.4th 123 (6th Cir. 2025)."


def research(schema, system, user, n):
    if schema.__name__ == "ResearchPlan":
        return {"queries": ['"exhaustion" "unavailable" grievance MDOC']}
    point = {"proposition": "Remedies thwarted by officials are unavailable", "authority_ids": ["cl:3195917", "cl:000"]}
    return {
        "claims_and_elements": [],
        "exhaustion_and_availability": [point],
        "defenses_to_anticipate": [],
        "open_questions": ["Which version of PD 03.02.130 applied?"],
    }


def advocate(side):
    def f(schema, system, user, n):
        return {
            "side": side,
            "strongest_points": [f"{side} point"],
            "weaknesses_in_opponent": [],
            "facts_needed": [],
            "authority_ids": ["cl:3195917", "cl:invented"],
        }

    return f


def drafter_script(texts):
    def f(schema, system, user, n):
        text = texts[min(n, len(texts) - 1)]
        return {"markdown": text, "authority_ids": ["cl:3195917"], "change_log": f"draft {n}"}

    return f


def judge_script(verdicts):
    def f(schema, system, user, n):
        v = verdicts[min(n, len(verdicts) - 1)]
        issues = (
            [
                {
                    "category": "personal_involvement",
                    "defendant": "Officer A",
                    "problem": "conclusory",
                    "fix": "add who/when",
                }
            ]
            if v == "revise"
            else []
        )
        return {"verdict": v, "standard_applied": "§ 1915A / 12(b)(6)", "issues": issues, "summary": v}

    return f


def make(llm, cl, corpus, settings):
    return build_graph(llm, cl, corpus, settings, checkpointer=InMemorySaver(serde=serializer()))


def run_to_interrupt(graph, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}}
    graph.invoke({"case_facts": "Officer A tore up my grievance and threatened me."}, cfg)
    return cfg


def test_judge_loop_then_verify_loop_then_approve(cl, corpus, settings):
    llm = FakeLLM(
        {
            "research": research,
            "plaintiff": advocate("plaintiff"),
            "defense": advocate("defense"),
            "drafter": drafter_script(["draft0", WITH_FAKE_CITE, CLEAN]),
            "judge": judge_script(["revise", "pass", "pass"]),
        }
    )
    g = make(llm, cl, corpus, settings)
    cfg = run_to_interrupt(g)

    snap = g.get_state(cfg)
    assert snap.next == ("human_review",)
    payload = snap.tasks[0].interrupts[0].value
    assert payload["verification_passed"] is True
    assert any("FACT NEEDED" in w for w in payload["verification_warnings"])

    v = snap.values
    assert "cl:000" in v["research_memo"].dropped_ids  # invented id removed by code
    assert v["plaintiff_brief"].authority_ids == ["cl:3195917"]
    assert llm.calls["drafter"].count("ComplaintDraft") == 3
    # The last redraft was triggered by the verifier, so its failures were fed back.
    assert "VERIFICATION FAILURES" in llm.last_user[("drafter", "ComplaintDraft")]
    log = v["log"]
    assert [line.split(":")[0] for line in log if line.split(":")[0] in {"judge", "verifier"}] == [
        "judge",
        "judge",
        "verifier",
        "judge",
        "verifier",
    ]
    assert "verifier: FAIL" in "\n".join(log)

    g.invoke(Command(resume={"action": "approve"}), cfg)
    final = g.get_state(cfg).values["final_complaint"]
    assert final == CLEAN


def test_judge_budget_exhausted_reaches_human_with_banner(cl, corpus, settings):
    llm = FakeLLM(
        {
            "research": research,
            "plaintiff": advocate("plaintiff"),
            "defense": advocate("defense"),
            "drafter": drafter_script([CLEAN]),
            "judge": judge_script(["revise"]),
        }
    )
    g = make(llm, cl, corpus, settings)
    cfg = run_to_interrupt(g)
    # initial draft + max_judge_revisions (2) redrafts
    assert llm.calls["drafter"].count("ComplaintDraft") == settings.max_judge_revisions + 1
    g.invoke(Command(resume={"action": "approve"}), cfg)
    final = g.get_state(cfg).values["final_complaint"]
    assert final.startswith("> **Approved with unresolved issues")
    assert "JUDGE ISSUE (personal_involvement)" in final


def test_verify_budget_exhausted_reaches_human(cl, corpus, settings):
    llm = FakeLLM(
        {
            "research": research,
            "plaintiff": advocate("plaintiff"),
            "defense": advocate("defense"),
            "drafter": drafter_script([WITH_FAKE_CITE]),
            "judge": judge_script(["pass"]),
        }
    )
    g = make(llm, cl, corpus, settings)
    cfg = run_to_interrupt(g)
    payload = g.get_state(cfg).tasks[0].interrupts[0].value
    assert payload["verification_passed"] is False
    assert any("999 F.4th 123" in b for b in payload["verification_blocking"])
    assert llm.calls["drafter"].count("ComplaintDraft") == settings.max_verify_retries + 1


def test_human_revise_feeds_back_then_reject(cl, corpus, settings):
    llm = FakeLLM(
        {
            "research": research,
            "plaintiff": advocate("plaintiff"),
            "defense": advocate("defense"),
            "drafter": drafter_script([CLEAN]),
            "judge": judge_script(["pass"]),
        }
    )
    g = make(llm, cl, corpus, settings)
    cfg = run_to_interrupt(g)
    g.invoke(Command(resume={"action": "revise", "feedback": "Name the housing unit."}), cfg)
    assert "HUMAN FEEDBACK:\nName the housing unit." in llm.last_user[("drafter", "ComplaintDraft")]
    assert g.get_state(cfg).next == ("human_review",)

    g.invoke(Command(resume={"action": "reject"}), cfg)
    snap = g.get_state(cfg)
    assert snap.next == ()
    assert "final_complaint" not in snap.values


def test_judge_feedback_reaches_drafter(cl, corpus, settings):
    seen = []

    def drafter(schema, system, user, n):
        seen.append(user)
        return {"markdown": CLEAN, "authority_ids": [], "change_log": ""}

    llm = FakeLLM(
        {
            "research": research,
            "plaintiff": advocate("plaintiff"),
            "defense": advocate("defense"),
            "drafter": drafter,
            "judge": judge_script(["revise", "pass"]),
        }
    )
    run_to_interrupt(make(llm, cl, corpus, settings))
    assert "JUDGE ISSUES" not in seen[0]
    assert "[personal_involvement/Officer A] conclusory -> FIX: add who/when" in seen[1]
