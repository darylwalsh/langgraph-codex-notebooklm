"""Command line entry point: `s1983 run | resume | verify`.

Runs persist in a SQLite checkpoint, so a run paused at human review can be
resumed later (even from another terminal) with its thread id.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from .graph import build_graph, default_dependencies, sqlite_checkpointer
from .verifier import verify_draft


def _out_dir(base: Path, thread_id: str) -> Path:
    d = base / thread_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pending_interrupt(graph, config) -> dict | None:
    snap = graph.get_state(config)
    for task in snap.tasks:
        for intr in task.interrupts:
            return intr.value
    return None


def _stream(graph, payload, config) -> None:
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                continue
            for line in (delta or {}).get("log", []):
                print(f"  · {line}")


def _report(graph, config, out: Path) -> int:
    pending = _pending_interrupt(graph, config)
    state = graph.get_state(config).values
    if pending is not None:
        (out / "draft.md").write_text(pending["draft"], encoding="utf-8")
        (out / "review.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")
        if memo := state.get("research_memo"):
            (out / "research_memo.json").write_text(memo.model_dump_json(indent=2), encoding="utf-8")
        status = "PASSED" if pending["verification_passed"] else "FAILED"
        print(f"\nPaused for human review. Verification {status}. Judge: {pending['judge_verdict']}.")
        for b in pending["verification_blocking"]:
            print(f"  ✗ {b}")
        for w in pending["verification_warnings"]:
            print(f"  ! {w}")
        print(f"\nDraft: {out / 'draft.md'}\nDetails: {out / 'review.json'}")
        return 2
    if final := state.get("final_complaint"):
        (out / "complaint.md").write_text(final, encoding="utf-8")
        print(f"\nApproved. Complaint written to {out / 'complaint.md'}")
        return 0
    print("\nRun ended without an approved complaint (rejected).")
    return 1


def _ask() -> dict:
    action = ""
    while action not in {"approve", "revise", "reject"}:
        action = input("\nDecision [approve/revise/reject]: ").strip().lower()
    feedback = input("Feedback for the drafter: ").strip() if action == "revise" else ""
    return {"action": action, "feedback": feedback}


def cmd_run(args: argparse.Namespace) -> int:
    llm, cl, corpus, s = default_dependencies()
    if cl is None:
        print("warning: COURTLISTENER_API_TOKEN not set; case law cannot be retrieved or verified.", file=sys.stderr)
    if not corpus.docs:
        print(
            "warning: no verified files in data/corpus/; statute and rule citations will fail verification.",
            file=sys.stderr,
        )

    thread_id = args.thread_id or uuid.uuid4().hex[:12]
    out = _out_dir(args.out, thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    with sqlite_checkpointer(s.checkpoint_db) as saver:
        graph = build_graph(llm, cl, corpus, s, checkpointer=saver)
        print(f"thread: {thread_id}")
        payload = {"case_facts": Path(args.facts).read_text(encoding="utf-8")}
        _stream(graph, payload, config)
        code = _report(graph, config, out)
        while code == 2 and args.interactive:
            _stream(graph, Command(resume=_ask()), config)
            code = _report(graph, config, out)
        if code == 2:
            print(
                f"\nResume with: uv run s1983 resume --thread-id {thread_id} "
                "--action approve|revise|reject [--feedback ...]"
            )
        return code


def cmd_resume(args: argparse.Namespace) -> int:
    llm, cl, corpus, s = default_dependencies()
    out = _out_dir(args.out, args.thread_id)
    config = {"configurable": {"thread_id": args.thread_id}, "recursion_limit": 60}
    with sqlite_checkpointer(s.checkpoint_db) as saver:
        graph = build_graph(llm, cl, corpus, s, checkpointer=saver)
        if _pending_interrupt(graph, config) is None:
            print(f"No paused run for thread {args.thread_id}.", file=sys.stderr)
            return 1
        _stream(graph, Command(resume={"action": args.action, "feedback": args.feedback or ""}), config)
        return _report(graph, config, out)


def cmd_verify(args: argparse.Namespace) -> int:
    _, cl, corpus, _ = default_dependencies()
    text = Path(args.file).read_text(encoding="utf-8")
    report = verify_draft(text, None, corpus.citation_keys, cl, require_grounding=False)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        for c in report.checks:
            mark = {"verified": "✓", "skipped": "·"}.get(c.status, "✗")
            print(f"{mark} {c.status:<13} {c.text:<40} {c.detail}")
        for b in report.blocking:
            print(f"BLOCKING: {b}")
        for w in report.warnings:
            print(f"warning:  {w}")
        print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="s1983", description="§ 1983 multi-agent drafting pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run the full pipeline on a facts file")
    r.add_argument("--facts", required=True, help="Markdown/text file with the case facts")
    r.add_argument("--thread-id", help="Reuse a thread id (default: random)")
    r.add_argument("--out", type=Path, default=Path("out"))
    r.add_argument(
        "--no-interactive", dest="interactive", action="store_false", help="Stop at human review instead of prompting"
    )
    r.set_defaults(func=cmd_run)

    rs = sub.add_parser("resume", help="Answer a paused human review")
    rs.add_argument("--thread-id", required=True)
    rs.add_argument("--action", required=True, choices=["approve", "revise", "reject"])
    rs.add_argument("--feedback")
    rs.add_argument("--out", type=Path, default=Path("out"))
    rs.set_defaults(func=cmd_resume)

    v = sub.add_parser("verify", help="Run only the citation gate on any document")
    v.add_argument("file")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
