"""Agent 5: the verification gate. Deterministic: no model is involved.

A draft passes only if every citation in it is
  (a) grounded: it was retrieved by Agent 1 (memo authority) or is in the local
      primary-source corpus, and
  (b) real: case citations resolve in CourtListener to a case whose name
      matches the parties in the cite.

Anything else blocks. Warnings don't block but are shown to the human reviewer.
"""

from __future__ import annotations

import re

from . import citations as cites
from .courtlistener import CourtListener, CourtListenerError, LookupResult
from .state import Authority, CitationCheck, ResearchMemo, VerificationReport

FACT_NEEDED = re.compile(r"\[FACT NEEDED[^\]]*\]")


def _lookup_index(results: list[LookupResult]) -> dict[str, LookupResult]:
    """Index CourtListener results by our normalized case key."""
    out: dict[str, LookupResult] = {}
    for r in results:
        for s in [*r.normalized, r.citation]:
            for k in cites.keys(s):
                out.setdefault(k, r)
    return out


def _memo_keys(authorities: dict[str, Authority]) -> dict[str, str]:
    """citation key -> authority id, for everything Agent 1 retrieved."""
    out: dict[str, str] = {}
    for a in authorities.values():
        for k in cites.keys(a.citation):
            out.setdefault(k, a.id)
    return out


def verify_draft(
    text: str,
    memo: ResearchMemo | None,
    corpus_keys: set[str],
    cl: CourtListener | None,
    *,
    require_grounding: bool = True,
) -> VerificationReport:
    """Check every citation in `text`.

    require_grounding=False is for checking an arbitrary document (`s1983 verify`):
    cases only need to exist in CourtListener, and statutes missing from the
    corpus become warnings instead of blocking.
    """
    authorities = memo.authorities if memo else {}
    grounded = _memo_keys(authorities)
    statute_ok = corpus_keys | {k for k, aid in grounded.items() if not k.startswith("case:")}

    extracted = cites.extract(text)
    checks: list[CitationCheck] = []
    blocking: list[str] = []
    warnings: list[str] = []

    # One CourtListener call for the whole draft.
    index: dict[str, LookupResult] = {}
    has_cases = any(c.kind == "case" for c in extracted)
    if has_cases:
        if cl is None:
            blocking.append(
                "CourtListener is not configured (set COURTLISTENER_API_TOKEN); case citations cannot be verified."
            )
        else:
            try:
                index = _lookup_index(cl.citation_lookup(text))
            except CourtListenerError as e:
                blocking.append(f"CourtListener lookup failed: {e}")

    full_case_volreps = {":".join(c.key.split(":")[1:3]) for c in extracted if c.kind == "case"}

    for c in extracted:
        if c.kind == "short_form":
            if c.antecedent and c.antecedent not in full_case_volreps:
                checks.append(
                    CitationCheck(
                        text=c.text,
                        kind="short_form",
                        status="ungrounded",
                        detail="Short-form cite has no matching full citation in the draft.",
                    )
                )
                blocking.append(f"Short cite '{c.text}' has no full citation earlier in the draft.")
            else:
                checks.append(
                    CitationCheck(
                        text=c.text, kind="short_form", status="skipped", detail="Resolves to an earlier full citation."
                    )
                )
            continue

        if c.kind != "case":
            if c.key in statute_ok:
                checks.append(
                    CitationCheck(
                        text=c.text,
                        kind=c.kind,
                        status="verified",
                        detail="Present in the local primary-source corpus or memo.",
                    )
                )
            else:
                checks.append(
                    CitationCheck(
                        text=c.text,
                        kind=c.kind,
                        status="ungrounded",
                        detail="Not in data/corpus/ (or still a placeholder). Add the official text.",
                    )
                )
                msg = f"'{c.text}' is not backed by a verified corpus file."
                (blocking if require_grounding else warnings).append(msg)
            continue

        # ---- case citations
        aid = grounded.get(c.key)
        if aid is None and require_grounding:
            checks.append(
                CitationCheck(
                    text=c.text,
                    kind="case",
                    status="ungrounded",
                    detail="Cited in the draft but never retrieved by research (Agent 1).",
                )
            )
            blocking.append(f"'{c.text}' was not retrieved by research; the drafter may have produced it from memory.")
            continue

        r = index.get(c.key)
        if r is None:
            # Either the whole lookup was unavailable (already blocking) or
            # CourtListener parsed the text differently and skipped this cite.
            lookup_ran = cl is not None and not any("CourtListener lookup failed" in b for b in blocking)
            checks.append(
                CitationCheck(
                    text=c.text,
                    kind="case",
                    status="error",
                    matched_authority_id=aid,
                    detail="CourtListener did not return this citation."
                    if lookup_ran
                    else "Not checked: CourtListener unavailable.",
                )
            )
            if lookup_ran:
                blocking.append(f"'{c.text}' was not returned by CourtListener citation lookup.")
            continue

        if r.status == 404:
            checks.append(
                CitationCheck(
                    text=c.text,
                    kind="case",
                    status="not_found",
                    matched_authority_id=aid,
                    detail="Valid format but not in CourtListener: probable hallucination.",
                )
            )
            blocking.append(f"'{c.text}' not found in CourtListener.")
            continue

        if r.status == 300:
            match = next((cl_ for cl_ in r.clusters if cites.names_match(c, cl_.get("case_name", ""))), None)
            if match is None:
                checks.append(
                    CitationCheck(
                        text=c.text,
                        kind="case",
                        status="ambiguous",
                        matched_authority_id=aid,
                        detail="Citation resolves to several cases; none matches the party names.",
                    )
                )
                blocking.append(f"'{c.text}' is ambiguous in CourtListener.")
                continue
            r = LookupResult(r.citation, r.normalized, 200, "", [match])

        if r.status != 200 or not r.clusters:
            checks.append(
                CitationCheck(
                    text=c.text,
                    kind="case",
                    status="error",
                    matched_authority_id=aid,
                    detail=f"CourtListener status {r.status}: {r.error}",
                )
            )
            blocking.append(f"'{c.text}': CourtListener status {r.status}.")
            continue

        cluster = r.clusters[0]
        url = f"https://www.courtlistener.com{cluster['absolute_url']}" if cluster.get("absolute_url") else None
        case_name = cluster.get("case_name", "")
        if not cites.names_match(c, case_name):
            checks.append(
                CitationCheck(
                    text=c.text,
                    kind="case",
                    status="name_mismatch",
                    matched_authority_id=aid,
                    url=url,
                    detail=f"Reporter cite belongs to '{case_name}'.",
                )
            )
            blocking.append(f"'{c.text}' is cited as {c.plaintiff} v. {c.defendant} but is {case_name}.")
            continue

        checks.append(
            CitationCheck(
                text=c.text, kind="case", status="verified", matched_authority_id=aid, url=url, detail=case_name
            )
        )

    # Unpublished authorities without reporter cites can't be machine-checked.
    for a in authorities.values():
        if a.kind == "case" and "(no reporter citation)" in a.citation and a.title and a.title in text:
            warnings.append(f"'{a.title}' has no reporter citation; check it by hand at {a.url}.")

    needed = FACT_NEEDED.findall(text)
    if needed:
        warnings.append(f"{len(needed)} [FACT NEEDED] placeholder(s) remain in the draft.")
    warnings.append(
        "Existence checks do not prove an authority is still good law. Run a citator "
        "(KeyCite/Shepard's) on every case before filing."
    )

    return VerificationReport(passed=not blocking, checks=checks, blocking=blocking, warnings=warnings)
