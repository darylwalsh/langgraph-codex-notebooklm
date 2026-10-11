"""Local citation extraction (no network).

Cases come from eyecite. Statutes, rules, and MDOC policy directives are matched
with explicit patterns, because eyecite's statute coverage is thin for the
sources this project depends on (MCL, Mich. Admin. Code, E.D. Mich. LR, MDOC PDs).

Every citation gets a normalized `key` so the verifier can compare a cite in the
draft against the cites Agent 1 actually retrieved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from eyecite import get_citations
from eyecite.models import (
    FullCaseCitation,
    FullLawCitation,
    IdCitation,
    ShortCaseCitation,
    SupraCitation,
)

# eyecite logs benign "overlap" notices when a short cite follows its full cite.
logging.getLogger("eyecite").setLevel(logging.ERROR)

Kind = Literal["case", "statute", "regulation", "court_rule", "policy", "short_form"]


@dataclass(frozen=True)
class ExtractedCitation:
    text: str
    kind: Kind
    key: str  # normalized identity used for grounding checks
    plaintiff: str | None = None
    defendant: str | None = None
    start: int = -1
    # For short forms like "578 U.S. at 643": the volume/reporter they point back to.
    antecedent: str | None = None


# ------------------------------------------------------------ non-case patterns
# Each pattern: (kind, compiled regex, key builder from match)

_SECTION_BASE = re.compile(r"^[0-9A-Za-z.\-]+?(?=\(|$)")


def _base(section: str) -> str:
    """'1997e(a)(1)' -> '1997e'; '600.5805' -> '600.5805'."""
    m = _SECTION_BASE.match(section.strip())
    return (m.group(0) if m else section).rstrip(".").lower()


_PATTERNS: list[tuple[Kind, re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (
        "statute",
        re.compile(r"\b(\d{1,2})\s+U\.\s?S\.\s?C\.(?:A\.)?\s*§{1,2}\s*([0-9A-Za-z.\-]+(?:\([0-9A-Za-z]+\))*)"),
        lambda m: f"usc:{m.group(1)}:{_base(m.group(2))}",
    ),
    (
        "statute",
        re.compile(r"\b(?:MCL|M\.\s?C\.\s?L\.|Mich\.\s*Comp\.\s*Laws(?:\s*Ann\.)?)\s*§*\s*(\d+\.\d+[a-z]*)"),
        lambda m: f"mcl:{m.group(1).lower()}",
    ),
    (
        "regulation",
        re.compile(r"Mich\.\s*Admin\.\s*Code\s*R\.?\s*(\d+\.\d+)"),
        lambda m: f"micode:{m.group(1)}",
    ),
    (
        "court_rule",
        re.compile(r"Fed\.\s*R\.\s*Civ\.\s*P\.\s*(\d+)((?:\([0-9A-Za-z]+\))*)"),
        lambda m: f"frcp:{m.group(1)}",
    ),
    (
        "court_rule",
        re.compile(r"E\.\s?D\.\s*Mich\.\s*(?:LR|L\.\s?R\.)\s*(\d+\.\d+)"),
        lambda m: f"edmichlr:{m.group(1)}",
    ),
    (
        "policy",
        re.compile(r"(?:MDOC\s+)?(?:Policy\s+Directive|PD)\s*(\d{2}\.\d{2}\.\d{3})"),
        lambda m: f"mdocpd:{m.group(1)}",
    ),
]


def _case_key(c: FullCaseCitation | ShortCaseCitation) -> str:
    g = c.groups
    reporter = (c.corrected_reporter() if hasattr(c, "corrected_reporter") else g.get("reporter")) or ""
    reporter = re.sub(r"\s+", "", reporter).lower()
    return f"case:{g.get('volume')}:{reporter}:{g.get('page')}"


def extract(text: str) -> list[ExtractedCitation]:
    """All citations in `text`, in document order, de-duplicated by key."""
    found: list[ExtractedCitation] = []

    for c in get_citations(text):
        if isinstance(c, FullCaseCitation):
            md = c.metadata
            found.append(
                ExtractedCitation(
                    text=c.corrected_citation(),
                    kind="case",
                    key=_case_key(c),
                    plaintiff=getattr(md, "plaintiff", None),
                    defendant=getattr(md, "defendant", None),
                    start=c.span()[0],
                )
            )
        elif isinstance(c, ShortCaseCitation):
            full = _case_key(c)  # case:vol:reporter:page-ish
            vol_rep = ":".join(full.split(":")[1:3])
            found.append(
                ExtractedCitation(
                    text=c.matched_text(),
                    kind="short_form",
                    key=f"short:{c.span()[0]}",
                    start=c.span()[0],
                    antecedent=vol_rep,
                )
            )
        elif isinstance(c, (IdCitation, SupraCitation)):
            found.append(
                ExtractedCitation(
                    text=c.matched_text(), kind="short_form", key=f"short:{c.span()[0]}", start=c.span()[0]
                )
            )
        elif isinstance(c, FullLawCitation):
            # Picked up below by the explicit patterns; eyecite keys differ.
            continue

    for kind, rx, keyfn in _PATTERNS:
        for m in rx.finditer(text):
            found.append(ExtractedCitation(text=m.group(0), kind=kind, key=keyfn(m), start=m.start()))

    found.sort(key=lambda x: x.start)
    seen: set[str] = set()
    unique: list[ExtractedCitation] = []
    for f in found:
        if f.key in seen:
            continue
        seen.add(f.key)
        unique.append(f)
    return unique


def keys(text: str) -> set[str]:
    return {c.key for c in extract(text) if c.kind != "short_form"}


def _norm_name(s: str | None) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def names_match(extracted: ExtractedCitation, case_name: str) -> bool:
    """Loose check that the cited party names appear in the resolved case name."""
    target = _norm_name(case_name)
    parties = [p for p in (extracted.plaintiff, extracted.defendant) if p]
    if not parties or not target:
        return True  # nothing to compare; don't fail on absence
    # Compare on the first distinctive word of each party name.
    hits = 0
    for p in parties:
        first = next((w for w in re.split(r"\W+", p) if len(w) > 2), p)
        if _norm_name(first) and _norm_name(first) in target:
            hits += 1
    return hits >= 1
