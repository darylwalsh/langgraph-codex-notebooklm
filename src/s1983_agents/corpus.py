"""Local primary-source corpus: statutes, rules, MDOC policy directives.

Each source is a Markdown file under data/corpus/ with YAML front matter:

    ---
    id: usc-42-1997e
    kind: statute            # statute | regulation | court_rule | policy
    title: 42 U.S.C. § 1997e — Suits by prisoners
    citations: ["42 U.S.C. § 1997e"]
    url: https://uscode.house.gov/...
    as_of: 2026-10-01        # date you copied the official text
    status: verified         # 'placeholder' files are skipped at load time
    ---
    <official text, pasted verbatim from the url>

Retrieval here is a simple keyword ranker so the scaffold runs with no extra
infrastructure. Swap `Corpus.search` for a vector store (pgvector, Azure AI
Search, etc.) without changing the rest of the graph.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import citations as cites
from .state import Authority

log = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9.§]+")
_STOP = frozenset(
    "the a an of to and or in on for by with is are be as that this any shall may".split()  # noqa: SIM905
)


@dataclass
class CorpusDoc:
    authority: Authority
    body: str
    citation_keys: set[str]

    @property
    def paragraphs(self) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", self.body) if p.strip()]


def _tokens(s: str) -> list[str]:
    return [t for t in _WORD.findall(s.lower()) if t not in _STOP]


def _parse(path: Path) -> CorpusDoc | None:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        log.warning("corpus: %s has no front matter; skipped", path.name)
        return None
    meta = yaml.safe_load(m.group(1)) or {}
    if meta.get("status") == "placeholder":
        log.warning("corpus: %s is a placeholder; paste the official text to enable it", path.name)
        return None
    citation_list = meta.get("citations") or [meta.get("title", "")]
    auth = Authority(
        id=f"corpus:{meta['id']}",
        kind=meta.get("kind", "statute"),
        title=meta.get("title", meta["id"]),
        citation=citation_list[0],
        url=meta.get("url"),
        as_of=str(meta.get("as_of")) if meta.get("as_of") else None,
    )
    keys = set().union(*(cites.keys(c) for c in citation_list)) if citation_list else set()
    return CorpusDoc(authority=auth, body=m.group(2), citation_keys=keys)


class Corpus:
    def __init__(self, docs: list[CorpusDoc]) -> None:
        self.docs = docs

    @classmethod
    def load(cls, directory: Path) -> Corpus:
        if not directory.exists():
            log.warning("corpus dir %s does not exist", directory)
            return cls([])
        docs = [d for p in sorted(directory.glob("*.md")) if (d := _parse(p))]
        return cls(docs)

    @property
    def citation_keys(self) -> set[str]:
        return set().union(*(d.citation_keys for d in self.docs)) if self.docs else set()

    def search(self, query: str, k: int = 6, passages_per_doc: int = 2) -> list[Authority]:
        """Rank docs by keyword overlap; attach the best-matching passages as excerpt."""
        q = Counter(_tokens(query))
        if not q:
            return []
        scored: list[tuple[float, Authority]] = []
        for d in self.docs:
            paras = d.paragraphs
            ranked = sorted(
                ((sum(q[t] for t in _tokens(p)) / (1 + len(p) / 800), p) for p in paras),
                reverse=True,
            )
            score = sum(s for s, _ in ranked[:passages_per_doc])
            if score <= 0:
                continue
            excerpt = "\n\n".join(p for _, p in ranked[:passages_per_doc])
            scored.append((score, d.authority.model_copy(update={"excerpt": excerpt[:2000]})))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:k]]
