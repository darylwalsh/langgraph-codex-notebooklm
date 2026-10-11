"""Offline fakes: a scripted LLM and a mock CourtListener HTTP backend."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from eyecite import get_citations
from eyecite.models import FullCaseCitation

from s1983_agents.config import Settings
from s1983_agents.corpus import Corpus
from s1983_agents.courtlistener import CourtListener

# "Real" cases the mock CourtListener knows about: cite -> (cluster id, case name)
KNOWN_CASES = {
    "578 U.S. 632": (3195917, "Ross v. Blake"),
    "549 U.S. 199": (145662, "Jones v. Bock"),
    "556 U.S. 662": (145875, "Ashcroft v. Iqbal"),
    "550 U.S. 544": (145730, "Bell Atlantic Corp. v. Twombly"),
}


def _lookup(text: str) -> list[dict]:
    out = []
    for c in get_citations(text):
        if not isinstance(c, FullCaseCitation):
            continue
        cite = c.corrected_citation()
        hit = KNOWN_CASES.get(cite)
        out.append(
            {
                "citation": c.matched_text(),
                "normalized_citations": [cite],
                "start_index": c.span()[0],
                "end_index": c.span()[1],
                "status": 200 if hit else 404,
                "error_message": "" if hit else "Citation not found",
                "clusters": [
                    {
                        "id": hit[0],
                        "case_name": hit[1],
                        "absolute_url": f"/opinion/{hit[0]}/x/",
                        "date_filed": "2016-06-06",
                    }
                ]
                if hit
                else [],
            }
        )
    return out


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/citation-lookup/"):
        text = parse_qs(request.content.decode())["text"][0]
        return httpx.Response(200, json=_lookup(text))
    if request.url.path.endswith("/search/"):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "cluster_id": 3195917,
                        "caseName": "Ross v. Blake",
                        "citation": ["578 U.S. 632"],
                        "court_id": "scotus",
                        "dateFiled": "2016-06-06",
                        "absolute_url": "/opinion/3195917/ross-v-blake/",
                        "opinions": [{"snippet": "test snippet about grievance availability"}],
                    }
                ]
            },
        )
    return httpx.Response(404, json={"detail": "not mocked"})


@pytest.fixture
def cl() -> CourtListener:
    return CourtListener("test-token", transport=httpx.MockTransport(_handler), min_interval_s=0)


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    (tmp_path / "usc-42-1983.md").write_text(
        "---\nid: usc-42-1983\nkind: statute\ntitle: 42 U.S.C. § 1983\n"
        'citations: ["42 U.S.C. § 1983"]\nas_of: 2026-10-01\nstatus: verified\n---\n'
        "Test fixture text about civil action for deprivation of rights under color of state law.\n",
        encoding="utf-8",
    )
    (tmp_path / "usc-42-1997e.md").write_text(
        "---\nid: usc-42-1997e\nkind: statute\ntitle: 42 U.S.C. § 1997e\n"
        'citations: ["42 U.S.C. § 1997e"]\nas_of: 2026-10-01\nstatus: verified\n---\n'
        "Test fixture text about prisoner exhaustion of administrative remedies.\n",
        encoding="utf-8",
    )
    (tmp_path / "placeholder.md").write_text(
        "---\nid: x\nkind: policy\ntitle: X\nstatus: placeholder\n---\nnothing\n", encoding="utf-8"
    )
    return Corpus.load(tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        max_judge_revisions=2,
        max_verify_retries=1,
        seed_citations=("Ross v. Blake, 578 U.S. 632 (2016)", "Jones v. Bock, 549 U.S. 199 (2007)"),
        corpus_dir=tmp_path,
        checkpoint_db=tmp_path / "cp.sqlite",
    )


class FakeLLM:
    """Scripted LLM. `script[role]` is a callable(schema, system, user, call_no) -> object."""

    def __init__(self, script: dict[str, Callable]) -> None:
        self.script = script
        self.calls: dict[str, list[str]] = defaultdict(list)

    def structured(self, role, schema, system, user):
        n = len([c for c in self.calls[role] if c == schema.__name__])
        self.calls[role].append(schema.__name__)
        self.last_user = getattr(self, "last_user", {})
        self.last_user[(role, schema.__name__)] = user
        result = self.script[role](schema, system, user, n)
        return result if isinstance(result, schema) else schema.model_validate(result)


def dump(obj) -> str:
    return json.dumps(obj, indent=2)
