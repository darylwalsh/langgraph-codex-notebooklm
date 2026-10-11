"""Thin CourtListener v4 client: opinion search + citation lookup.

Docs: https://www.courtlistener.com/help/api/rest/
The transport is injectable so tests can run fully offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .state import Authority

# citation-lookup limits per request (documented by CourtListener)
MAX_LOOKUP_CHARS = 64_000


@dataclass
class LookupResult:
    """One citation found by /citation-lookup/."""

    citation: str
    normalized: list[str]
    status: int  # 200 found, 404 not found, 300 ambiguous, 400 bad reporter, 429 over limit
    error: str
    clusters: list[dict[str, Any]] = field(default_factory=list)


class CourtListenerError(RuntimeError):
    pass


class CourtListener:
    def __init__(
        self,
        token: str | None,
        base_url: str = "https://www.courtlistener.com/api/rest/v4",
        *,
        transport: httpx.BaseTransport | None = None,
        min_interval_s: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        headers = {"User-Agent": "s1983-agents/0.1"}
        if token:
            headers["Authorization"] = f"Token {token}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )
        self._min_interval = min_interval_s
        self._last = 0.0

    # ------------------------------------------------------------ plumbing

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        for attempt in range(4):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self._http.request(method, path, **kw)
            # A top-level 429 is throttling; retry with backoff.
            if resp.status_code == 429 and attempt < 3:
                time.sleep(float(resp.headers.get("Retry-After", 2 ** (attempt + 1))))
                continue
            if resp.status_code >= 400:
                raise CourtListenerError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp
        raise CourtListenerError(f"{method} {path}: retries exhausted")

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------ search

    def search_opinions(self, query: str, courts: tuple[str, ...], limit: int = 8) -> list[Authority]:
        """Full-text opinion search restricted to the given court ids."""
        resp = self._request(
            "GET",
            "/search/",
            params={
                "q": query,
                "type": "o",
                "court": " ".join(courts),
                "order_by": "score desc",
            },
        )
        out: list[Authority] = []
        for r in resp.json().get("results", [])[:limit]:
            cites = r.get("citation") or []
            if not cites:
                # No reporter citation (often unpublished). Keep it, but flag it:
                # the verifier will not be able to resolve it by citation.
                cites = [f"{r.get('docketNumber', '')} (no reporter citation)".strip()]
            snippet = " ".join((op.get("snippet") or "") for op in (r.get("opinions") or [])).strip()
            out.append(
                Authority(
                    id=f"cl:{r.get('cluster_id')}",
                    kind="case",
                    title=r.get("caseName") or "",
                    citation=cites[0],
                    court=r.get("court_id"),
                    date=r.get("dateFiled"),
                    url=f"https://www.courtlistener.com{r['absolute_url']}" if r.get("absolute_url") else None,
                    excerpt=snippet[:1200],
                )
            )
        return out

    # ------------------------------------------------------------ lookup

    def citation_lookup(self, text: str) -> list[LookupResult]:
        """Extract and resolve every case citation in `text`."""
        if len(text) > MAX_LOOKUP_CHARS:
            results: list[LookupResult] = []
            for i in range(0, len(text), MAX_LOOKUP_CHARS):
                results.extend(self.citation_lookup(text[i : i + MAX_LOOKUP_CHARS]))
            return results
        resp = self._request("POST", "/citation-lookup/", data={"text": text})
        return [
            LookupResult(
                citation=item.get("citation", ""),
                normalized=item.get("normalized_citations") or [],
                status=int(item.get("status", 0)),
                error=item.get("error_message") or "",
                clusters=item.get("clusters") or [],
            )
            for item in resp.json()
        ]

    def resolve_citations(self, citations: list[str]) -> list[Authority]:
        """Turn known-good seed citations into Authority records."""
        out: list[Authority] = []
        for res in self.citation_lookup("\n".join(citations)):
            if res.status != 200 or not res.clusters:
                continue
            c = res.clusters[0]
            out.append(
                Authority(
                    id=f"cl:{c.get('id')}",
                    kind="case",
                    title=c.get("case_name") or "",
                    citation=(res.normalized or [res.citation])[0],
                    court=None,
                    date=c.get("date_filed"),
                    url=f"https://www.courtlistener.com{c['absolute_url']}" if c.get("absolute_url") else None,
                )
            )
        return out
