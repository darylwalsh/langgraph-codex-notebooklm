from s1983_agents.state import Authority, ResearchMemo
from s1983_agents.verifier import verify_draft


def memo_with(*auths: Authority) -> ResearchMemo:
    return ResearchMemo(
        claims_and_elements=[],
        exhaustion_and_availability=[],
        defenses_to_anticipate=[],
        authorities={a.id: a for a in auths},
    )


ROSS = Authority(id="cl:3195917", kind="case", title="Ross v. Blake", citation="578 U.S. 632")
JONES = Authority(id="cl:145662", kind="case", title="Jones v. Bock", citation="549 U.S. 199")


def statuses(report):
    return {c.text: c.status for c in report.checks}


def test_clean_draft_passes(cl, corpus):
    text = "Ross v. Blake, 578 U.S. 632 (2016); Ross, 578 U.S. at 643; 42 U.S.C. § 1997e(a); 42 U.S.C. § 1983."
    r = verify_draft(text, memo_with(ROSS), corpus.citation_keys, cl)
    assert r.passed, r.blocking
    assert statuses(r)["578 U.S. 632"] == "verified"


def test_fabricated_case_blocks(cl, corpus):
    # Grounded (pretend research returned it) but CourtListener has never heard of it.
    fake = Authority(id="cl:999", kind="case", title="Smith v. Warden", citation="999 F.4th 123")
    r = verify_draft("Smith v. Warden, 999 F.4th 123 (6th Cir. 2025).", memo_with(fake), corpus.citation_keys, cl)
    assert not r.passed
    assert statuses(r)["999 F.4th 123"] == "not_found"


def test_case_from_memory_is_ungrounded(cl, corpus):
    # Real case, but research never retrieved it: the drafter cited from memory.
    r = verify_draft("Jones v. Bock, 549 U.S. 199 (2007).", memo_with(ROSS), corpus.citation_keys, cl)
    assert not r.passed
    assert statuses(r)["549 U.S. 199"] == "ungrounded"


def test_wrong_case_name_blocks(cl, corpus):
    r = verify_draft("Miranda v. Arizona, 578 U.S. 632 (2016).", memo_with(ROSS), corpus.citation_keys, cl)
    assert statuses(r)["578 U.S. 632"] == "name_mismatch"
    assert not r.passed


def test_statute_not_in_corpus_blocks(cl, corpus):
    # MDOC PD file in conftest is a placeholder, so it is not loaded.
    r = verify_draft("See MDOC Policy Directive 03.02.130.", memo_with(), corpus.citation_keys, cl)
    assert not r.passed
    assert "MDOC Policy Directive 03.02.130" in statuses(r)


def test_orphan_short_cite_blocks(cl, corpus):
    r = verify_draft("Ross, 578 U.S. at 643.", memo_with(ROSS), corpus.citation_keys, cl)
    assert not r.passed


def test_no_courtlistener_blocks_case_cites(corpus):
    r = verify_draft("Ross v. Blake, 578 U.S. 632 (2016).", memo_with(ROSS), corpus.citation_keys, None)
    assert not r.passed
    assert any("not configured" in b for b in r.blocking)


def test_standalone_mode_skips_grounding(cl, corpus):
    r = verify_draft("Jones v. Bock, 549 U.S. 199 (2007).", None, corpus.citation_keys, cl, require_grounding=False)
    assert r.passed, r.blocking


def test_fact_needed_is_a_warning(cl, corpus):
    r = verify_draft("On [FACT NEEDED: date], Defendant ...", memo_with(), corpus.citation_keys, cl)
    assert r.passed
    assert any("FACT NEEDED" in w for w in r.warnings)
