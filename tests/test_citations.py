from s1983_agents.citations import extract, keys, names_match

TEXT = """Plaintiff need not plead exhaustion. Jones v. Bock, 549 U.S. 199, 216 (2007).
Remedies must be available. Ross v. Blake, 578 U.S. 632, 642 (2016). Id. at 643. Ross, 578 U.S. at 644.
42 U.S.C. § 1997e(a); 42 U.S.C. § 1983; 28 U.S.C. §§ 1915A; MCL 600.5805; Mich. Admin. Code R 791.3310;
Fed. R. Civ. P. 12(b)(6); E.D. Mich. LR 5.1; MDOC Policy Directive 03.02.130.
"""


def test_extracts_every_kind():
    found = {c.key: c for c in extract(TEXT)}
    assert {"case:549:u.s.:199", "case:578:u.s.:632"} <= found.keys()
    assert {
        "usc:42:1997e",
        "usc:42:1983",
        "usc:28:1915a",
        "mcl:600.5805",
        "micode:791.3310",
        "frcp:12",
        "edmichlr:5.1",
        "mdocpd:03.02.130",
    } <= found.keys()
    shorts = [c for c in extract(TEXT) if c.kind == "short_form"]
    assert any(c.antecedent == "578:u.s." for c in shorts)


def test_subsections_collapse_to_section():
    assert keys("42 U.S.C. § 1997e(a)") == keys("42 U.S.C. § 1997e(e)") == {"usc:42:1997e"}


def test_name_match():
    ross = next(c for c in extract(TEXT) if c.key == "case:578:u.s.:632")
    assert names_match(ross, "Ross v. Blake")
    assert not names_match(ross, "Miranda v. Arizona")
