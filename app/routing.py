"""Lightweight query routing helpers shared by chat and evaluation."""

from __future__ import annotations

from typing import Literal

RerankMode = Literal["auto", "off", "always"]

CLAUSE_TYPE_TERMS: dict[str, tuple[str, ...]] = {
    "Anti-Assignment": (
        "anti-assignment",
        "assign",
        "assignment",
        "transfer the agreement",
        "transfer this agreement",
        "transferred to another party",
    ),
    "Cap On Liability": (
        "cap on liability",
        "liability cap",
        "limit liability",
        "limitation of liability",
        "limited liability",
        "damages",
        "consequential loss",
        "consequential damages",
        "indirect damages",
        "lost profits",
        "categories of loss",
        "category of loss",
        "excluded losses",
        "losses excluded",
        "anticipated savings",
        "prospective profits",
        "special damages",
        "punitive damages",
        "responsibility for losses",
    ),
    "License Grant": (
        "license",
        "licence",
        "licensed materials",
        "usage rights",
        "right to use",
        "rights to use",
        "use intellectual property",
        "use the intellectual property",
    ),
    "Audit Rights": (
        "audit",
        "inspect records",
        "inspect books",
        "review records",
        "review compliance records",
        "books and records",
    ),
    "Termination For Convenience": (
        "terminate",
        "termination",
        "for convenience",
        "without cause",
        "end the agreement",
        "ending the agreement",
        "cancel the agreement",
        "walk away",
    ),
}


def infer_clause_type(query: str) -> str | None:
    """Infer a supported starter clause type from legal concept phrases."""

    normalized_query = " ".join(query.lower().replace("-", " ").split())
    matches: list[tuple[int, str]] = []
    for clause_type, terms in CLAUSE_TYPE_TERMS.items():
        score = sum(
            len(term.split())
            for term in terms
            if term.replace("-", " ") in normalized_query
        )
        if score:
            matches.append((score, clause_type))

    # Legal paraphrases often reverse word order, for example
    # "intellectual property use" instead of "use intellectual property".
    if "intellectual property" in normalized_query and any(
        term in normalized_query
        for term in ("use", "right", "rights", "grant", "granted", "license", "licence")
    ):
        matches.append((4, "License Grant"))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def choose_reranking(
    *,
    mode: RerankMode,
    query: str,
    resolved_clause_type: str,
) -> tuple[bool, str]:
    """Choose reranking only where the measured starter evaluation benefits."""

    if mode == "off":
        return False, "disabled by request"
    if mode == "always":
        return True, "enabled by request"

    normalized = " ".join(query.lower().replace("-", " ").split())
    has_ip_phrase = "intellectual property" in normalized
    has_usage_language = any(
        term in normalized
        for term in ("use", "usage", "right", "rights", "grant", "granted", "provision")
    )
    has_explicit_license = "license" in normalized or "licence" in normalized
    ip_paraphrase = (
        resolved_clause_type == "License Grant"
        and has_ip_phrase
        and has_usage_language
        and not has_explicit_license
    )
    detail_terms = (
        "affiliate",
        "anniversary",
        "cost",
        "consequence",
        "consequences",
        "days",
        "duration",
        "effective",
        "exception",
        "exceptions",
        "how much",
        "how often",
        "operation of law",
        "percent",
        "perpetual",
        "prior notice",
        "subsidiary",
        "territory",
        "threshold",
        "void",
        "voidable",
        "what happens",
        "wholly owned",
        "written notice",
    )
    nuanced_detail = any(term in normalized for term in detail_terms)
    if ip_paraphrase:
        return True, "adaptive intellectual-property paraphrase"
    if nuanced_detail:
        return True, "adaptive clause-detail question"
    return False, "adaptive vector search"
