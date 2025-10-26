"""
Domain-specific triple rules for process_csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


Triple = Dict[str, object]

@dataclass
class TripleRules:
    """
    Simple include/exclude filtering for predicates plus hook placeholders.

    Attributes:
        include_relations: If provided, keep only triples whose normalized
            predicate is in this whitelist.
        exclude_relations: If provided, drop triples whose normalized predicate
            is in this blacklist.
    """

    include_relations: Optional[Sequence[str]] = None
    exclude_relations: Optional[Sequence[str]] = None
    lowercase_matching: bool = True
    strip_matching: bool = True
    _include_norm: set[str] = field(init=False, default_factory=set)
    _exclude_norm: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        def _norm(items: Optional[Sequence[str]]) -> set[str]:
            if not items:
                return set()
            return {self._normalize_rel(x) for x in items if x}

        self._include_norm = _norm(self.include_relations)
        self._exclude_norm = _norm(self.exclude_relations)

    def apply(self, triples: Iterable[Triple]) -> List[Triple]:
        """
        Apply all configured rules to the incoming triples.

        Returns a *new list*; the caller remains responsible for transforming
        the triples into normalized (s, p, o) tuples later on.
        """
        filtered: List[Triple] = []
        for triple in triples:
            if not self._passes_predicate_filters(triple):
                continue
            triple = self._transform_triple(triple)
            if triple is None:
                continue
            filtered.append(triple)
        return filtered

    # ------------------------------------------------------------------ #
    # Rule helpers
    # ------------------------------------------------------------------ #
    def _normalize_rel(self, rel: str) -> str:
        if not isinstance(rel, str):
            rel = str(rel)
        if self.strip_matching:
            rel = rel.strip()
        if self.lowercase_matching:
            rel = rel.lower()
        return rel

    def _extract_rel(self, triple: Triple) -> Optional[str]:
        # Accept common shapes from Plumber
        for key in ("predicate", "rel", "relation", "p"):
            if key in triple:
                value = triple[key]
                if isinstance(value, dict):
                    value = value.get("label") or value.get("text") or value.get("value")
                if value is not None:
                    return str(value)
        return None

    def _passes_predicate_filters(self, triple: Triple) -> bool:
        rel = self._extract_rel(triple)
        if rel is None:
            return False
        rel_norm = self._normalize_rel(rel)
        if self._include_norm and rel_norm not in self._include_norm:
            return False
        if self._exclude_norm and rel_norm in self._exclude_norm:
            return False
        return True

    def _transform_triple(self, triple: Triple) -> Optional[Triple]:
        """
        Hook for downstream customization.
        Return None to drop the triple after inspection/modification.
        """
        return triple


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #
def build_rules(
    include_relations: Optional[Sequence[str]] = None,
    exclude_relations: Optional[Sequence[str]] = None,
    **kwargs,
) -> TripleRules:
    """
    Factory used by process_csv to instantiate the rule set.

    kwargs are forwarded to TripleRules for future flexibility.
    """
    return TripleRules(include_relations=include_relations, exclude_relations=exclude_relations, **kwargs)


def apply_rules(rules: Optional[TripleRules], triples: Iterable[Triple]) -> List[Triple]:
    """
    Convenience wrapper: if no rules are provided, return the triples unchanged.
    """
    if rules is None:
        return list(triples)
    return rules.apply(triples)

