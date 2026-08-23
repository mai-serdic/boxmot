"""Evidence-backed mapping from anonymous path identity to a named person."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceSource(str, Enum):
    BADGE = "badge"
    UWB = "uwb"
    TURNSTILE = "turnstile"
    OPERATOR = "operator"
    FACE_CONFIRM = "face_confirm"
    APPEARANCE_CANDIDATE = "appearance_candidate"


AUTHORITATIVE = {
    EvidenceSource.BADGE,
    EvidenceSource.UWB,
    EvidenceSource.TURNSTILE,
    EvidenceSource.OPERATOR,
}


@dataclass(frozen=True)
class IdentityEvidence:
    path_id: int
    person_id: str
    source: EvidenceSource
    confidence: float
    frame: int | None = None
    reviewed: bool = False


@dataclass(frozen=True)
class PathIdentity:
    path_id: int
    person_id: str | None
    confidence: float
    status: str
    evidence: tuple[IdentityEvidence, ...]


def resolve_path_identity(path_id, evidence):
    ev = tuple(e for e in evidence if e.path_id == path_id)
    hard = [e for e in ev if e.source in AUTHORITATIVE]
    people = {e.person_id for e in hard}
    if len(people) > 1:
        return PathIdentity(path_id, None, 0.0, "conflict", ev)
    if hard:
        best = max(hard, key=lambda e: (e.reviewed, e.confidence))
        confirms = [
            e
            for e in ev
            if e.source == EvidenceSource.FACE_CONFIRM and e.person_id == best.person_id
        ]
        confidence = min(1.0, best.confidence + (0.05 if confirms else 0.0))
        return PathIdentity(path_id, best.person_id, confidence, "anchored", ev)
    # Face and appearance can rank candidates for review, never name a person.
    return PathIdentity(path_id, None, 0.0, "unresolved", ev)


def event_attribution(path_id, evidence):
    ident = resolve_path_identity(path_id, evidence)
    return {
        "path_id": path_id,
        "person_id": ident.person_id,
        "identity_status": ident.status,
        "identity_confidence": ident.confidence,
        "requires_review": ident.person_id is None or ident.status != "anchored",
    }
