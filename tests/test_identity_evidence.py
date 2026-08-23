from reid.identity_evidence import *


def test_appearance_and_face_alone_cannot_name_employee():
    ev = [
        IdentityEvidence(2, "alice", EvidenceSource.APPEARANCE_CANDIDATE, 0.99),
        IdentityEvidence(2, "alice", EvidenceSource.FACE_CONFIRM, 0.9),
    ]
    assert resolve_path_identity(2, ev).person_id is None


def test_badge_anchor_can_be_confirmed_but_conflicts_are_unresolved():
    ev = [
        IdentityEvidence(2, "alice", EvidenceSource.BADGE, 0.9),
        IdentityEvidence(2, "alice", EvidenceSource.FACE_CONFIRM, 0.8),
    ]
    assert resolve_path_identity(2, ev).person_id == "alice"
    ev.append(IdentityEvidence(2, "bob", EvidenceSource.OPERATOR, 1.0, reviewed=True))
    assert resolve_path_identity(2, ev).status == "conflict"
