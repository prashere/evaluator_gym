from evaluator_gym.reference import tags


def test_tag_severity_map_matches_section_13():
    assert len(tags.ALL_TAGS) == 15
    assert len(tags.TAG_SEVERITY) == 15
    assert tags.BLOCKING_TAGS <= tags.ALL_TAGS
    assert len(tags.BLOCKING_TAGS) == 6
    escalate = {t for t, s in tags.TAG_SEVERITY.items() if s == "ESCALATE"}
    assert escalate == {
        tags.PO_CONFLICT_UNRESOLVED,
        tags.VENDOR_SUSPENDED,
        tags.OUTSIDE_DELEGATION,
    }
