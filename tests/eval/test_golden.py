from evaluator_gym.eval.golden import golden_task_ids, validate_golden_suite


def test_golden_suite_valid():
    errors = validate_golden_suite()
    assert errors == []


def test_golden_task_count():
    ids = golden_task_ids()
    assert len(ids) == 10
    assert len(set(ids)) == 10
