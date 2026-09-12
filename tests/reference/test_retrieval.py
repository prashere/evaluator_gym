import pytest

from evaluator_gym.retrieval.ground_truth import compute_retrieval_ground_truth


def test_compute_retrieval_vendor_status():
    payload = {
        "case": {"vendor_record": {"vendor_id": "V-100", "status": "ACTIVE"}},
        "retrieval_spec": {"fields": {"vendor_status": ["vendor_record", "status"]}},
    }
    assert compute_retrieval_ground_truth(payload) == {"vendor_status": "ACTIVE"}


def test_compute_retrieval_with_const_field():
    payload = {
        "case": {"goods_receipt": {"received_quantities": {"ITEM-A": "10"}}},
        "retrieval_spec": {
            "fields": {
                "item_id": {"const": "ITEM-A"},
                "received_quantity": ["goods_receipt", "received_quantities", "ITEM-A"],
            }
        },
    }
    assert compute_retrieval_ground_truth(payload) == {
        "item_id": "ITEM-A",
        "received_quantity": "10",
    }


def test_compute_retrieval_missing_path_raises():
    payload = {
        "case": {"vendor_record": None},
        "retrieval_spec": {"fields": {"vendor_status": ["vendor_record", "status"]}},
    }
    with pytest.raises(KeyError):
        compute_retrieval_ground_truth(payload)


def test_compute_retrieval_line_item_lookup():
    payload = {
        "case": {
            "purchase_order": {
                "lines": [
                    {"item_id": "ITEM-A", "unit_price": "100.00"},
                    {"item_id": "ITEM-NOISE", "unit_price": "50.00"},
                ]
            }
        },
        "retrieval_spec": {
            "fields": {
                "item_id": {"const": "ITEM-A"},
                "unit_price": {
                    "line_item": {
                        "list": ["purchase_order", "lines"],
                        "match_field": "item_id",
                        "match": "ITEM-A",
                        "read": "unit_price",
                    }
                },
            }
        },
    }
    assert compute_retrieval_ground_truth(payload) == {
        "item_id": "ITEM-A",
        "unit_price": "100.00",
    }


def test_compute_retrieval_amendment_lookup():
    payload = {
        "case": {
            "purchase_order": {
                "amendments": [
                    {"amendment_id": "AMD-1", "effective_date": "2026-01-15"},
                    {"amendment_id": "AMD-2", "effective_date": "2026-02-15"},
                ]
            }
        },
        "retrieval_spec": {
            "fields": {
                "amendment_id": {"const": "AMD-2"},
                "effective_date": {
                    "amendment": {
                        "list": ["purchase_order", "amendments"],
                        "id_field": "amendment_id",
                        "id": "AMD-2",
                        "read": "effective_date",
                    }
                },
            }
        },
    }
    assert compute_retrieval_ground_truth(payload) == {
        "amendment_id": "AMD-2",
        "effective_date": "2026-02-15",
    }
