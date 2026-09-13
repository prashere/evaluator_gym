"""Dashboard transcript extraction."""

from dashboard.extract import extract_messages


def test_extract_messages_keeps_full_user_prompt():
    long_body = "x" * 5000
    messages = [f"role='user' content='{long_body}'"]
    out = extract_messages(messages)
    assert out["user_excerpt"] == long_body
    assert not out["user_excerpt"].endswith("…")
