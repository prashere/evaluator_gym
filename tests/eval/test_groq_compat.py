from evaluator_gym.eval.groq_compat import GPT_OSS_MAX_TOKENS, groq_http_chat_body, groq_sampling_args


def test_groq_oss_defaults_max_tokens():
    args = groq_sampling_args({"temperature": 0.7})
    assert args["max_tokens"] == 1000
    assert "reasoning_effort" not in args


def test_groq_qwen_disables_reasoning_and_requests_json():
    args = groq_sampling_args({"temperature": 0.7}, api_model_id="qwen/qwen3.6-27b")
    assert args["reasoning_effort"] == "none"
    assert args["reasoning_format"] == "hidden"
    assert args["response_format"] == {"type": "json_object"}
    assert args["max_tokens"] == 1000


def test_groq_gpt_oss_hides_reasoning_via_extra_body():
    args = groq_sampling_args({"temperature": 0.7}, api_model_id="openai/gpt-oss-20b")
    assert args["max_tokens"] == GPT_OSS_MAX_TOKENS
    assert args["extra_body"]["reasoning_format"] == "hidden"
    assert "reasoning_effort" not in args
    assert "response_format" not in args


def test_groq_http_chat_body_flattens_extra_body_for_rest():
    body = groq_http_chat_body({"max_tokens": 8}, api_model_id="openai/gpt-oss-120b")
    assert body["max_tokens"] == 8
    assert body["reasoning_format"] == "hidden"
    assert "extra_body" not in body
