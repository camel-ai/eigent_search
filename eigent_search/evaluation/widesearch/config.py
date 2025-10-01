"""
adapted from https://github.com/ByteDance-Seed/WideSearch
"""

model_config = {
    "model_config_name": {
        "model_name": "MODEL_NAME",
        "base_url": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY",
    },
    "k2": {
        "model_name": "kimi-k2-250711",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 32768,
        },
    },
    "doubao-1.6": {
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "enabled"},
            "max_tokens": 65535,
        },
    },
    "deepseek-r1": {
        "model_name": "deepseek-r1",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    "doubao-1.6-non-thinking": {  # for eval
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "disabled"},
            "max_tokens": 65535,
        },
    },
    "claude37-sonnet-thinking": {
        "model_name": "gcp-claude37-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            "max_tokens": 10240,
        },
        "is_claude_thinking": True,
    },
    "claude4-sonnet-thinking": {
        "model_name": "claude4-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 32768}},
            "max_tokens": 64000,
        },
        "is_claude_thinking": True,
    },
    "o3-medium": {
        "model_name": "o3-2025-04-16",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
            "reasoning_effort": "medium",
        },
        "default_system_prompt": "Formatting re-enabled",
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro-preview-06-05",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    # "default_eval_config": {
    #     "model_name": "",
    #     "base_url": "",
    #     "api_key": "",
    #     "generate_kwargs": {
    #         "max_tokens": 10240,
    #     },
    #     "temperature": 0,
    # },
    "default_eval_config": {
        "model_name": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "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",
        "generate_kwargs": {
            "max_tokens": 10240,
        },
        "temperature": 0,
    },
    "gpt": {
        "model_name": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "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",
        "generate_kwargs": {
            "max_tokens": 10240,
        },
        "temperature": 0,
    },
}