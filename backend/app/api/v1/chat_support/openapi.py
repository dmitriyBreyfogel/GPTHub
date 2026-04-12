GENERATION_OPTION_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
    "response_format",
    "tools",
    "tool_choice",
}

CHAT_COMPLETIONS_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["messages"],
                    "additionalProperties": True,
                    "properties": {
                        "model": {"type": "string", "example": "gpt-4o-mini"},
                        "workspace_id": {"type": "string", "format": "uuid"},
                        "task_type": {"type": "string"},
                        "stream": {"type": "boolean", "default": False},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "integer"},
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["role", "content"],
                                "additionalProperties": True,
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["system", "user", "assistant", "tool"],
                                    },
                                    "content": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                            },
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
                "example": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain FastAPI router in simple terms",
                        }
                    ],
                    "stream": False,
                },
            }
        },
    }
}
