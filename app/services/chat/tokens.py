import tiktoken
from langchain.messages import AIMessage, HumanMessage, SystemMessage, UsageMetadata

from app.services.chat.types import UsageAccumulator


def empty_usage() -> UsageAccumulator:
    return {
        "router_input_tokens": 0,
        "router_output_tokens": 0,
        "answer_input_tokens": 0,
        "answer_output_tokens": 0,
    }


def parse_usage(usage: UsageMetadata | None) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    return input_tokens, output_tokens, total_tokens


def estimate_token_count(content: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name="gpt-5")
    return len(encoding.encode(content))


def message_text(messages: list[SystemMessage | HumanMessage | AIMessage]) -> str:
    return "\n".join(
        str(message.content)
        for message in messages
        if isinstance(message.content, str)
    )


def count_tokens(model, content: str) -> int:
    try:
        return int(model.get_num_tokens(content))
    except Exception:
        return estimate_token_count(content)
