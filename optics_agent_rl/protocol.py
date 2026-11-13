"""The existing no-auxiliary-tool Agent contract, in native Qwen wire format."""
from __future__ import annotations

import json
import re

from optics_agent.protocol import DRAW_PROMPT_DESCRIPTION, IMAGE_REVIEW_PROMPT, SYSTEM_PROMPT
from optics_agent.runtime_state import draw_runtime_state
from optics_experiments.protocols import parse_qwen_action


DRAW_SCHEMA = {"type": "function", "function": {
    "name": "draw",
    "description": "Generate a new image from a text prompt. Returns the generated image for visual inspection. Text-to-image only: no editing and no input images.",
    "parameters": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": DRAW_PROMPT_DESCRIPTION}}, "required": ["prompt"]},
}}


class PolicyProtocolError(ValueError):
    """A recoverable policy action violation, distinct from a service failure."""


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PolicyProtocolError("duplicate_json_key")
        value[key] = item
    return value


def validate_tool_serialization(parser: str, chat_template_kwargs=None):
    """Keep the sampled protocol aligned with the saved SFT chat template."""
    if parser not in {"hermes", "qwen3_xml"}:
        raise ValueError("RL tool_call_parser must be hermes or qwen3_xml")
    if parser == "qwen3_xml" and (chat_template_kwargs or {}).get("enable_thinking") is not False:
        raise ValueError("Agent v3 XML rollout requires data.apply_chat_template_kwargs.enable_thinking=false")
    return parser


def parse_assistant(text: str, *, terminated: bool, call_id: str, tool_call_parser: str = "hermes"):
    """Parse the sampled text without reconstructing its training token stream."""
    if tool_call_parser not in {"hermes", "qwen3_xml"}:
        raise ValueError("Unsupported RL tool_call_parser")
    if not terminated:
        raise PolicyProtocolError("incomplete_model_response")
    text = text.strip().removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()
    if re.search(r"<\|[^>]+\|>", text):
        raise PolicyProtocolError("unexpected_control_token")
    if tool_call_parser == "qwen3_xml" and re.search(r"</?think>", text):
        raise PolicyProtocolError("unexpected_thinking_markup")
    if "<tool_call>" not in text and "</tool_call>" not in text:
        if not text:
            raise PolicyProtocolError("empty_final_response")
        if tool_call_parser == "qwen3_xml" and re.search(r"<(?:function|parameter)=|</(?:function|parameter)>", text):
            raise PolicyProtocolError("malformed_tool_call")
        return {"role": "assistant", "content": text}, None
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise PolicyProtocolError("expected_one_draw_call")
    match = re.fullmatch(r"(.*?)<tool_call>\s*(.*?)\s*</tool_call>\s*", text, re.DOTALL)
    if match is None:
        raise PolicyProtocolError("malformed_tool_call")
    if tool_call_parser == "qwen3_xml":
        # Share the native evaluation parser, but enforce the RL single-action
        # envelope and reject embedded control tags before template projection.
        # Qwen's function= syntax is not XML: do not entity-decode image text.
        body = match.group(2)
        if not body.startswith("<function="):
            raise PolicyProtocolError("invalid_tool_xml")
        try:
            message = parse_qwen_action(text, call_id)
            arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        except (ValueError, TypeError, KeyError, IndexError, RecursionError):
            raise PolicyProtocolError("invalid_tool_xml") from None
        prompt = arguments["prompt"]
        reserved = r"<\|[^>]+\|>|</?(?:tool_call|tool_response|tools|think)>|<(?:function|parameter)=|</(?:function|parameter)>"
        if any(re.search(reserved, part) for part in (prompt, message["content"])):
            raise PolicyProtocolError("control_markup_in_draw_prompt")
        # Unlike the older JSON template, Qwen3.5 iterates arguments.items().
        # This affects only context rendering, never the sampled action tokens.
        message["tool_calls"][0]["function"]["arguments"] = arguments
        return message, prompt
    try:
        value = json.loads(match.group(2), object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError):
        raise PolicyProtocolError("invalid_tool_json") from None
    if (not isinstance(value, dict) or set(value) != {"name", "arguments"} or value["name"] != "draw"
            or not isinstance(value["arguments"], dict) or set(value["arguments"]) != {"prompt"}):
        raise PolicyProtocolError("unsupported_tool_or_arguments")
    prompt = value["arguments"]["prompt"]
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 10000:
        raise PolicyProtocolError("invalid_draw_prompt")
    if re.search(r"<\|[^>]+\|>|</?(?:tool_call|tool_response|tools)>", prompt):
        raise PolicyProtocolError("control_markup_in_draw_prompt")
    return {"role": "assistant", "content": match.group(1).strip(), "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "draw", "arguments": json.dumps(value["arguments"], ensure_ascii=False)}}]}, prompt


def validate_initial_messages(messages):
    if (len(messages) != 2 or messages[0].get("role") != "system" or messages[0].get("content") != SYSTEM_PROMPT
            or messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), str)
            or not messages[1]["content"].strip()):
        raise ValueError("RL prompts must contain the exact no-tools SYSTEM_PROMPT and one original user request")
    if re.search(r"<\|[^>]+\|>|</?(?:tool_call|tool_response|tools)>", messages[1]["content"]):
        raise ValueError("User scene contains reserved chat-template markup")
    return messages[1]["content"]


def observation_messages(*, image, image_path, image_sha256, call_id, used, limit, same_prompt, same_image):
    """Match SFT's text-tool + user-image projection; never put pixels in text."""
    feedback = {"status": "ok", "image_path": str(image_path), "sha256": image_sha256,
                "runtime_state": draw_runtime_state(used=used, limit=limit,
                                                    same_prompt=same_prompt, same_image=same_image)}
    return [{"role": "tool", "tool_call_id": call_id, "name": "draw",
             "content": json.dumps(feedback, ensure_ascii=False)},
            {"role": "user", "content": [{"type": "text", "text": IMAGE_REVIEW_PROMPT},
                                          {"type": "image", "image": image}]}]


def recovery_messages(*, reason, used, limit, remaining_turns, call_id=None):
    """Runner feedback is context, never a policy target or a quality verdict.

    A parsed but illegal draw receives a tool response. Malformed calls have no
    valid tool identity, so they receive a user-role runtime correction instead.
    """
    feedback = {"status": "invalid_action", "error": reason,
                "successful_draws": used, "remaining_draws": max(0, limit - used),
                "remaining_assistant_turns": remaining_turns,
                "instruction": ("Correct the action using the original user request and native draw interface. "
                                "Before any image, call draw. After observing an image, submit with brief final "
                                "text or redraw if budget remains. This is execution feedback, not a quality score.")}
    if call_id is not None:
        return [{"role": "tool", "tool_call_id": call_id, "name": "draw",
                 "content": json.dumps(feedback, ensure_ascii=False)}]
    return [{"role": "user", "content": "Runtime action feedback: " + json.dumps(feedback, ensure_ascii=False)}]
