"""Keep native tools/calls/roles; adapt only image part spelling for AutoProcessor."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path

PROJECTION_VERSION = "native-tool-text-image-user-v1"
MAX_THOUGHT_SIGNATURE_CHARACTERS = 131072
_RESERVED_MARKUP = re.compile(r"<\|[^<>\r\n]{1,100}\|>|</?(?:tool_call|tool_response|tools|think)>")
_XML_TOOL_MARKUP = re.compile(r"<(?:function|parameter)=|</(?:function|parameter)>")
QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})


def reject_reserved_markup(value, _depth=0):
    """Raw text must not manufacture the chat template's roles/tools/vision tokens.

    The template itself adds these delimiters. Reject, rather than silently
    rewriting, reviewed text so a literal delimiter cannot become supervision
    for an unrecorded message or an unrecorded tool call.
    """
    if _depth > 64:
        raise ValueError("Native message/schema nesting is too deep")
    if isinstance(value, str):
        if _RESERVED_MARKUP.search(value):
            raise ValueError("Raw text contains reserved chat/template control markup")
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_reserved_markup(key, _depth + 1)
            reject_reserved_markup(item, _depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_reserved_markup(item, _depth + 1)


def validate_transport_metadata(call):
    """Gemini's opaque signature is transport only; it is never model text."""
    if "extra_content" not in call:
        return
    extra = call["extra_content"]
    if not isinstance(extra, dict) or set(extra) != {"google"}:
        raise ValueError("Unsupported native tool transport metadata")
    google = extra["google"]
    if not isinstance(google, dict) or set(google) != {"thought_signature"}:
        raise ValueError("Unsupported native tool transport metadata")
    signature = google["thought_signature"]
    if (not isinstance(signature, str) or not signature.strip()
            or len(signature) > MAX_THOUGHT_SIGNATURE_CHARACTERS):
        raise ValueError("Native tool thought_signature must be a bounded nonempty string")


def normalize_messages(messages: list[dict]) -> list[dict]:
    result = [{k: copy.deepcopy(v) for k, v in m.items()
               if k in {"role", "content", "tool_calls", "tool_call_id", "name"}} for m in messages]
    for message in result:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported native message role")
        message.pop("message_id", None)
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("Native tool_calls must be a list when present")
        for call in calls:
            validate_transport_metadata(call)
            # Work on a deep copy: archived signatures and approval hashes stay
            # intact. The Qwen processor must never tokenize opaque API state.
            call.pop("extra_content", None)
        reject_reserved_markup(message)
        content = message.get("content", "")
        message["content"] = content
        if content is None:
            message["content"] = ""
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else image
                    if not isinstance(url, str) or not url or "://" in url or url.startswith("data:"):
                        raise ValueError("Only local image paths enter the processor")
                    part.clear()
                    part.update(type="image", image=url)
                elif part.get("type") == "image" and isinstance(part.get("image"), str):
                    pass
                elif part.get("type") == "text" and isinstance(part.get("text"), str):
                    pass
                else:
                    raise ValueError("Unsupported multimodal part")
        elif not isinstance(content, str):
            raise ValueError("Native content must be text or multimodal parts")
    return result


def ordered_image_paths(messages: list[dict]) -> list[Path]:
    return [Path(part["image"]) for message in messages if isinstance(message.get("content"), list)
            for part in message["content"] if part.get("type") == "image"]


def processor_messages(messages: list[dict], *, model_type: str | None = None) -> list[dict]:
    """Project native API calls into the selected model's documented template.

    Qwen3-VL accepts the archived JSON argument string. Qwen3.5's XML template
    iterates ``arguments.items()`` and therefore requires an actual mapping.
    This is a tokenizer-only projection: never rewrite archived trajectories or
    approval hashes, and never turn Gemini's opaque signatures into target text.
    """
    result = normalize_messages(messages)
    if model_type not in QWEN35_MODEL_TYPES:
        return result

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Native tool arguments contain duplicate JSON keys")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"Native tool arguments contain non-finite JSON value: {value}")

    def reject_xml(value):
        if isinstance(value, str) and _XML_TOOL_MARKUP.search(value):
            raise ValueError("Raw text contains reserved XML tool control markup")
        if isinstance(value, dict):
            for key, item in value.items():
                reject_xml(key)
                reject_xml(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_xml(item)

    for message in result:
        for call in message.get("tool_calls", []):
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("Native tool call requires a function object")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments, object_pairs_hook=object_pairs,
                                       parse_constant=invalid_constant)
            if not isinstance(arguments, dict):
                raise ValueError("Native tool arguments must encode a JSON object")
            function["arguments"] = arguments
        # Check decoded strings as well: escaped XML in JSON must not manufacture
        # a closing tool parameter when inserted verbatim into the XML template.
        reject_reserved_markup(message)
        reject_xml(message)
    return result
