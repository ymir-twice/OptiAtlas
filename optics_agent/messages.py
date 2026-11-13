"""Versioned message projection shared by API input and SFT. Pure standard library.

Tool responses remain native role=tool. Image observations are explicit following
user messages, so neither Gemini adapters nor Qwen chat templates swallow images.
"""
import copy
import json

PROJECTION_VERSION = "native-tool-text-image-user-v1"
NATIVE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
MAX_THOUGHT_SIGNATURE_CHARACTERS = 131072


def _tool_extra_content(call):
    """Allow only Gemini's opaque continuity signature, never arbitrary metadata."""
    if "extra_content" not in call:
        return None
    extra = call["extra_content"]
    if not isinstance(extra, dict) or set(extra) != {"google"}:
        raise ValueError("Unsupported tool-call transport metadata")
    google = extra["google"]
    if not isinstance(google, dict) or set(google) != {"thought_signature"}:
        raise ValueError("Unsupported Google tool-call transport metadata")
    signature = google["thought_signature"]
    if not isinstance(signature, str) or not signature.strip() or len(signature) > MAX_THOUGHT_SIGNATURE_CHARACTERS:
        raise ValueError("Thought signature must be a bounded nonempty opaque string")
    return copy.deepcopy(extra)


def project_messages(messages, *, target="openai", resolve_image=lambda path: path):
    """Remove archive IDs; retain native calls; map images without losing order.

    target='openai': image_url parts (resolver can produce data URIs).
    target='qwen': type=image/image parts for AutoProcessor chat templates.
    """
    if target not in {"openai", "qwen"}:
        raise ValueError("target must be openai or qwen")
    output = []
    for archived in messages:
        item = {key: copy.deepcopy(value) for key, value in archived.items() if key in NATIVE_KEYS}
        if item.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported native message role")
        content = item.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    path = resolve_image(part["image_url"]["url"])
                    parts.append({"type": "image_url", "image_url": {"url": path}} if target == "openai"
                                 else {"type": "image", "image": path})
                else:
                    raise ValueError("Unsupported message content part")
            content = parts
        elif not isinstance(content, str):
            raise ValueError("Message content must be text or content parts")
        item["content"] = content
        if target == "qwen":
            # Transport-only encrypted state belongs to the teacher API history,
            # never the student's prompt or supervised tool-call serialization.
            for call in item.get("tool_calls") or []:
                call.pop("extra_content", None)
        output.append(item)
    return output


def qwen_messages_to_native(messages):
    result = []
    for message in messages:
        raw = message.model_dump() if hasattr(message, "model_dump") else message
        content = raw.get("content", "")
        if isinstance(content, list):
            content = [({"type": "text", "text": part["text"]} if "text" in part
                        else {"type": "image_url", "image_url": {"url": part["image"]}})
                       for part in content]
        item = {"role": raw["role"], "content": content}
        if raw["role"] == "function":
            item["role"] = "tool"
            item["tool_call_id"] = raw.get("extra", {}).get("function_id")
            item["name"] = raw["name"]
        if raw.get("function_call"):
            item["tool_calls"] = [{"id": raw.get("extra", {}).get("function_id"), "type": "function",
                                   "function": copy.deepcopy(raw["function_call"])}]
            extra = raw.get("extra", {}).get("optics_tool_extra_content")
            if extra is not None:
                item["tool_calls"][0]["extra_content"] = _tool_extra_content({"extra_content": extra})
        result.append(item)
    return project_messages(result)


def native_assistant_to_qwen(message):
    from qwen_agent.llm.schema import FunctionCall, Message
    calls = message.get("tool_calls", [])
    kwargs = {"role": "assistant", "content": message.get("content") or ""}
    if calls:
        kwargs["function_call"] = FunctionCall(**calls[0]["function"])
        kwargs["extra"] = {"function_id": calls[0]["id"]}
        extra = _tool_extra_content(calls[0])
        if extra is not None:
            kwargs["extra"]["optics_tool_extra_content"] = extra
    return Message(**kwargs)


def validate_tool_arguments(name, args):
    """Validate the bounded native interface without importing provider packages."""
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object")
    if name == "draw":
        if set(args) != {"prompt"} or not isinstance(args["prompt"], str) or not 1 <= len(args["prompt"].strip()) <= 10000:
            raise ValueError("draw accepts only a nonempty text prompt of at most 10000 characters")
    elif name == "knowledge_lookup":
        if set(args) - {"category_ids", "query", "max_results"} or "category_ids" not in args:
            raise ValueError("knowledge_lookup requires category_ids and optional query/max_results")
        ids, query, limit = args["category_ids"], args.get("query", ""), args.get("max_results", 3)
        if (not isinstance(ids, list) or len(ids) > 5
                or any(not isinstance(value, str) or not value.strip() or len(value) > 100 for value in ids)
                or not isinstance(query, str) or len(query) > 1000
                or type(limit) is not int or not 1 <= limit <= 5):
            raise ValueError("Invalid knowledge_lookup arguments")
    elif name == "image_search":
        if set(args) - {"query", "max_results"} or "query" not in args:
            raise ValueError("image_search requires query and optional max_results")
        query, limit = args["query"], args.get("max_results", 2)
        if (not isinstance(query, str) or not 1 <= len(query.strip()) <= 500
                or type(limit) is not int or not 1 <= limit <= 2):
            raise ValueError("Invalid image_search arguments")
    else:
        raise ValueError("Unsupported native tool")
    return args


def validate_assistant(message, *, allowed_tools=None):
    allowed = {"draw"} if allowed_tools is None else set(allowed_tools)
    if allowed - {"draw", "knowledge_lookup", "image_search"}:
        raise ValueError("Unsupported allowed tool")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected one native assistant message")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str) or len(content) > 6000:
        raise ValueError("Assistant text must be a bounded string")
    result = {"role": "assistant", "content": content}
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ValueError("tool_calls must be a list or null")
    if message.get("function_call") is not None:
        raise ValueError("Legacy function_call responses are unsupported; use native tool_calls")
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Exactly one tool call per assistant message")
        call = calls[0]
        if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("id"), str) or not 1 <= len(call["id"]) <= 200:
            raise ValueError("Native tool call needs type=function and a nonempty id")
        function = call.get("function", {})
        if not isinstance(function, dict) or function.get("name") not in allowed or not isinstance(function.get("arguments"), str):
            raise ValueError("Only enabled native tools with JSON arguments are supported")
        args = json.loads(function["arguments"])
        validate_tool_arguments(function["name"], args)
        result["tool_calls"] = [{"id": call["id"], "type": "function", "function": {"name": function["name"], "arguments": function["arguments"]}}]
        extra = _tool_extra_content(call)
        if extra is not None:
            result["tool_calls"][0]["extra_content"] = extra
    elif not content.strip():
        raise ValueError("Empty completion is neither a draw call nor a final response")
    return result
