"""Native API adapter within Qwen-Agent's BaseChatModel extension interface."""
import copy
import math
import urllib.parse
import uuid

from qwen_agent.llm.base import BaseChatModel

from .messages import (native_assistant_to_qwen, project_messages,
                       qwen_messages_to_native, validate_assistant)
from .providers import (HTTPTransport, ProviderError, _endpoint, _env_key,
                        _metadata, image_data_uri, safe_media_path)
from .qwen_compat import restore_qwen_dns_resolver

# Qwen imports its optional image search tool even when it is not enabled.
restore_qwen_dns_resolver()


class NativeChatModel(BaseChatModel):
    """Qwen-Agent model adapter for full deployment URLs, native tools and images.

    Overrides raw_chat rather than using Qwen-Agent's text-only OpenAI adapter,
    whose message conversion can erase multimodal input. No synthetic function-call
    tokens or JSON decision wrappers are introduced into assistant outputs.
    """
    def __init__(self, chat_completions_url, model, api_key_env="OPTICS_TEACHER_API_KEY",
                 auth="api-key", transport=None, max_tokens=2048, temperature=0.3,
                 enable_thinking=None, is_demo=False, token_parameter="max_tokens", include_model=True):
        parsed = urllib.parse.urlsplit(chat_completions_url)
        _endpoint(urllib.parse.urlunsplit(parsed._replace(query="")))
        query = urllib.parse.parse_qs(parsed.query)
        if set(query) - {"api-version"} or any(len(values) != 1 for values in query.values()):
            raise ValueError("Only the public api-version query parameter is supported")
        if auth not in {"api-key", "bearer"}:
            raise ValueError("auth must be api-key or bearer")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError("temperature must be null or finite and between 0 and 2")
        if not isinstance(token_parameter, str) or token_parameter not in {"max_tokens", "max_completion_tokens"} or type(include_model) is not bool:
            raise ValueError("Invalid token_parameter or include_model policy")
        self.url, self.auth = chat_completions_url, auth
        self.key = "offline-demo" if is_demo else _env_key(api_key_env)
        self.transport = transport or HTTPTransport()
        self.max_tokens, self.temperature = max_tokens, temperature
        self.token_parameter, self.include_model = token_parameter, include_model
        self.enable_thinking, self.is_demo = enable_thinking, bool(is_demo)
        self.last_metadata = None
        self.run_dir = None
        self.request_timeout = None
        self.expected_images = {}
        super().__init__({"model": model, "model_type": "optics_native",
                          "generate_cfg": {"use_raw_api": True, "max_input_tokens": -1, "max_retries": 0}})

    @property
    def support_multimodal_input(self):
        return True

    @property
    def identity(self):
        return {"provider": "offline_fixture" if self.is_demo else "native_chat_api",
                "model": self.model, "endpoint": self.url, "auth_header": self.auth,
                "generation": {"max_tokens": self.max_tokens, "temperature": self.temperature,
                               "token_parameter": self.token_parameter, "include_model": self.include_model,
                               "enable_thinking": self.enable_thinking, "parallel_tool_calls": False}}

    def bind_request(self, run_dir, timeout, *, expected_images=None):
        """Bind immutable artifact identities for this call; discard prior usage."""
        self.last_metadata = None
        self.run_dir, self.request_timeout = run_dir, timeout
        self.expected_images = copy.deepcopy(expected_images or {})

    def _preprocess_messages(self, messages, **kwargs):
        # Preserve exactly the archived input: no implicit retrieval, file upload
        # annotations, prompt rewriting, hidden function-call templates or truncation.
        return copy.deepcopy(messages)

    def raw_chat(self, messages, functions=None, stream=True, generate_cfg=None):
        if not stream:
            raise ValueError("The Agent uses Qwen-Agent's full-response iterator interface")
        if self.run_dir is None or self.request_timeout is None:
            raise ValueError("Bind a run-local request context before calling the model")
        self.last_metadata = None
        input_images = []
        def resolve_image(path):
            expected = self.expected_images.get(path)
            if expected is None:
                raise ProviderError("input_image_not_bound_to_artifact")
            data_uri = image_data_uri(safe_media_path(self.run_dir, path), expected_sha256=expected)
            input_images.append({"path": path, "sha256": expected})
            return data_uri
        native = qwen_messages_to_native(messages)
        payload = {"messages": project_messages(
            native, target="openai", resolve_image=resolve_image),
            "tools": [{"type": "function", "function": copy.deepcopy(function)} for function in functions or []],
            "tool_choice": "auto", "parallel_tool_calls": False, "stream": False,
            self.token_parameter: self.max_tokens}
        if self.include_model:
            payload["model"] = self.model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        headers = {"Content-Type": "application/json"}
        headers["api-key" if self.auth == "api-key" else "Authorization"] = self.key if self.auth == "api-key" else "Bearer " + self.key
        self.last_metadata = {"input_images": input_images}
        response, response_headers = self.transport.post_json(self.url, headers, payload, self.request_timeout)
        if not isinstance(response, dict):
            raise ProviderError("invalid_model_response")
        self.last_metadata = _metadata(response, response_headers, self.model, self.identity["provider"])
        self.last_metadata["input_images"] = input_images
        if response.get("error") or response.get("code"):
            raise ProviderError("model_request_rejected", self.last_metadata["request_id"])
        try:
            choice = response["choices"][0]
            if not isinstance(choice, dict):
                raise ValueError("Expected one completion choice")
            if choice.get("finish_reason") not in {"stop", "tool_calls", "function_call"}:
                raise ProviderError("incomplete_model_response", self.last_metadata["request_id"])
            native_message = copy.deepcopy(choice["message"])
            # Some Gemini OpenAI-compatible gateways return id="" for an
            # otherwise native function call. The ID is transport correlation,
            # not model reasoning or a tool argument. Give it a unique local ID
            # and preserve that same ID in subsequent assistant/tool messages.
            # Keep malformed payloads and every other provider fail-closed.
            if (self.model.casefold().startswith(("gemini-", "google/gemini-"))
                    and isinstance(native_message, dict)):
                calls = native_message.get("tool_calls")
                if (isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict)
                        and calls[0].get("type") == "function" and calls[0].get("id") in (None, "")):
                    calls[0]["id"] = "call_optics_" + uuid.uuid4().hex
                    self.last_metadata["normalizations"] = ["gemini_missing_tool_call_id"]
            message = validate_assistant(native_message, allowed_tools={
                function["name"] for function in functions or []})
            if choice["finish_reason"] in {"tool_calls", "function_call"} and not message.get("tool_calls"):
                raise ProviderError("tool_finish_without_native_call", self.last_metadata["request_id"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("invalid_native_tool_completion", self.last_metadata["request_id"]) from None
        yield [native_assistant_to_qwen(message)]

    # Required BaseChatModel interfaces. Native raw_chat is deliberately the sole
    # active path, so template-parsed or text-only modes fail explicitly.
    def _chat_with_functions(self, *args, **kwargs):
        raise NotImplementedError("Use native raw_chat")

    def _chat_stream(self, *args, **kwargs):
        raise NotImplementedError("Use native raw_chat")

    def _chat_no_stream(self, *args, **kwargs):
        raise NotImplementedError("Use native raw_chat")
