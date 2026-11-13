"""The native decision loop: ask, archive, draw, observe pixels, and eventually submit.

Budget/accounting/storage callbacks live in runner.py. Provider implementation is
separate from this decision flow, and any external judge runs after this loop.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Iterator

from qwen_agent import Agent
from qwen_agent.llm.schema import ContentItem, Message

from .messages import qwen_messages_to_native, validate_assistant
from .protocol import (IMAGE_REVIEW_PROMPT, REFERENCE_IMAGE_OBSERVATION_PROMPT,
                       RESEARCH_SYSTEM_PROMPT, SYSTEM_PROMPT)
from .providers import ProviderError
from .qwen_compat import restore_qwen_dns_resolver
from .tools import DrawTool

# Also covers locally hosted actors that do not import NativeChatModel.
restore_qwen_dns_resolver()

if TYPE_CHECKING:
    from .llm import NativeChatModel
    from .runner import AgentRunner


class OpticalAgent(Agent):
    """Native Qwen-Agent lifecycle with run-local auditing supplied by a recorder."""
    def __init__(self, llm: NativeChatModel, recorder: AgentRunner) -> None:
        self.recorder = recorder
        tools = [DrawTool(recorder.draw)]
        if recorder.research_enabled:
            from .tools import ImageSearchTool, KnowledgeLookupTool
            tools.extend([KnowledgeLookupTool(recorder.knowledge_lookup),
                          ImageSearchTool(recorder.image_search)])
        super().__init__(function_list=tools, llm=llm,
                         system_message=RESEARCH_SYSTEM_PROMPT if recorder.research_enabled else SYSTEM_PROMPT,
                         name=None)

    def _run(self, messages: list[Message], lang: str = "en", **kwargs: Any) -> Iterator[list[Message]]:
        """Yield the accumulated native response after each completed decision.

        A draw response is followed by a tool result and a user image observation.
        A plain assistant response submits the latest image and stops immediately.
        Exceptions propagate to the recorder; they never become synthetic images
        or optical correctness judgments inside the conversation.
        """
        messages = copy.deepcopy(messages)
        response = []
        seen_call_ids = set()
        self.recorder.begin_messages(qwen_messages_to_native(messages),
                                     [{"type": "function", "function": tool.function} for tool in self.function_map.values()])
        for number in range(1, self.recorder.config.max_steps + 1):
            self.recorder.before_model_call()
            context = qwen_messages_to_native(messages)
            output = []
            for output in self._call_llm(messages, functions=[tool.function for tool in self.function_map.values()], stream=True):
                pass
            self.recorder.after_model_call(self.llm.last_metadata)
            if len(output) != 1:
                raise ProviderError("expected_one_assistant_message")
            assistant = validate_assistant(qwen_messages_to_native(output)[0], allowed_tools=self.function_map)
            self.recorder.start_step(number, context, assistant, self.llm.last_metadata)
            # Preserve the actual response even when it used the remaining budget;
            # enforce the limit before executing its action or accepting completion.
            self.recorder.remaining()
            messages.extend(output)
            response.extend(output)
            if not assistant.get("tool_calls"):
                self.recorder.submit()
                yield response
                return
            call = assistant["tool_calls"][0]
            if call["id"] in seen_call_ids:
                raise ProviderError("reused_tool_call_id")
            seen_call_ids.add(call["id"])
            # The BaseTool performs schema verification and invokes the configured
            # generator; failures propagate to the run recorder instead of becoming
            # a guessed optical label or a verbose framework traceback observation.
            tool_name = call["function"]["name"]
            tool_items = self.function_map[tool_name].call(call["function"]["arguments"])
            text = "\n".join(item.text for item in tool_items if item.text)
            images = [item.image for item in tool_items if item.image]
            tool_msg = Message(role="function", name=tool_name, content=text, extra={"function_id": call["id"]})
            results = [tool_msg]
            if images:
                observation = Message(role="user", content=[ContentItem(
                    text=IMAGE_REVIEW_PROMPT if tool_name == "draw" else REFERENCE_IMAGE_OBSERVATION_PROMPT
                )] + [ContentItem(image=path) for path in images])
                results.append(observation)
            messages.extend(results)
            response.extend(results)
            self.recorder.finish_step(qwen_messages_to_native(results))
            yield response
        self.recorder.budget_exhausted("max_steps")
        yield response
