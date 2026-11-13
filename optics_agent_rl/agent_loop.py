"""Official VERL AgentLoop for the existing optical draw/observe/redraw Agent.

VERL owns policy sampling, GRPO advantages, actor/reference log probabilities,
distributed training and checkpointing. This module supplies the environment and
terminal image reward; it never implements an alternative policy-gradient loss.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from omegaconf import OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from .protocol import (DRAW_SCHEMA, PolicyProtocolError, observation_messages, parse_assistant, recovery_messages,
                       validate_initial_messages, validate_tool_serialization)
from .services import FrozenDrawClient, ServiceError, TerminalJudgeClient
from .validity import ExecutionValidity
from .quality_rules import quality_rubric
from .checklists import prompt_checklist
from .reward import judge_prompt_style


def _json_messages(messages, images_by_id):
    """Log local image references without embedding pixels into readable JSON."""
    def convert(value):
        if isinstance(value, Image.Image):
            return images_by_id[id(value)]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return convert(messages)


@register("optical_agent")
class OpticalAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        settings = self.config.get("optics_agent_rl")
        if settings is None:
            raise ValueError("Missing optics_agent_rl runtime configuration")
        self.settings = OmegaConf.to_container(settings, resolve=True) if OmegaConf.is_config(settings) else dict(settings)
        skip_failed = self.settings.get("judge", {}).get("skip_failed_samples", False)
        if not isinstance(skip_failed, bool):
            raise ValueError("judge.skip_failed_samples must be boolean")
        if skip_failed:
            # Empty outputs are supported by official V1, but its default sampler
            # admits partial groups. Require our group rejection/refill boundary.
            trainer = self.config.trainer
            sampler = trainer.v1.sampler
            if (not trainer.use_v1 or trainer.v1.trainer_mode != "sync"
                    or sampler.custom_sampler.name != "JudgeSkipReplayBuffer"
                    or not sampler.get("sync_refill_failed_groups", False)
                    or sampler.sampler_kwargs.get("expected_group_size") != self.config.actor_rollout_ref.rollout.n):
                raise ValueError("Judge skipping requires synchronous V1 JudgeSkipReplayBuffer with exact group size/refill")
            if trainer.get("val_before_train", True) or trainer.get("val_only", False) or trainer.get("test_freq", 0) > 0:
                raise ValueError("Judge skipping supports training only; disable in-run validation")
        self.tool_call_parser = validate_tool_serialization(
            self.settings.get("tool_call_parser", "hermes"), self.apply_chat_template_kwargs)
        self.max_draws = int(self.settings.get("max_draws", 3))
        self.max_assistant_turns = int(self.settings.get("max_assistant_turns", self.max_draws + 3))
        self.max_tokens_per_turn = int(self.settings.get("max_tokens_per_turn", 4096))
        if self.max_draws < 1 or self.max_assistant_turns < self.max_draws + 1 or self.max_tokens_per_turn < 1:
            raise ValueError("Provide at least one post-draw policy turn beyond the enforced draw budget")
        if self.processor is None:
            raise ValueError("The optical RL Agent requires a Qwen VL processor")
        self._assert_mm_supported(True)
        if self.rollout_config.name != "vllm":
            raise ValueError("The optical Agent integration currently validates the official vLLM rollout backend")

    async def run(self, sampling_params, priority=0, **kwargs):
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        original_request = validate_initial_messages(messages)
        extra_info = kwargs.get("extra_info") or {}
        request_id = uuid4().hex
        artifact = Path(self.settings["artifact_dir"]).expanduser().resolve() / request_id
        artifact.mkdir(parents=True, exist_ok=False)
        events_path = artifact / "events.jsonl"

        def event(name, **fields):
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": name, "time": time.time(), **fields}, ensure_ascii=False) + "\n")

        event("rollout_started", sample_id=extra_info.get("sample_id", extra_info.get("index")), request_id=request_id)
        runtime_ids, mask, logprobs = [], [], None
        image_data, image_paths = [], {}
        draws, turns, last_caption, last_hash, final_image = 0, 0, None, None, None
        metrics = {"generate_sequences": 0., "tool_calls": 0., "compute_score": 0., "num_preempted": 0}
        reward, judgement, status = 0., None, "max_assistant_turns"
        validity = ExecutionValidity()
        quality_reward = 0.
        exception = None
        policy_versions = {}
        draw = judge = None
        prompt_scoring = prompt_physics = prompt_binary = hybrid_quality = image_only = False
        initial_ids = []
        try:
            # Fail metadata/configuration errors before paying for a rollout.
            # Only the explicit simple-judge profile bypasses the 37-class rubric.
            category_id = extra_info.get("category_id")
            style = judge_prompt_style(self.settings["judge"])
            prompt_scoring = style == "simple_draw_prompt_v1"
            prompt_binary = style == "prompt_physics_binary_v1"
            prompt_physics = style in {"prompt_physics_product_v1", "prompt_physics_binary_v1"}
            hybrid_quality = style == "prompt_binary_image_quality_v1"
            image_only = style == "image_quality_only_v1"
            contract = self.settings.get("reward_contract_version")
            for text_contract in ("prompt_physics_product_v1", "prompt_physics_binary_v1", "prompt_binary_image_quality_v1", "image_quality_only_v1"):
                if ((contract == text_contract) != (style == text_contract)):
                    raise ValueError(f"Prompt physics contract requires matching {text_contract} judge and vice versa")
            if ((contract == "simple_draw_prompt_quality_validity_v1") != prompt_scoring):
                raise ValueError("Draw-prompt quality contract requires simple_draw_prompt_v1 judge and vice versa")
            if contract == "simple_quality_validity_v1" and style != "legacy_simple_v1":
                raise ValueError("Simple quality contract requires legacy_simple_v1 judge")
            if contract == "quality_validity_v2" and style != "category_rules_v2":
                raise ValueError("Quality-validity v2 requires category_rules_v2 judge")
            if ((contract == "checklist_quality_validity_v1") != (style == "prompt_checklist_v1")):
                raise ValueError("Checklist reward contract requires prompt_checklist_v1 judge and vice versa")
            if ((contract == "online_checklist_quality_validity_v1") != (style == "online_checklist_v1")):
                raise ValueError("Online checklist reward contract requires online_checklist_v1 judge and vice versa")
            if contract in {"quality_validity_v2", "simple_quality_validity_v1", "checklist_quality_validity_v1",
                            "online_checklist_quality_validity_v1", "simple_draw_prompt_quality_validity_v1",
                            "prompt_physics_product_v1", "prompt_physics_binary_v1", "prompt_binary_image_quality_v1", "image_quality_only_v1"} and category_id is None:
                raise ValueError("Quality-validity v2 rollouts require extra_info.category_id")
            if style in {"category_rules_v2", "online_checklist_v1", "prompt_physics_product_v1", "prompt_physics_binary_v1", "prompt_binary_image_quality_v1", "image_quality_only_v1"}:
                quality_rubric(category_id)
            elif style == "prompt_checklist_v1":
                # Only a published, image-independent checklist is allowed.
                # Fail missing/configuration errors before sampling or drawing.
                prompt_checklist(original_request, category_id, self.settings["judge"].get("checklist_path"))
            elif category_id is not None and (not isinstance(category_id, str) or not category_id.strip()):
                raise ValueError("Simple judge category_id must be nonempty text or null")
            # Unlike the generic text helper, never left-truncate the authoritative
            # original request/system contract when configuration is too small.
            initial_ids = await self.loop.run_in_executor(None, lambda: self.continuous_token_builder.build_initial_tokens(
                messages, tools=[DRAW_SCHEMA], images=None))
            if not initial_ids or len(initial_ids) > self.rollout_config.prompt_length:
                raise ValueError("Original Agent prompt exceeds rollout.prompt_length; increase it, do not truncate")
            runtime_ids = list(initial_ids)
            draw = FrozenDrawClient(self.settings["draw"], event=event)
            # Construct now so missing judge credentials cannot waste a GPU rollout.
            judge = TerminalJudgeClient(self.settings["judge"], self.settings.get("reward", {}), event=event)
            stop_ids = {self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
            if self.tokenizer.eos_token_id is not None:
                stop_ids.add(self.tokenizer.eos_token_id)
            if None in stop_ids or any(not isinstance(token_id, int) for token_id in stop_ids):
                raise ValueError("Qwen end-of-turn token identity is unavailable")
            # Generated visual/control placeholders have no backing image and
            # would corrupt multimodal feature alignment during PPO. vLLM's
            # native bad_words filter prevents sampling them; EOS stays allowed.
            added_vocab = (self.tokenizer.get_added_vocab() if hasattr(self.tokenizer, "get_added_vocab")
                           else self.tokenizer.get_vocab())
            blocked = {token: index for token, index in added_vocab.items()
                       if token == "<|im_start|>" or re.fullmatch(r"<\|(?:vision|image|video|audio)[^|]*\|>", token)}
            blocked_ids = set(blocked.values())

            async def recover(reason, turn, *, call_id=None, terminated=True):
                nonlocal runtime_ids, mask, logprobs, messages
                validity.recoverable(reason)
                event("policy_action_violation", reason=reason, severity="recoverable",
                      validity_reward=validity.reward, turn=turn + 1)
                if turn + 1 >= self.max_assistant_turns:
                    return False
                additions = recovery_messages(reason=reason, used=draws, limit=self.max_draws,
                                              remaining_turns=self.max_assistant_turns - turn - 1,
                                              call_id=call_id)
                # A length-truncated sampled turn has no closing ChatML token.
                # Close it as runner context (mask=0), preserving every sampled
                # token and its actual logprob rather than retokenizing the turn.
                ids, context_mask = list(runtime_ids), list(mask)
                context_logs = list(logprobs) if logprobs is not None else None
                if not terminated:
                    ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                    context_mask.append(0)
                    if context_logs is not None:
                        context_logs.append(0.)
                updated = [*messages, *additions]
                merged, next_mask, next_logs = await self.ct_merge_context_msg(
                    messages, updated, ids, context_mask, context_logs, tools=[DRAW_SCHEMA])
                # Leave at least one sampled policy token for a terminal reward.
                # Do not admit feedback without a following assistant turn.
                if len(next_mask) >= self.rollout_config.response_length:
                    event("recovery_budget_exhausted", reason=reason)
                    return False
                if sum(next_mask) != sum(mask):
                    raise RuntimeError("Runtime correction unexpectedly acquired policy loss")
                runtime_ids, mask, logprobs, messages = merged.token_ids, next_mask, next_logs, updated
                return True

            for turn in range(self.max_assistant_turns):
                remaining = self.rollout_config.response_length - len(mask)
                if remaining < 1:
                    status = "response_budget_exhausted"
                    validity.recoverable(status)
                    break
                params = dict(sampling_params)
                params["max_tokens"] = min(self.max_tokens_per_turn, remaining)
                params["bad_words"] = list(dict.fromkeys([*(params.get("bad_words") or []), *blocked]))
                # No stop-at-tool-close: the sampled native assistant turn includes
                # its real terminator, as in SFT. All sampled tokens remain targets.
                started = time.monotonic()
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=runtime_ids, sampling_params=params,
                    image_data=image_data or None, mm_processor_kwargs=self._get_mm_processor_kwargs(), priority=priority)
                metrics["generate_sequences"] += time.monotonic() - started
                metrics["num_preempted"] += output.num_preempted or 0
                for key in ("min_global_steps", "max_global_steps"):
                    value = (getattr(output, "extra_fields", None) or {}).get(key)
                    if value is not None:
                        reducer = min if key == "min_global_steps" else max
                        policy_versions[key] = reducer(policy_versions.get(key, value), value)
                if not output.token_ids or output.stop_reason == "aborted":
                    raise ServiceError("policy", "empty_or_aborted_generation")
                if len(output.token_ids) > params["max_tokens"]:
                    raise ServiceError("policy", "generation_exceeded_token_budget")
                if blocked_ids.intersection(output.token_ids):
                    # Do not sanitize/re-tokenize sampled actions or attach
                    # fabricated image features to recover an unsafe stream.
                    raise ServiceError("policy", "unbacked_visual_or_role_token")
                turns += 1
                sampled_logprobs = output.log_probs
                if sampled_logprobs is not None and len(sampled_logprobs) != len(output.token_ids):
                    raise ServiceError("policy", "logprob_length_mismatch")
                if self.rollout_config.calculate_log_probs and sampled_logprobs is None:
                    raise ServiceError("policy", "missing_sampled_logprobs")
                if sampled_logprobs is not None and logprobs is None:
                    if mask:
                        raise ServiceError("policy", "inconsistent_logprob_availability")
                    logprobs = []
                previous_targets = sum(mask)
                merged, mask, logprobs = await self.ct_merge_assistant_token(
                    runtime_ids, output.token_ids, mask, logprobs, assistant_logprobs=sampled_logprobs)
                runtime_ids = merged.token_ids
                if sum(mask) - previous_targets != len(output.token_ids):
                    raise RuntimeError("Continuous-token merge dropped sampled policy supervision")
                text = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
                event("policy_turn", turn=turn + 1, text=text, sampled_tokens=len(output.token_ids),
                      sampled_logprob_count=len(sampled_logprobs) if sampled_logprobs is not None else None)
                # Rejected actions do not consume draw budget, but their tool
                # responses still need a unique call identity on every turn.
                call_id = f"draw_turn_{turn + 1}"
                try:
                    message, caption = parse_assistant(
                        text, terminated=output.token_ids[-1] in stop_ids, call_id=call_id,
                        tool_call_parser=self.settings.get("tool_call_parser", "hermes"))
                except PolicyProtocolError as error:
                    status = str(error)
                    # Only the readable trace uses this string representation;
                    # training retains the untouched sampled token stream above.
                    messages.append({"role": "assistant", "content": text.strip().removesuffix(
                        "<|im_end|>").removesuffix("<|endoftext|>").strip()})
                    if await recover(status, turn, terminated=output.token_ids[-1] in stop_ids):
                        continue
                    break
                messages.append(message)
                if caption is None:
                    if not draws:
                        status = "submit_without_image"
                        validity.fatal(status)
                        event("policy_action_violation", reason=status, severity="fatal", validity_reward=validity.reward)
                        break
                    # Native final text always selects the last observed image;
                    # arbitrary prose/Markdown paths are not submission selectors.
                    # Loss of a real generated file is infrastructure failure.
                    if final_image is None or not final_image.is_file():
                        raise ServiceError("draw", "submitted_image_unavailable")
                    status = "submitted"
                    started = time.monotonic()
                    try:
                        # last_caption is updated only after successful drawing
                        # and observation, together with final_image/last_hash.
                        # A rejected/failed later draw or final prose cannot
                        # replace the instruction bound to the submitted pixels.
                        judge_options = {"draw_prompt": last_caption} if prompt_scoring or prompt_physics or hybrid_quality else {}
                        judgement = await judge.score(original_request, final_image, image_calls=draws,
                                                      category_id=extra_info.get("category_id"), **judge_options)
                    except ServiceError as error:
                        if not (self.settings["judge"].get("skip_failed_samples", False)
                                and error.service == "judge"
                                and error.kind in ({"http_error", "transport_error", "invalid_judgement",
                                                   "invalid_response_json", "invalid_response_envelope",
                                                   "response_too_large", "total_timeout"}
                                                   | ({"admission_timeout", "provider_error", "invalid_image", "image_byte_limit"}
                                                      if hybrid_quality or image_only else set()))
                                and not (error.kind == "http_error" and error.status in
                                         ({400, 401, 402, 403, 404, 422} if prompt_physics or hybrid_quality or image_only else {401, 403, 404}))):
                            raise
                        status = "judge_skipped"
                        exception = {"type": type(error).__name__, "service": error.service,
                                     "kind": error.kind, "http_status": error.status,
                                     "diagnostics": error.diagnostics}
                        event("rollout_judge_skipped", **exception)
                        # No reward tensor, placeholder score, or policy sample is
                        # emitted. V1 accepts []; JudgeSkipReplayBuffer excludes
                        # the entire incomplete group before any trainer stage.
                        return []
                    finally:
                        metrics["compute_score"] += time.monotonic() - started
                    quality_reward = judgement["reward"]
                    event("terminal_judgement", **judgement)
                    break
                if draws >= self.max_draws:
                    status = "draw_budget_exceeded"
                    if await recover(status, turn, call_id=call_id):
                        continue
                    break
                # Last policy turn cannot draw without leaving a subsequent turn
                # for actual-image observation and submission.
                if turn + 1 == self.max_assistant_turns:
                    status = "draw_without_review_budget"
                    validity.recoverable(status)
                    break
                started = time.monotonic()
                destination = artifact / f"draw_{draws + 1:02d}.png"
                image, digest = await draw.draw(caption, destination=destination, request_id=request_id, draw_index=draws)
                metrics["tool_calls"] += time.monotonic() - started
                draws += 1
                additions = observation_messages(image=image, image_path=destination, image_sha256=digest,
                                                 call_id=call_id, used=draws, limit=self.max_draws,
                                                 same_prompt=caption == last_caption, same_image=digest == last_hash)
                previous_messages = list(messages)
                updated_messages = [*messages, *additions]
                merged, next_mask, next_logprobs = await self.ct_merge_context_msg(
                    previous_messages, updated_messages, runtime_ids, mask, logprobs, tools=[DRAW_SCHEMA])
                # No token slicing through visual placeholders: it breaks the
                # image-feature alignment of both rollout and policy training.
                if len(next_mask) >= self.rollout_config.response_length:
                    event("observation_budget_exhausted", image_path=str(destination))
                    # The environment could not expose actual pixels within its
                    # configured capacity. Do not create an artificial zero or
                    # negative policy reward for a truncated visual observation.
                    raise ServiceError("runtime", "observation_exceeds_response_budget")
                if sum(next_mask) != sum(mask):
                    raise RuntimeError("Tool/user/image context unexpectedly acquired policy loss")
                runtime_ids, mask, logprobs, messages = merged.token_ids, next_mask, next_logprobs, updated_messages
                image_data.append(image)
                image_paths[id(image)] = str(destination)
                last_caption, last_hash, final_image = caption, digest, destination
                event("draw_observed", draw=draws, image_path=str(destination), image_sha256=digest,
                      context_tokens=len(next_mask), supervised_tokens=sum(mask))
            if status == "max_assistant_turns":
                validity.recoverable(status)
            # Explicit prompt, image-only and hybrid objectives have no execution penalty or bonus.
            # Without a legal submission no judge ran, so quality stays zero;
            # infrastructure failures still propagate or skip the whole group.
            reward = float(quality_reward if prompt_physics or hybrid_quality or image_only else quality_reward + validity.reward)
            if not mask or len(runtime_ids) != len(initial_ids) + len(mask):
                raise RuntimeError("Agent response tokens/masks no longer align with original prompt")
            if logprobs is not None and len(logprobs) != len(mask):
                raise RuntimeError("Agent response log probabilities are misaligned")
            if not mask[-1]:
                raise RuntimeError("Terminal reward must attach to a sampled assistant token")
            if hybrid_quality:
                summary_quality = {
                    "prompt_physics_score": judgement["prompt_physics_score"] if judgement else 0.,
                    "image_quality_score": judgement["image_quality_score"] if judgement else 0.,
                    "optical_score": judgement["optical_score"] if judgement else 0.,
                    "prompt_score": judgement["prompt_score"] if judgement else 0.,
                    "assessable": bool(judgement and judgement["assessable"]),
                }
                reward_quality = {**summary_quality, "assessable": float(summary_quality["assessable"]),
                    **{key: judgement[key] if judgement else 0. for key in ("physics_score", "alignment_score")},
                    **{key: judgement["reward_components"][key] if judgement else 0.
                       for key in ("prompt_binary_reward", "image_quality_reward")}}
            elif prompt_binary:
                summary_quality = {
                    "prompt_physics_score": judgement["score"] if judgement else 0.,
                    "prompt_physics_reward": quality_reward,
                }
                reward_quality = dict(summary_quality)
            elif prompt_physics:
                summary_quality = {
                    "correctness_score": judgement["correctness_score"] if judgement else 0.,
                    "completeness_score": judgement["completeness_score"] if judgement else 0.,
                    "prompt_physics_reward": quality_reward,
                }
                reward_quality = dict(summary_quality)
            else:
                summary_quality = {"optical_score": judgement["optical_score"] if judgement else 0.,
                                   "prompt_score": judgement["prompt_score"] if judgement else 0.,
                                   "assessable": bool(judgement and judgement["assessable"])}
                reward_quality = {**summary_quality, "assessable": float(summary_quality["assessable"]),
                                  "physics_score": (judgement.get("physics_score", judgement["optical_score"]) if judgement else 0.),
                                  "alignment_score": (judgement.get("alignment_score", judgement["prompt_score"]) if judgement else 0.)}
                if image_only:
                    reward_quality.update({key: judgement["reward_components"][key] if judgement else 0.
                                           for key in ("image_physics_reward", "image_alignment_reward")})
            output = AgentLoopOutput(
                prompt_ids=initial_ids, response_ids=runtime_ids[len(initial_ids):], response_mask=mask,
                response_logprobs=logprobs, multi_modal_data={"images": image_data} if image_data else {},
                mm_processor_kwargs=self._get_mm_processor_kwargs(), reward_score=reward,
                num_turns=len(messages), metrics=metrics,
                extra_fields={**policy_versions, "optical_status": status, "image_calls": draws, "policy_turns": turns,
                              "artifact_dir": str(artifact), **summary_quality,
                              "reward_extra_info": {
                                  **reward_quality,
                                  "image_calls": draws, "policy_turns": turns,
                                  "quality_reward": quality_reward, "validity_reward": validity.reward,
                                  "recoverable_violations": len(validity.recoverable_errors),
                                  "fatal_violation": float(validity.fatal_error is not None),
                                  **({"draw_prompt_score": judgement.get("draw_prompt_score", 0.) if judgement else 0.,
                                      **{key: (judgement.get("reward_components", {}).get(key, 0.) if judgement else 0.)
                                         for key in ("image_physics_reward", "image_alignment_reward", "draw_prompt_reward")}}
                                     if prompt_scoring else {}),
                                  "submitted": float(status == "submitted"),
                                  "budget_exceeded": float(status in {"draw_budget_exceeded", "response_budget_exhausted",
                                                                      "observation_exceeds_response_budget", "draw_without_review_budget"}),
                              }})
            event("rollout_finished", status=status, reward=reward, quality_reward=quality_reward,
                  validity_reward=validity.reward, image_calls=draws, supervised_tokens=sum(mask))
            return output
        except BaseException as error:
            exception = {"type": type(error).__name__}
            if isinstance(error, ServiceError):
                exception.update(service=error.service, kind=error.kind, http_status=error.status)
            event("rollout_technical_failure", **exception)
            raise
        finally:
            record = {"request_id": request_id, "sample_id": extra_info.get("sample_id", extra_info.get("index")),
                      "original_request": original_request, "system_prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
                      "tools": [DRAW_SCHEMA], "messages": _json_messages(messages, image_paths),
                      "status": (status if status == "judge_skipped" else "technical_failure") if exception else status,
                      "technical_error": exception,
                      "image_calls": draws, "policy_turns": turns, "reward": None if exception else reward,
                      "quality_reward": None if exception else quality_reward,
                      "validity": validity.as_dict(), "category_id": extra_info.get("category_id"),
                      "judgement": judgement, "metrics": metrics}
            if prompt_scoring or prompt_physics or hybrid_quality or image_only:
                record.update(final_draw_prompt=last_caption, final_image_path=str(final_image) if final_image else None,
                              final_image_sha256=last_hash,
                              final_draw_prompt_sha256=hashlib.sha256(last_caption.encode()).hexdigest() if last_caption else None)
            if image_only:
                record["reward_components"] = judgement["reward_components"] if judgement else None
            elif hybrid_quality:
                prompt_judgement = judgement["prompt_judgement"] if judgement else {}
                record.update(
                    prompt_physics_score=judgement["prompt_physics_score"] if judgement else None,
                    image_quality_score=judgement["image_quality_score"] if judgement else None,
                    reward_components=judgement["reward_components"] if judgement else None,
                    prompt_physics_rule_version=prompt_judgement.get("rubric", {}).get("version"),
                    prompt_physics_rubric_sha256=prompt_judgement.get("rubric_sha256"))
            elif prompt_physics:
                record.update(
                    prompt_physics_reward=None if exception else quality_reward,
                    prompt_physics_rule_version=(judgement.get("rubric", {}).get("version") if judgement else None),
                    prompt_physics_rubric_sha256=judgement.get("rubric_sha256") if judgement else None)
                if prompt_binary:
                    record["prompt_physics_score"] = judgement["score"] if judgement else None
                else:
                    record.update(correctness_score=judgement["correctness_score"] if judgement else None,
                                  completeness_score=judgement["completeness_score"] if judgement else None)
            if self.settings.get("save_tokens", True):
                record.update(prompt_ids=initial_ids, response_ids=runtime_ids[len(initial_ids):],
                              response_mask=mask, response_logprobs=logprobs)
            (artifact / "rollout.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if draw is not None:
                await draw.close()
            if judge is not None:
                await judge.close()
