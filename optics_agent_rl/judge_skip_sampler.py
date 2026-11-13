"""Exclude unjudged prompt groups before official VERL constructs training tensors.

OpticalAgentLoop returns [] only for an explicitly skippable judge service error.
Official V1 marks that prompt finished after every sibling settles. A finished
group missing any of its expected sessions is therefore excluded in its entirety.
No zero reward, fabricated sample, or all-masked optimizer step is needed.
"""
from __future__ import annotations

import json

from optics_agent_rl.replay_buffer import StrictOpticalReplayBuffer


class JudgeSkipReplayBuffer(StrictOpticalReplayBuffer):
    """Keep full on-policy batches by replacing skipped groups with fresh prompts.

    The official refill callback advances the training dataloader; it never retries
    the failed image or directly reissues its prompt. At dataset end VERL cycles its
    dataloader, retaining its configured optimizer-step count. A bounded number of
    skipped groups per batch prevents an unavailable judge from looping forever.
    All actual rollout exceptions still fail closed through StrictOpticalReplayBuffer.
    In-run validation is deliberately unsupported by this opt-in training sampler.
    """

    def __init__(self, *args, **kwargs):
        settings = kwargs.get("sampler_kwargs") or {}
        self.expected_group_size = settings.get("expected_group_size")
        self.max_skipped_groups_per_batch = settings.get("max_skipped_groups_per_batch", 80)
        if type(self.expected_group_size) is not int or self.expected_group_size < 2:
            raise ValueError("JudgeSkipReplayBuffer requires expected_group_size >= 2")
        if type(self.max_skipped_groups_per_batch) is not int or self.max_skipped_groups_per_batch < 1:
            raise ValueError("max_skipped_groups_per_batch must be a positive integer")
        if kwargs.get("filter_groups_metric") is not None:
            raise ValueError("Judge skipping cannot be combined with DAPO reward filtering")
        # These are not forwarded to custom samplers by official PPOTrainer.
        kwargs.update(sync_refill_failed_groups=True, gen_batch_size=1)
        super().__init__(*args, **kwargs)
        self._skipped_groups = self._skipped_samples = self._discarded_samples = 0

    def _incomplete_groups(self, partition_id):
        sessions = {uid: set() for uid in self.finished_keys[partition_id]}
        for key in self.partitions[partition_id]:
            uid, session, index = key.rsplit("_", 2)
            if uid not in sessions:
                continue
            session, index = int(session), int(index)
            if index != 0 or session < 0 or session >= self.expected_group_size or session in sessions[uid]:
                raise RuntimeError("Judge skipping requires exactly one optical trajectory per rollout session")
            sessions[uid].add(session)
        return {uid: self.expected_group_size - len(found)
                for uid, found in sessions.items() if len(found) != self.expected_group_size}

    def _terminal_eviction_reasons(self, global_steps, partition_id):
        # A configuration, draw, policy or programming exception is never
        # converted into a retry. Only successful empty outputs are skippable.
        reasons = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id != "train":
            raise ValueError("JudgeSkipReplayBuffer supports training only; disable in-run validation")
        stale, filtered, failed, counts = reasons
        return stale, filtered, failed | set(self._incomplete_groups(partition_id)), counts

    def _evict_terminal_groups(self, global_steps, partition_id, eviction_reasons):
        missing = self._incomplete_groups(partition_id)
        skipped = eviction_reasons[2]
        if skipped:
            self._skipped_groups += len(skipped)
            self._skipped_samples += sum(missing[uid] for uid in skipped)
            self._discarded_samples += len(skipped) * self.expected_group_size
            report = {"event": "judge_groups_skipped", "global_steps": global_steps,
                      "skipped_groups": self._skipped_groups, "failed_judge_samples": self._skipped_samples,
                      "discarded_samples": self._discarded_samples,
                      "max_skipped_groups_per_batch": self.max_skipped_groups_per_batch}
            print(json.dumps(report), flush=True)
            if self._skipped_groups > self.max_skipped_groups_per_batch:
                raise RuntimeError(
                    f"Judge unavailable: skipped {self._skipped_groups} prompt groups while assembling one batch "
                    f"(limit {self.max_skipped_groups_per_batch}). No incomplete group or placeholder reward "
                    "entered training and this batch made no optimizer update. Inspect judge rollout artifacts."
                )
        return super()._evict_terminal_groups(global_steps, partition_id, eviction_reasons)

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id != "train":
            raise ValueError("JudgeSkipReplayBuffer supports training only; disable in-run validation")
        self._skipped_groups = self._skipped_samples = self._discarded_samples = 0
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        if len(batch.keys) != batch_size * self.expected_group_size:
            raise RuntimeError("Judge skipping admitted an unexpected distributed batch shape")
        metrics.update({"training/judge_skip/groups": self._skipped_groups,
                        "training/judge_skip/failed_samples": self._skipped_samples,
                        "training/judge_skip/discarded_samples": self._discarded_samples,
                        "training/judge_skip/admitted_groups": batch_size,
                        "training/judge_skip/admitted_samples": len(batch.keys)})
        return batch, metrics
