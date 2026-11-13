"""Fail closed on incomplete GRPO groups using VERL's custom sampler hook."""
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class StrictOpticalReplayBuffer(ReplayBuffer):
    """Keep official sync sampling; stop on technical failures before padding.

    In pinned VERL V1, a failed prompt group can still contain successful sibling
    trajectories. The default sync buffer permits sampling those partial groups.
    A service outage must not change the requested GRPO group population: after
    bounded client retries we stop the job, retaining existing trainer checkpoints
    and rollout artifacts, instead of treating missing siblings as zero rewards.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("trainer_mode") != "sync":
            raise ValueError("StrictOpticalReplayBuffer supports the configured synchronous GRPO trainer only")
        super().__init__(*args, **kwargs)

    def _terminal_eviction_reasons(self, global_steps, partition_id):
        failed = self.failure_keys[partition_id]
        if failed:
            raise RuntimeError(
                f"Optical RL stopped before sampling {len(failed)} incomplete {partition_id} GRPO group(s). "
                "A rollout or service failed after retries; inspect optics_agent_rl.artifact_dir. "
                "No partial group was admitted. Restore the service and resume the trainer checkpoint."
            )
        return super()._terminal_eviction_reasons(global_steps, partition_id)
