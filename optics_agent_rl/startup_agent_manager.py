"""Portable manager: no machine-specific compilation-cache startup hooks."""
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

class StartupAgentLoopManager(AgentLoopManagerTQ):
    pass
