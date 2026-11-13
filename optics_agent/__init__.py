"""Independent, auditable optical image agent. No benchmark/runtime imports."""

__version__ = "0.3.0"


def __getattr__(name):
    if name in {"AgentRunner", "RunConfig"}:
        from .runner import AgentRunner, RunConfig
        return {"AgentRunner": AgentRunner, "RunConfig": RunConfig}[name]
    raise AttributeError(name)
