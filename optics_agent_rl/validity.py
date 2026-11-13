"""Policy execution validity, independent of image quality and service health."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionValidity:
    """Charge recoverable violations once; fatal violations take precedence.

    Service/configuration errors must be raised by the caller instead of being
    registered here. This object only describes actions sampled from the policy.
    """

    recoverable_errors: list[str] = field(default_factory=list)
    fatal_error: str | None = None

    def recoverable(self, reason: str) -> None:
        self.recoverable_errors.append(reason)

    def fatal(self, reason: str) -> None:
        self.fatal_error = reason

    @property
    def reward(self) -> float:
        if self.fatal_error is not None:
            return -1.0
        return -0.2 if self.recoverable_errors else 0.0

    def as_dict(self) -> dict:
        return {"reward": self.reward, "recoverable_errors": list(self.recoverable_errors),
                "fatal_error": self.fatal_error}
