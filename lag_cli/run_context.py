from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class GeminiUsageTotals:
    n_call_count: int = 0
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    n_total_tokens: int = 0

    @staticmethod
    def _as_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    def add_usage_metadata(self, usage_metadata: dict[str, object] | None) -> None:
        self.n_call_count += 1
        if not isinstance(usage_metadata, dict):
            return
        self.n_prompt_tokens += self._as_int(usage_metadata.get("promptTokenCount"))
        self.n_output_tokens += self._as_int(usage_metadata.get("candidatesTokenCount"))
        self.n_total_tokens += self._as_int(usage_metadata.get("totalTokenCount"))

    def to_dict(self) -> dict[str, int]:
        return {
            "n_call_count": self.n_call_count,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_output_tokens": self.n_output_tokens,
            "n_total_tokens": self.n_total_tokens,
        }


@dataclass(slots=True)
class RunContext:
    run_uid: str
    mode: str
    prompt: str
    model: str
    track_outputs: bool = True
    gemini_usage: GeminiUsageTotals = field(default_factory=GeminiUsageTotals)


def create_run_uid(lamindb_run_uid: str | None = None) -> str:
    if lamindb_run_uid:
        return str(lamindb_run_uid)
    return f"agent-{uuid4().hex[:12]}"
