"""Runtime configuration, read once from the environment.

Every knob has a safe default and is clamped, so a typo in a notebook cell
degrades to a sane value instead of crashing a 9-hour run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # --- model server -------------------------------------------------------
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "flashnext"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    max_output_tokens: int = 16 * 1024
    request_timeout_s: float = 1800.0
    # Name of the assistant-message field that carries chain-of-thought back to
    # the server. SGLang's qwen3 template reads ``reasoning_content``.
    reasoning_field: str = "reasoning_content"

    # --- efficiency pressure --------------------------------------------------
    # Later levels cost far more actions than early ones (mechanics compound),
    # and the scoring formula weights later levels more heavily too, so a model
    # that keeps exploring past a reasonable budget loses score twice over.
    # These thresholds add an escalating in-context reminder once a level's
    # action count crosses them, on top of the one-time system-prompt warning.
    efficiency_warn_actions: int = 30
    efficiency_critical_actions: int = 60

    # --- context summarization (opt-in) --------------------------------------
    # Off by default: this costs one extra, bounded model call each time context
    # is about to be evicted, trading a small amount of turn throughput for a
    # real compressed account of what was learned (vs. the static memory
    # template + whatever `notes` happened to catch). Unvalidated against a live
    # run; enable deliberately, not as part of a routine change.
    use_llm_summary: bool = False
    summary_max_tokens: int = 700

    # --- forced commitment ----------------------------------------------------
    # The trace evidence (not just a hunch): levels that cost the most action
    # budget without ever dying weren't failed plans, they were many consecutive
    # small calls (1-2 actions each) that never committed to a plan at all --
    # one run spent 80 model calls on 175 actions on a single level and never
    # advanced. A text reminder about the scoring formula did not change this
    # behavior. So: once a level racks up `probe_streak_limit` consecutive calls
    # each under `min_commit_batch` actions, bare action() calls are refused
    # (free) until the model commits a real plan via commit_plan([...]).
    min_commit_batch: int = 5
    probe_streak_limit: int = 6

    # --- context ------------------------------------------------------------
    context_budget_tokens: int = 118 * 1024
    context_drain_target_tokens: int = 58 * 1024
    chars_per_token: float = 3.0
    image_tokens: int = 1600
    keep_images: int = 1
    max_history_turns: int = 150
    history_turn_drain: int = 30
    image_scale: int = 10

    # --- tool sandbox -------------------------------------------------------
    tool_timeout_s: float = 45.0
    tool_output_tokens: int = 3072
    max_actions_per_call: int = 64
    # Repeat/no-op guards stay off on the first levels, where free experimentation is needed.
    guards_from_level: int = 1

    # --- scheduling ---------------------------------------------------------
    slots: int = 10
    yield_after_generated_tokens: int = 2048
    max_runtime_s_per_game: float = 532 * 60.0
    tail_fade_fraction: float = 0.2

    # --- behaviour ----------------------------------------------------------
    expose_undo: bool = True
    max_consecutive_model_failures: int = 6
    max_noop_streak: int = 12
    warmup_after_s: float = 240.0
    status_every_s: float = 600.0
    debug_dir: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        budget = _env_int("ARCD_CONTEXT_BUDGET", 118 * 1024, 8 * 1024, 1024 * 1024)
        drain = _env_int("ARCD_CONTEXT_DRAIN_TARGET", 58 * 1024, 4 * 1024, budget - 1024)
        return cls(
            base_url=_env_str("ARCD_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            model=_env_str("ARCD_MODEL", "flashnext"),
            api_key=_env_str("ARCD_API_KEY", "EMPTY"),
            temperature=_env_float("ARCD_TEMPERATURE", 0.7, 0.0, 2.0),
            top_p=_env_float("ARCD_TOP_P", 0.95, 0.01, 1.0),
            top_k=_env_int("ARCD_TOP_K", 20, 1, 1000),
            max_output_tokens=_env_int("ARCD_MAX_OUTPUT", 16 * 1024, 256, 64 * 1024),
            request_timeout_s=_env_float("ARCD_REQUEST_TIMEOUT", 1800.0, 5.0, 7200.0),
            reasoning_field=_env_str("ARCD_REASONING_FIELD", "reasoning_content"),
            efficiency_warn_actions=_env_int("ARCD_EFFICIENCY_WARN_ACTIONS", 30, 1, 10000),
            efficiency_critical_actions=_env_int("ARCD_EFFICIENCY_CRITICAL_ACTIONS", 60, 1, 10000),
            use_llm_summary=_env_bool("ARCD_USE_LLM_SUMMARY", False),
            summary_max_tokens=_env_int("ARCD_SUMMARY_MAX_TOKENS", 700, 64, 8192),
            min_commit_batch=_env_int("ARCD_MIN_COMMIT_BATCH", 5, 1, 1000),
            probe_streak_limit=_env_int("ARCD_PROBE_STREAK_LIMIT", 6, 1, 1000),
            context_budget_tokens=budget,
            context_drain_target_tokens=drain,
            chars_per_token=_env_float("ARCD_CHARS_PER_TOKEN", 3.0, 1.0, 8.0),
            image_tokens=_env_int("ARCD_IMAGE_TOKENS", 1600, 0, 20000),
            keep_images=_env_int("ARCD_KEEP_IMAGES", 1, 0, 8),
            image_scale=_env_int("ARCD_IMAGE_SCALE", 10, 1, 16),
            tool_timeout_s=_env_float("ARCD_TOOL_TIMEOUT", 45.0, 1.0, 3600.0),
            tool_output_tokens=_env_int("ARCD_TOOL_OUTPUT_TOKENS", 3072, 128, 32768),
            max_actions_per_call=_env_int("ARCD_MAX_ACTIONS_PER_CALL", 64, 1, 1024),
            guards_from_level=_env_int("ARCD_GUARDS_FROM_LEVEL", 1, 0, 100),
            max_history_turns=_env_int("ARCD_MAX_HISTORY_TURNS", 150, 4, 100000),
            history_turn_drain=_env_int("ARCD_HISTORY_TURN_DRAIN", 30, 1, 100000),
            slots=_env_int("ARCD_SLOTS", 10, 1, 256),
            yield_after_generated_tokens=_env_int("ARCD_YIELD_TOKENS", 2048, 256, 65536),
            max_runtime_s_per_game=_env_float("ARCD_MAX_RUNTIME_PER_GAME", 532 * 60.0, 10.0, 7 * 24 * 3600.0),
            tail_fade_fraction=_env_float("ARCD_TAIL_FADE", 0.2, 0.0, 0.9),
            expose_undo=_env_bool("ARCD_EXPOSE_UNDO", True),
            max_consecutive_model_failures=_env_int("ARCD_MAX_MODEL_FAILURES", 6, 1, 100),
            max_noop_streak=_env_int("ARCD_MAX_NOOP_STREAK", 12, 2, 1000),
            warmup_after_s=_env_float("ARCD_WARMUP_AFTER", 240.0, 1.0, 3600.0),
            status_every_s=_env_float("ARCD_STATUS_EVERY", 600.0, 1.0, 7200.0),
            debug_dir=_env_str("ARCD_DEBUG_DIR", ""),
        )
