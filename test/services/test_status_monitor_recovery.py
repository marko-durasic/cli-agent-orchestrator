"""Opt-in recovery bootstraps status from a pinned, live viewport only."""

from types import SimpleNamespace

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import status_monitor as module


class SnapshotProvider:
    """Snapshot-safe detector; real provider parsing is covered by provider tests."""

    session_name = "session"
    window_name = "window"
    supports_screen_detection = True
    supports_direct_status_probe = False

    def __init__(self):
        self.screen_inputs = []
        self.raw_inputs = []
        self.after_detect = None

    def get_status_from_screen(self, lines):
        self.screen_inputs.append(lines)
        result = TerminalStatus(lines[-1])
        if self.after_detect:
            self.after_detect()
        return result

    def get_status(self, output):
        self.raw_inputs.append(output)
        return TerminalStatus(output)

    def initialize(self):
        raise AssertionError("recovery must never initialize a provider")


@pytest.fixture
def recovery(monkeypatch):
    monitor = module.StatusMonitor()
    provider = SnapshotProvider()
    state = SimpleNamespace(
        monitor=monitor,
        provider=provider,
        providers={"t1": provider},
        now=100.0,
        screen="idle",
        captures=[],
        published=[],
        valid=True,
        identity_error=False,
        during_capture=None,
    )

    def get_provider(terminal_id):
        return state.providers.get(terminal_id)

    def capture():
        state.captures.append(state.screen)
        if state.during_capture:
            state.during_capture()
        return state.screen

    def identity_check():
        if state.identity_error:
            raise RuntimeError("identity check failed")
        return state.valid

    state.capture = capture
    state.identity_check = identity_check
    state.bootstrap = lambda: monitor.bootstrap_status(
        "t1", capture=state.capture, identity_check=state.identity_check
    )
    monkeypatch.setattr(
        module,
        "provider_manager",
        SimpleNamespace(_providers=state.providers, get_provider=get_provider),
    )
    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(module, "CAO_PYTE_STATUS", False)
    monkeypatch.setattr(module, "get_server_settings", lambda: {"state_buffer_max": 32768})
    monkeypatch.setattr(
        module.bus, "publish", lambda topic, data: state.published.append((topic, data))
    )
    return state


def advance(state, amount=module.STALE_PROCESSING_CAPTURE_INTERVAL_S):
    state.now += amount


def test_idle_needs_two_live_samples_and_publishes_only_status(recovery):
    state = recovery
    assert state.bootstrap() == TerminalStatus.UNKNOWN
    assert state.monitor.get_buffer("t1") == ""
    assert state.published == []
    advance(state)
    assert state.bootstrap() == TerminalStatus.IDLE
    assert state.monitor._last_status["t1"] == TerminalStatus.IDLE
    assert state.provider.screen_inputs == [["idle"], ["idle"]]
    assert state.provider.raw_inputs == []
    assert state.monitor.get_buffer("t1") == ""
    assert state.published == [("terminal.t1.status", {"status": "idle"})]


@pytest.mark.parametrize("status", list(TerminalStatus))
def test_recognized_states_are_never_misreported_as_ready(recovery, status):
    recovery.screen = status.value
    result = recovery.bootstrap()
    assert result == (
        TerminalStatus.PROCESSING if status == TerminalStatus.PROCESSING else TerminalStatus.UNKNOWN
    )
    advance(recovery)
    assert recovery.bootstrap() == status


@pytest.mark.parametrize("status", [s for s in TerminalStatus if s != TerminalStatus.UNKNOWN])
def test_known_pipeline_status_is_unchanged_without_any_capture(recovery, status):
    recovery.monitor._last_status["t1"] = status
    recovery.screen = "idle"
    assert recovery.bootstrap() == status
    assert recovery.captures == []
    assert recovery.published == []


def test_unrecognized_frame_fails_closed(recovery):
    recovery.screen = "some unrecognized rendered frame"
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert recovery.published == []


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_unsupported_provider_never_captures(recovery, missing):
    if missing:
        recovery.providers.clear()
    else:
        recovery.provider.supports_screen_detection = False
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert recovery.captures == []
    assert recovery.published == []


def test_direct_snapshot_provider_uses_raw_detector(recovery):
    recovery.provider.supports_screen_detection = False
    recovery.provider.supports_direct_status_probe = True
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE
    assert recovery.provider.screen_inputs == []
    assert recovery.provider.raw_inputs == ["idle", "idle"]


def test_capture_rate_limit_and_candidate_ttl_are_preserved(recovery):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery, module.STALE_PROCESSING_CAPTURE_INTERVAL_S - 0.01)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert len(recovery.captures) == 1
    advance(recovery, module.STALE_PROCESSING_CONFIRM_TTL_S + 0.01)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert len(recovery.captures) == 2
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_new_turn_or_real_output_between_samples_requires_new_pair(recovery, boundary):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    if boundary == "input":
        recovery.monitor.notify_input_sent("t1")
    else:
        recovery.monitor._process_chunk("t1", "unknown")
        assert recovery.bootstrap() == TerminalStatus.UNKNOWN
        assert len(recovery.captures) == 1  # A fresh output burst keeps the quiet gate shut.
        advance(recovery, module.STALE_PROCESSING_BUFFER_QUIET_S)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE


@pytest.mark.parametrize("boundary", ["input", "output", "clear", "reset"])
@pytest.mark.parametrize("sample", [1, 2])
def test_boundary_during_capture_never_seeds_or_applies_stale_ready(recovery, boundary, sample):
    if sample == 2:
        assert recovery.bootstrap() == TerminalStatus.UNKNOWN
        advance(recovery)

    def change():
        if boundary == "input":
            recovery.monitor.notify_input_sent("t1")
        elif boundary == "output":
            recovery.monitor._process_chunk("t1", "unknown")
        elif boundary == "clear":
            recovery.monitor.clear_terminal("t1")
        else:
            recovery.monitor.reset_buffer("t1")

    recovery.during_capture = change
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert "t1" not in recovery.monitor._pending_stale_capture
    assert not any(data["status"] == "idle" for _, data in recovery.published)
    recovery.during_capture = None
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN


@pytest.mark.parametrize("change", ["replace", "delete", "move"])
def test_provider_identity_change_during_capture_discards_sample(recovery, change):
    def mutate():
        if change == "replace":
            recovery.providers["t1"] = SnapshotProvider()
        elif change == "delete":
            recovery.providers.clear()
        else:
            recovery.provider.window_name = "replacement-window"

    recovery.during_capture = mutate
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert "t1" not in recovery.monitor._pending_stale_capture
    assert recovery.published == []


def test_provider_replacement_between_samples_cannot_confirm_old_identity(recovery):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    recovery.providers["t1"] = SnapshotProvider()
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE


def test_new_pinned_capture_attempt_cannot_confirm_previous_pane(recovery):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    old_capture = recovery.capture
    recovery.capture = lambda: old_capture()
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE


@pytest.mark.parametrize("when", ["before", "during_capture", "after_detection"])
def test_live_identity_failure_never_seeds_or_publishes_ready(recovery, when):
    def invalidate():
        recovery.valid = False

    if when == "before":
        recovery.valid = False
    elif when == "during_capture":
        recovery.during_capture = invalidate
    else:
        recovery.provider.after_detect = invalidate
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert "t1" not in recovery.monitor._pending_stale_capture
    assert recovery.published == []
    if when == "before":
        assert recovery.captures == []


def test_pipeline_becoming_known_during_capture_wins(recovery):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    recovery.during_capture = lambda: recovery.monitor._apply_detection(
        "t1", TerminalStatus.PROCESSING
    )
    assert recovery.bootstrap() == TerminalStatus.PROCESSING
    assert recovery.published == [("terminal.t1.status", {"status": "processing"})]


def test_input_after_confirmation_before_apply_discards_ready(recovery, monkeypatch):
    real_capture = recovery.monitor._fresh_capture_pane_status

    def capture_then_input(*args, **kwargs):
        detected = real_capture(*args, **kwargs)
        if detected == TerminalStatus.IDLE:
            recovery.monitor.notify_input_sent("t1")
        return detected

    monkeypatch.setattr(recovery.monitor, "_fresh_capture_pane_status", capture_then_input)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert recovery.monitor._allow_processing_revert["t1"] is True
    assert recovery.published == []


def test_identity_failure_after_confirmation_before_apply_discards_ready(recovery, monkeypatch):
    real_capture = recovery.monitor._fresh_capture_pane_status

    def capture_then_invalidate(*args, **kwargs):
        detected = real_capture(*args, **kwargs)
        if detected == TerminalStatus.IDLE:
            recovery.valid = False
        return detected

    monkeypatch.setattr(recovery.monitor, "_fresh_capture_pane_status", capture_then_invalidate)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert recovery.published == []


def test_unknown_polling_does_not_probe_or_bootstrap(recovery, monkeypatch):
    from cli_agent_orchestrator.backends import registry

    backend = SimpleNamespace(supports_event_inbox=lambda: False)
    monkeypatch.setattr(registry, "get_backend", lambda: backend)
    for _ in range(4):
        assert recovery.monitor.get_status("t1") == TerminalStatus.UNKNOWN
        advance(recovery)
    assert recovery.captures == []
    assert recovery.provider.screen_inputs == []
    assert recovery.published == []


@pytest.mark.parametrize(
    "failure", ["identity_false", "identity_error", "capture_error", "empty", "detection_error"]
)
def test_failed_sample_breaks_recovery_confirmation_pair(recovery, failure):
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)

    def fail():
        raise RuntimeError("target no longer observable")

    if failure == "identity_false":
        recovery.valid = False
    elif failure == "identity_error":
        recovery.identity_error = True
    elif failure == "capture_error":
        recovery.during_capture = fail
    elif failure == "empty":
        recovery.screen = ""
    else:
        recovery.screen = "not a recognized screen"
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    assert "t1" not in recovery.monitor._pending_stale_capture
    recovery.valid = True
    recovery.during_capture = None
    recovery.screen = "idle"
    recovery.identity_error = False
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.UNKNOWN
    advance(recovery)
    assert recovery.bootstrap() == TerminalStatus.IDLE
