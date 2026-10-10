"""Targeted recovery must never recreate or drive an existing terminal."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from cli_agent_orchestrator.backends.tmux_backend import TmuxBackend
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services import terminal_output_recovery as recovery


@pytest.fixture
def state(monkeypatch):
    metadata = {
        "id": "abcdef12",
        "tmux_session": "cao-existing",
        "tmux_window": "worker-abcdef12",
        "provider": "claude_code",
        "agent_profile": "developer",
    }
    backend = MagicMock(spec=TmuxBackend)
    backend.pin_output_target.return_value = ("$1", "@2", "%3", "456")
    backend.output_target_is_piped.return_value = False
    backend.capture_output_target.return_value = "current visible viewport"
    manager = MagicMock()
    manager.ensure_reader.return_value = True

    @contextmanager
    def recovery_reader(*args, **kwargs):
        yield manager.ensure_reader(*args, **kwargs)

    manager.recovery_reader.side_effect = recovery_reader
    monitor = MagicMock()
    monitor.bootstrap_status.side_effect = [TerminalStatus.UNKNOWN, TerminalStatus.IDLE]
    reads = []

    def get_metadata(terminal_id):
        reads.append(terminal_id)
        return metadata.copy()

    monkeypatch.setattr(recovery, "get_terminal_metadata", get_metadata)
    monkeypatch.setattr(recovery, "get_backend", lambda: backend)
    monkeypatch.setattr(recovery, "fifo_manager", manager)
    monkeypatch.setattr(recovery, "status_monitor", monitor)
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    return metadata, backend, manager, monitor, reads


def recover():
    return recovery.recover_output(
        "abcdef12", expected_session="cao-existing", expected_window="worker-abcdef12"
    )


def test_recovers_only_named_worker_and_confirms_status(state):
    metadata, backend, manager, monitor, reads = state
    before = metadata.copy()
    result = recover()
    assert result == {
        "terminal_id": "abcdef12",
        "reader_recovered": True,
        "pipe_rearmed": True,
        "status": "idle",
        "pane_id": "%3",
    }
    assert set(reads) == {"abcdef12"}
    assert metadata == before
    assert monitor.bootstrap_status.call_count == 2
    first = monitor.bootstrap_status.call_args_list[0].kwargs
    second = monitor.bootstrap_status.call_args_list[1].kwargs
    assert first["capture"] is second["capture"]
    assert first["identity_check"] is second["identity_check"]
    assert first["capture"]() == "current visible viewport"
    backend.rearm_output_target.assert_called_once()
    backend.send_keys.assert_not_called()
    backend.send_special_key.assert_not_called()
    backend.create_window.assert_not_called()
    backend.kill_window.assert_not_called()
    manager.stop_reader.assert_not_called()


def test_healthy_repeat_preserves_pipe_and_reader(state):
    _, backend, manager, monitor, _ = state
    manager.ensure_reader.return_value = False
    backend.output_target_is_piped.return_value = True
    monitor.bootstrap_status.side_effect = [TerminalStatus.COMPLETED]
    result = recover()
    assert not result["reader_recovered"] and not result["pipe_rearmed"]
    backend.rearm_output_target.assert_not_called()


def test_retries_pipe_failure_even_when_reader_exists(state):
    _, backend, manager, _, _ = state
    manager.ensure_reader.return_value = False
    assert recover()["pipe_rearmed"]
    backend.rearm_output_target.assert_called_once()


@pytest.mark.parametrize(
    "status",
    [TerminalStatus.PROCESSING, TerminalStatus.WAITING_USER_ANSWER, TerminalStatus.UNKNOWN],
)
def test_nonready_verdict_is_preserved(state, status):
    _, _, _, monitor, _ = state
    monitor.bootstrap_status.side_effect = [status, status, status]
    assert recover()["status"] == status.value


def test_wrong_caller_identity_fails_before_any_pipe_change(state):
    state[0]["tmux_window"] = "different-worker"
    with pytest.raises(recovery.OutputRecoveryConflict):
        recover()
    state[2].ensure_reader.assert_not_called()
    state[1].rearm_output_target.assert_not_called()


def test_metadata_missing_is_not_found(state, monkeypatch):
    monkeypatch.setattr(recovery, "get_terminal_metadata", lambda _: None)
    with pytest.raises(ValueError):
        recover()
    state[2].ensure_reader.assert_not_called()


def test_unsupported_backend_does_not_touch_fifo(state, monkeypatch):
    monkeypatch.setattr(recovery, "get_backend", lambda: object())
    with pytest.raises(recovery.OutputRecoveryConflict):
        recover()
    state[2].ensure_reader.assert_not_called()


def test_identity_changes_during_reader_setup_fail_closed(state):
    metadata, backend, manager, monitor, _ = state
    manager.ensure_reader.side_effect = (
        lambda *a, **k: metadata.update(tmux_window="replaced") or True
    )
    with pytest.raises(recovery.OutputRecoveryConflict):
        recover()
    backend.rearm_output_target.assert_not_called()
    monitor.bootstrap_status.assert_not_called()


def test_live_pane_replacement_is_rejected(state):
    _, backend, manager, monitor, _ = state
    backend.pin_output_target.side_effect = [("$1", "@2", "%3", "456"), ("$1", "@9", "%8", "999")]
    with pytest.raises(recovery.OutputRecoveryConflict):
        recover()
    manager.ensure_reader.assert_not_called()
    monitor.bootstrap_status.assert_not_called()


def test_reader_open_failure_does_not_rearm_or_bootstrap(state):
    _, backend, manager, monitor, _ = state
    manager.ensure_reader.side_effect = OSError("cannot open fifo")
    with pytest.raises(OSError):
        recover()
    backend.rearm_output_target.assert_not_called()
    monitor.bootstrap_status.assert_not_called()


def test_recent_output_does_not_starve_same_attempt_confirmation(state, monkeypatch):
    from types import SimpleNamespace

    from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
    from cli_agent_orchestrator.services import status_monitor as status_module

    metadata, backend, _, _, _ = state
    monitor = status_module.StatusMonitor()
    provider = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
    provider.session_name, provider.window_name = metadata["tmux_session"], metadata["tmux_window"]
    providers = {"abcdef12": provider}
    monkeypatch.setattr(
        status_module,
        "provider_manager",
        SimpleNamespace(_providers=providers, get_provider=lambda t: providers.get(t)),
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(status_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        recovery.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    monitor._buffer_changed_at["abcdef12"] = clock["now"]
    backend.capture_output_target.return_value = "────────────────────\n❯\n────────────────────"
    monkeypatch.setattr(recovery, "status_monitor", monitor)
    monkeypatch.setattr(status_module.bus, "publish", lambda *args: None)
    assert recover()["status"] == "idle"
    assert backend.capture_output_target.call_count == 2
