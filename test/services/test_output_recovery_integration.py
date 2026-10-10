"""Combine real FIFO readiness, Claude viewport detection and inbox gate."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli_agent_orchestrator.backends.tmux_backend import TmuxBackend
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider
from cli_agent_orchestrator.services import fifo_reader, inbox_service, status_monitor
from cli_agent_orchestrator.services import terminal_output_recovery as recovery


@pytest.mark.parametrize(
    "screen,expected,deliveries",
    [
        ("────────────────────\n❯\n────────────────────", TerminalStatus.IDLE, 1),
        (
            "✻ Thinking…\n────────────────────\n❯\n────────────────────",
            TerminalStatus.PROCESSING,
            0,
        ),
        ("unrecognized current viewport", TerminalStatus.UNKNOWN, 0),
    ],
)
def test_reload_recovery_preserves_fifo_and_only_truth_releases_inbox(
    monkeypatch, tmp_path, screen, expected, deliveries
):
    tid = "abcdef12"
    metadata = {
        "id": tid,
        "tmux_session": "cao-existing",
        "tmux_window": "worker-abcdef12",
        "provider": "claude_code",
        "agent_profile": "developer",
    }
    path = tmp_path / f"{tid}.fifo"
    os.mkfifo(path)
    inode = path.stat().st_ino
    log = tmp_path / f"{tid}.log"
    log.write_text("existing output must stay intact\n")
    monkeypatch.setattr(fifo_reader, "FIFO_DIR", tmp_path)
    monkeypatch.setattr(recovery, "FIFO_DIR", tmp_path)
    manager = fifo_reader.FifoManager()
    monkeypatch.setattr(manager, "_ensure_watchdog", lambda: None)
    monitor = status_monitor.StatusMonitor()
    provider = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
    provider.session_name, provider.window_name = metadata["tmux_session"], metadata["tmux_window"]
    provider._initialized = False
    providers = {tid: provider}
    monkeypatch.setattr(
        status_monitor,
        "provider_manager",
        SimpleNamespace(_providers=providers, get_provider=lambda t: providers.get(t)),
    )
    monkeypatch.setattr(status_monitor, "STALE_PROCESSING_CAPTURE_INTERVAL_S", 0)
    monkeypatch.setattr(recovery, "STALE_PROCESSING_CAPTURE_INTERVAL_S", 0)
    backend = MagicMock(spec=TmuxBackend)
    backend.pin_output_target.return_value = ("$1", "@2", "%3", "456")
    backend.capture_output_target.return_value = screen
    backend.output_target_is_piped.return_value = True
    monkeypatch.setattr(recovery, "get_backend", lambda: backend)
    monkeypatch.setattr(
        recovery, "get_terminal_metadata", lambda t: metadata.copy() if t == tid else None
    )
    monkeypatch.setattr(recovery, "fifo_manager", manager)
    monkeypatch.setattr(recovery, "status_monitor", monitor)
    events = []
    monkeypatch.setattr(
        status_monitor.bus, "publish", lambda topic, data: events.append((topic, data))
    )
    monkeypatch.setattr(inbox_service, "status_monitor", monitor)
    monkeypatch.setattr(inbox_service, "EAGER_INBOX_DELIVERY", False)
    pending = [SimpleNamespace(id=123, sender_id="sender", message="queued before reload")]
    monkeypatch.setattr(inbox_service, "get_pending_messages", lambda *a, **k: pending[:])
    sent = []
    monkeypatch.setattr(inbox_service, "update_message_status", lambda *a: pending.clear())
    monkeypatch.setattr(
        inbox_service.terminal_service, "send_input", lambda *a, **k: sent.append(a)
    )
    inbox = inbox_service.InboxService()
    try:
        inbox.deliver_pending(tid)
        assert sent == [] and len(pending) == 1
        result = recovery.recover_output(
            tid, expected_session=metadata["tmux_session"], expected_window=metadata["tmux_window"]
        )
        assert result["status"] == expected.value
        assert path.stat().st_ino == inode
        assert log.read_text() == "existing output must stay intact\n"
        assert provider._initialized is False
        assert all(not topic.endswith(".output") for topic, _ in events)
        assert monitor.get_buffer(tid) == ""
        assert (
            backend.capture_output_target.call_count
            == {
                TerminalStatus.PROCESSING: 1,
                TerminalStatus.IDLE: 2,
                TerminalStatus.UNKNOWN: 3,
            }[expected]
        )
        backend.send_keys.assert_not_called()
        backend.send_special_key.assert_not_called()
        backend.kill_window.assert_not_called()
        backend.create_window.assert_not_called()
        # The repair never drains directly. Only the unchanged delivery gate is used.
        assert sent == []
        inbox.deliver_pending(tid)
        assert len(sent) == deliveries
        if not deliveries:
            assert len(pending) == 1
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
    finally:
        manager.stop_reader(tid)
        manager.stop_watchdog()
