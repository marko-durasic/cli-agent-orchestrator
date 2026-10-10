"""Recovery I/O stays pinned to one live existing pane."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from cli_agent_orchestrator.clients.tmux import TmuxClient


@pytest.fixture
def target():
    client = TmuxClient.__new__(TmuxClient)
    pane = MagicMock()
    pane.pane_id, pane.pane_pid, pane.pane_dead, pane.pane_pipe = "%3", "456", "0", "0"
    pane.cmd.return_value = SimpleNamespace(stdout=["visible", "viewport"], stderr=[])

    def command(name, *args):
        if name == "display-message":
            value = pane.pane_dead if args[-1] == "#{pane_dead}" else pane.pane_pipe
            return SimpleNamespace(stdout=[value], stderr=[])
        return pane.cmd.return_value

    pane.cmd.side_effect = command
    window = SimpleNamespace(window_id="@2", panes=[pane])
    session = SimpleNamespace(session_id="$1")
    client._find_session = MagicMock(return_value=session)
    client._find_window = MagicMock(return_value=window)
    return client, session, window, pane


def test_pin_live_single_pane_identity(target):
    assert target[0].pin_output_target("cao-one", "worker") == ("$1", "@2", "%3", "456")
    assert all(c.args[0] == "display-message" for c in target[3].cmd.call_args_list)


def test_capture_reads_only_pinned_visible_viewport(target):
    client, _, _, pane = target
    assert (
        client.capture_output_target("cao-one", "worker", ("$1", "@2", "%3", "456"))
        == "visible\nviewport"
    )
    assert [c for c in pane.cmd.call_args_list if c.args[0] != "display-message"] == [
        call("capture-pane", "-p", "-S", "0")
    ]


def test_rearm_is_stop_then_start_on_same_pane(target):
    client, _, _, pane = target
    client.rearm_output_target("cao-one", "worker", ("$1", "@2", "%3", "456"), "/tmp/a b.fifo")
    assert [c for c in pane.cmd.call_args_list if c.args[0] != "display-message"] == [
        call("pipe-pane"),
        call("pipe-pane", "-o", "cat >> '/tmp/a b.fifo'"),
    ]


@pytest.mark.parametrize(
    "operation", ["capture_output_target", "rearm_output_target", "output_target_is_piped"]
)
def test_changed_pane_or_process_refuses_io(target, operation):
    client, _, _, pane = target
    args = ["cao-one", "worker", ("$1", "@2", "%3", "789")]
    if operation == "rearm_output_target":
        args.append("/tmp/known.fifo")
    with pytest.raises(ValueError, match="identity"):
        getattr(client, operation)(*args)
    assert all(c.args[0] == "display-message" for c in pane.cmd.call_args_list)


@pytest.mark.parametrize("dead,panes", [("1", 1), ("0", 2), ("0", 0)])
def test_dead_or_ambiguous_pane_refuses_recovery(target, dead, panes):
    client, _, window, pane = target
    pane.pane_dead = dead
    window.panes = [pane] * panes
    with pytest.raises(ValueError):
        client.pin_output_target("cao-one", "worker")
    assert all(c.args[0] == "display-message" for c in pane.cmd.call_args_list)


def test_absent_session_refuses_recovery(target):
    target[0]._find_session.return_value = None
    with pytest.raises(ValueError):
        target[0].pin_output_target("cao-one", "worker")


def test_pipe_state_is_checked_without_mutation(target):
    client, _, _, pane = target
    assert client.output_target_is_piped("cao-one", "worker", ("$1", "@2", "%3", "456")) is False
    pane.pane_pipe = "1"
    assert client.output_target_is_piped("cao-one", "worker", ("$1", "@2", "%3", "456")) is True
    pane.pane_pipe = "unknown"
    with pytest.raises(ValueError):
        client.output_target_is_piped("cao-one", "worker", ("$1", "@2", "%3", "456"))


def test_failed_stop_never_starts_new_forwarder(target):
    client, _, _, pane = target
    pane.cmd.return_value.stderr = ["pane vanished"]
    with pytest.raises(RuntimeError):
        client.rearm_output_target(
            "cao-one", "worker", ("$1", "@2", "%3", "456"), "/tmp/known.fifo"
        )
    assert [c for c in pane.cmd.call_args_list if c.args[0] != "display-message"] == [
        call("pipe-pane")
    ]


def test_real_libtmux_pane_uses_explicit_tmux_flag_formats(target):
    """Installed libtmux Pane has neither pane_dead nor pane_pipe attributes."""
    from libtmux.pane import Pane

    client, _, window, _ = target
    pane = Pane(server=MagicMock(), pane_id="%3", pane_pid="456")

    def command(name, *args):
        if name == "display-message":
            return SimpleNamespace(stdout=["0"], stderr=[])
        return SimpleNamespace(stdout=["visible"], stderr=[])

    pane.cmd = MagicMock(side_effect=command)
    window.panes = [pane]
    identity = client.pin_output_target("cao-one", "worker")
    assert identity == ("$1", "@2", "%3", "456")
    assert client.output_target_is_piped("cao-one", "worker", identity) is False
    assert client.capture_output_target("cao-one", "worker", identity) == "visible"
    assert call("display-message", "-p", "#{pane_dead}") in pane.cmd.call_args_list
    assert call("display-message", "-p", "#{pane_pipe}") in pane.cmd.call_args_list
