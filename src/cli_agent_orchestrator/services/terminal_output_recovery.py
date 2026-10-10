"""Opt-in monitoring repair for one surviving terminal after a server reload.

No startup sweep, provider initialization, input, history replay, status override,
or inbox bypass. Successful status detection may wake ordinary inbox delivery.
"""

import re
import time

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.backends.tmux_backend import TmuxBackend
from cli_agent_orchestrator.clients.database import get_terminal_metadata
from cli_agent_orchestrator.constants import FIFO_DIR
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.services.fifo_reader import fifo_manager
from cli_agent_orchestrator.services.session_lock import session_lifecycle_lock
from cli_agent_orchestrator.services.status_monitor import (
    STALE_PROCESSING_CAPTURE_INTERVAL_S,
    status_monitor,
)


class OutputRecoveryConflict(RuntimeError):
    """The requested terminal cannot safely be repaired in place."""


def _registration(metadata):
    if metadata is None:
        return None
    return tuple(
        metadata.get(key)
        for key in (
            "id",
            "tmux_session",
            "tmux_window",
            "provider",
            "agent_profile",
            "engine",
            "shell_command",
            "allowed_tools",
        )
    )


def recover_output(terminal_id: str, *, expected_session: str, expected_window: str) -> dict:
    """Reconnect only an explicitly selected existing tmux terminal's monitoring."""
    if not re.fullmatch(r"[a-f0-9]{8}", terminal_id):
        raise ValueError("Invalid terminal ID")
    with session_lifecycle_lock(expected_session):
        metadata = get_terminal_metadata(terminal_id)
        if metadata is None:
            raise ValueError("Terminal not found")
        if (metadata["tmux_session"], metadata["tmux_window"]) != (
            expected_session,
            expected_window,
        ):
            raise OutputRecoveryConflict("Terminal registration does not match the request")
        backend = get_backend()
        if not isinstance(backend, TmuxBackend):
            raise OutputRecoveryConflict("Targeted output recovery requires the tmux backend")
        registration = _registration(metadata)
        target = backend.pin_output_target(expected_session, expected_window)

        def identity_check():
            return (
                get_backend() is backend
                and _registration(get_terminal_metadata(terminal_id)) == registration
                and backend.pin_output_target(expected_session, expected_window) == target
            )

        def require_identity():
            if not identity_check():
                raise OutputRecoveryConflict("Terminal identity changed during output recovery")

        def capture():
            require_identity()
            return backend.capture_output_target(expected_session, expected_window, target)

        fifo_path = FIFO_DIR / f"{terminal_id}.fifo"

        def rearm():
            require_identity()
            backend.rearm_output_target(expected_session, expected_window, target, str(fifo_path))

        require_identity()
        with fifo_manager.recovery_reader(
            terminal_id, pane_probe=capture, rearm=rearm
        ) as reader_recovered:
            require_identity()
            pipe_rearmed = reader_recovered or not backend.output_target_is_piped(
                expected_session, expected_window, target
            )
            if pipe_rearmed:
                rearm()
            # One first observation can be skipped by the existing quiet/rate
            # gate. Keep one bounded extra observation in THIS pinned attempt,
            # so immediate retries do not continually discard an unconfirmed pair.
            for attempt in range(3):
                require_identity()
                detected = status_monitor.bootstrap_status(
                    terminal_id, capture=capture, identity_check=identity_check
                )
                if detected != TerminalStatus.UNKNOWN:
                    break
                if attempt < 2:
                    time.sleep(STALE_PROCESSING_CAPTURE_INTERVAL_S)
            require_identity()
            return {
                "terminal_id": terminal_id,
                "reader_recovered": reader_recovered,
                "pipe_rearmed": pipe_rearmed,
                "status": detected.value,
                "pane_id": target[2],
            }
