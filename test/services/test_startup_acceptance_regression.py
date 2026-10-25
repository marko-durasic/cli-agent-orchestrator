"""Deterministic #2047 regressions; stdlib runner needs no installs or live CAO."""

import asyncio
import itertools
import unittest
from unittest.mock import AsyncMock, patch

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.codex import CodexProvider
from cli_agent_orchestrator.services import terminal_service as ts

TOOL_SCREEN = """› bounded synthetic task
• Ran pwd
  └ /synthetic/worktree
› Ask Codex to do anything
  GPT-6.1-Sol default · ~/synthetic
"""
RATE_SCREEN = """› bounded synthetic task
■ exceeded retry limit, last status: 429 Too Many Requests
› Ask Codex to do anything
  GPT-6.1-Sol default · ~/synthetic
"""
IDLE_SCREEN = """› Ask Codex to do anything
  GPT-6.1-Sol default · ~/synthetic
"""


class StartupAcceptanceRegression(unittest.IsolatedAsyncioTestCase):
    def provider(self):
        return CodexProvider("synthetic", "synthetic-session", "synthetic-window", None)

    async def confirm(self, screen, provider=None):
        provider = provider or self.provider()
        with (
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=False)),
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={
                    "tmux_session": "synthetic-session",
                    "tmux_window": "synthetic-window",
                },
            ),
            patch.object(ts, "get_backend") as backend,
            patch.object(ts, "send_input") as resend,
            patch.object(ts, "send_special_key") as enter,
            patch.object(ts, "_message_visible_in_box", return_value=False),
        ):
            backend.return_value.get_history.return_value = screen
            result = await ts._confirm_worker_started_or_resubmit(
                "synthetic",
                "bounded synthetic task",
                None,
                "supervisor",
                None,
                provider,
                baseline_output=IDLE_SCREEN,
            )
        return result, resend.call_count, enter.call_count

    async def test_tool_receipt_latches_acceptance_despite_stale_idle_cache(self):
        accepted, resends, enters = await self.confirm(TOOL_SCREEN)
        self.assertTrue(accepted)
        self.assertEqual((resends, enters), (0, 0))

    async def test_unknown_startup_never_resends(self):
        accepted, resends, enters = await self.confirm(IDLE_SCREEN)
        self.assertFalse(accepted)
        self.assertEqual((resends, enters), (0, 0))

    async def test_capture_failure_never_resends(self):
        with (
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=False)),
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={"tmux_session": "s", "tmux_window": "w"},
            ),
            patch.object(ts, "get_backend") as backend,
            patch.object(ts, "send_input") as resend,
            patch.object(ts, "send_special_key") as enter,
            patch.object(ts, "_message_visible_in_box", return_value=False),
        ):
            backend.return_value.get_history.side_effect = RuntimeError("synthetic capture failed")
            accepted = await ts._confirm_worker_started_or_resubmit(
                "synthetic", "bounded synthetic task", None, None, None, self.provider()
            )
        self.assertFalse(accepted)
        resend.assert_not_called()
        enter.assert_not_called()

    async def test_rate_limit_is_failure_without_resend(self):
        self.assertEqual(self.provider().get_status(RATE_SCREEN), TerminalStatus.ERROR)
        accepted, resends, enters = await self.confirm(RATE_SCREEN)
        self.assertFalse(accepted)
        self.assertEqual((resends, enters), (0, 0))

    async def test_mcp_tool_receipt_alone_proves_acceptance(self):
        screen = TOOL_SCREEN.replace("• Ran pwd", "• Called synthetic.read({})")
        accepted, resends, enters = await self.confirm(screen)
        self.assertTrue(accepted)
        self.assertEqual((resends, enters), (0, 0))

    async def test_visible_composer_does_not_authorize_enter(self):
        with (
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=False)),
            patch.object(ts, "_message_visible_in_box", return_value=True),
            patch.object(ts, "send_input") as resend,
            patch.object(ts, "send_special_key") as enter,
        ):
            accepted = await ts._confirm_worker_started_or_resubmit(
                "synthetic", "bounded synthetic task", None, None, None, None
            )
        self.assertFalse(accepted)
        resend.assert_not_called()
        enter.assert_not_called()

    async def test_positive_status_requires_delivery_correlation_before_latching(self):
        with (
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=True)),
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={"tmux_session": "s", "tmux_window": "w"},
            ),
            patch.object(ts, "get_backend") as backend,
            patch.object(
                ts, "_worker_is_started_direct", wraps=ts._worker_is_started_direct
            ) as probe,
            patch.object(ts, "send_input") as resend,
        ):
            backend.return_value.get_history.return_value = TOOL_SCREEN
            accepted = await ts._confirm_worker_started_or_resubmit(
                "synthetic",
                "bounded synthetic task",
                None,
                None,
                None,
                self.provider(),
                baseline_output=IDLE_SCREEN,
            )
        self.assertTrue(accepted)
        probe.assert_called_once()
        backend.return_value.get_history.assert_called_once()
        resend.assert_not_called()

    def test_previous_turn_tool_is_not_acceptance_of_new_task(self):
        screen = TOOL_SCREEN.replace("› Ask Codex", "› new unaccepted task\n› Ask Codex")
        self.assertTrue(hasattr(self.provider(), "has_task_acceptance"))
        self.assertFalse(self.provider().has_task_acceptance(screen))

    def test_previous_turn_429_is_not_a_new_failure(self):
        screen = RATE_SCREEN.replace(
            "› Ask Codex", "› new task\n• Current task finished\n› Ask Codex"
        )
        self.assertEqual(self.provider().get_status(screen), TerminalStatus.COMPLETED)

    def test_rate_limit_after_tool_is_not_success(self):
        screen = TOOL_SCREEN.replace(
            "› Ask Codex",
            "■ exceeded retry limit, last status: 429 Too Many Requests\n› Ask Codex",
        )
        self.assertEqual(self.provider().get_status(screen), TerminalStatus.ERROR)

    def test_failure_receipt_does_not_copy_arbitrary_output(self):
        self.assertTrue(hasattr(ts, "_startup_failure_receipt"))
        with (
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={"tmux_session": "s", "tmux_window": "w"},
            ),
            patch.object(ts, "get_backend") as backend,
        ):
            backend.return_value.get_history.return_value = (
                IDLE_SCREEN + "SYNTHETIC_PRIVATE_TEXT" * 1000
            )
            receipt = ts._startup_failure_receipt("synthetic", self.provider())
        self.assertLessEqual(len(receipt), 1024)
        self.assertNotIn("SYNTHETIC_PRIVATE_TEXT", receipt)
        self.assertIn("startup_unknown", receipt)

    def test_missing_metadata_is_explicit_uncertainty(self):
        self.assertTrue(hasattr(ts, "_startup_failure_receipt"))
        with patch.object(ts, "get_terminal_metadata", return_value=None):
            receipt = ts._startup_failure_receipt("synthetic", self.provider())
        self.assertIn("evidence unavailable", receipt)
        self.assertIn("retained", receipt)

    async def schedule_real_confirmation(self, screen):
        provider = self.provider()
        provider.initialize = AsyncMock()
        provider.shell_baseline = None
        with (
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={
                    "caller_id": "supervisor",
                    "tmux_session": "s",
                    "tmux_window": "w",
                },
            ),
            patch.object(ts, "get_backend") as backend,
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=False)),
            patch.object(ts, "send_input") as send,
            patch.object(ts, "send_special_key") as enter,
            patch.object(ts, "_message_visible_in_box", return_value=False),
            patch.object(ts, "_notify_caller_of_deferred_failure") as notify,
        ):
            backend.return_value.get_history.side_effect = itertools.chain(
                [IDLE_SCREEN], itertools.repeat(screen)
            )
            ts._schedule_deferred_init(provider, "synthetic", "bounded synthetic task", None, None)
            await asyncio.gather(*list(ts._deferred_init_tasks))
        self.assertEqual(send.call_count, 1)  # initial send only
        enter.assert_not_called()
        return notify

    async def test_scheduler_latches_real_tool_receipt_without_replay_or_deletion(self):
        notify = await self.schedule_real_confirmation(TOOL_SCREEN)
        notify.assert_not_called()

    async def test_scheduler_retains_429_evidence_without_replay_or_deletion(self):
        notify = await self.schedule_real_confirmation(RATE_SCREEN)
        args, kwargs = notify.call_args
        self.assertFalse(kwargs.get("delete_worker", args[3] if len(args) > 3 else None))
        self.assertIn("rate_limited_429", args[1])

    async def scheduled(self, *, send_error=None):
        provider = self.provider()
        provider.initialize = AsyncMock()
        provider.shell_baseline = None
        with (
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={
                    "caller_id": "supervisor",
                    "tmux_session": "s",
                    "tmux_window": "w",
                },
            ),
            patch.object(ts, "send_input", side_effect=send_error),
            patch.object(
                ts,
                "_confirm_worker_started_or_resubmit",
                new=AsyncMock(return_value=False),
            ),
            patch.object(ts, "_notify_caller_of_deferred_failure") as notify,
            patch.object(ts, "get_backend") as backend,
        ):
            backend.return_value.get_history.side_effect = itertools.chain(
                [IDLE_SCREEN], itertools.repeat(RATE_SCREEN)
            )
            ts._schedule_deferred_init(provider, "synthetic", "bounded synthetic task", None, None)
            await asyncio.gather(*list(ts._deferred_init_tasks))
        return notify

    async def test_uncertain_startup_retains_worker_and_bounded_receipt(self):
        notify = await self.scheduled()
        args, kwargs = notify.call_args
        delete = kwargs.get("delete_worker", args[3] if len(args) > 3 else None)
        self.assertFalse(delete)
        self.assertLessEqual(len(args[1]), 1024)
        self.assertIn("retained", args[1])
        self.assertIn("rate_limited_429", args[1])
        self.assertIn("cause unknown", args[1])
        self.assertNotIn("re-assign", args[1])

    async def test_exception_after_delivery_attempt_does_not_delete_worker(self):
        notify = await self.scheduled(send_error=RuntimeError("synthetic transport uncertainty"))
        args, kwargs = notify.call_args
        self.assertFalse(kwargs.get("delete_worker", args[3] if len(args) > 3 else None))

    def test_quoted_historical_429_is_not_active_failure(self):
        screen = TOOL_SCREEN.replace("• Ran pwd", "• Historical example: 429 Too Many Requests")
        self.assertEqual(self.provider().get_status(screen), TerminalStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
