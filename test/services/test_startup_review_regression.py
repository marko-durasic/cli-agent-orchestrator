"""Independent r1 review fixtures and dispatch-correlation regressions."""

import asyncio
import inspect
import unittest
from unittest.mock import AsyncMock, patch

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.codex import CodexProvider
from cli_agent_orchestrator.services import terminal_service as ts

A = "› explain this error\n- Here is the historical error example:\n  ■ exceeded retry limit, last status: 429 Too Many Requests\n› Ask Codex to do anything\n  GPT-6.1-Sol default · ~/synthetic\n"
B = "› Explain the following log:\n  ■ exceeded retry limit, last status: 429 Too Many Requests\n› Ask Codex to do anything\n  GPT-6.1-Sol default · ~/synthetic\n"
C = "› old task\n- Ran pwd\n  └ /old\n› new task not submitted\n  GPT-6.1-Sol default · ~/synthetic\n"
C_NATIVE_BULLET = "› old task\n• Ran pwd\n  └ /old\n› new task not submitted\n  GPT-6.1-Sol default · ~/synthetic\n"

IDLE = "› Ask Codex to do anything\n  GPT-6.1-Sol default · ~/synthetic\n"
CURRENT = "› bounded synthetic task\n• Ran pwd\n  └ /synthetic/worktree\n" + IDLE
OLD_SAME = CURRENT


class StartupReviewRegression(unittest.IsolatedAsyncioTestCase):
    def provider(self):
        return CodexProvider("synthetic", "s", "w", None)

    def test_a_indented_historical_example_is_not_native_failure(self):
        p = self.provider()
        self.assertIsNone(p.startup_failure_reason(A))
        self.assertNotEqual(p.get_status(A), TerminalStatus.ERROR)

    def test_b_multiline_user_content_is_not_native_failure(self):
        p = self.provider()
        self.assertIsNone(p.startup_failure_reason(B))
        self.assertNotEqual(p.get_status(B), TerminalStatus.ERROR)

    def test_c_literal_pending_composer_is_not_accepted(self):
        p = self.provider()
        self.assertFalse(p.has_task_acceptance(C))
        self.assertNotEqual(p.get_status(C), TerminalStatus.COMPLETED)

    def test_c_native_bullet_pending_composer_is_not_accepted(self):
        p = self.provider()
        self.assertFalse(p.has_task_acceptance(C_NATIVE_BULLET))
        self.assertNotEqual(p.get_status(C_NATIVE_BULLET), TerminalStatus.COMPLETED)

    def test_historical_receipt_without_delivery_context_is_not_accepted(self):
        self.assertFalse(self.provider().has_task_acceptance(CURRENT))

    async def confirmation(self, screen, before, cache=True, message="bounded synthetic task"):
        # Cross-revision runner: r1 had no baseline parameter. Exercise its
        # original confirmation path rather than failing merely on its API.
        context = {}
        if (
            "baseline_output"
            in inspect.signature(ts._confirm_worker_started_or_resubmit).parameters
        ):
            context["baseline_output"] = before
        with (
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=cache)),
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={"tmux_session": "s", "tmux_window": "w"},
            ),
            patch.object(ts, "get_backend") as backend,
            patch.object(ts, "send_input") as send,
            patch.object(ts, "send_special_key") as enter,
        ):
            backend.return_value.get_history.return_value = screen
            result = await ts._confirm_worker_started_or_resubmit(
                "synthetic",
                message,
                None,
                "supervisor",
                None,
                self.provider(),
                **context,
            )
        send.assert_not_called()
        enter.assert_not_called()
        return result

    async def test_old_cached_completed_cannot_accept_pending_dispatch(self):
        self.assertFalse(await self.confirmation(C_NATIVE_BULLET, CURRENT))

    async def test_unchanged_old_same_task_receipt_is_not_a_fresh_dispatch(self):
        self.assertFalse(await self.confirmation(OLD_SAME, OLD_SAME))

    async def test_receipt_for_wrong_task_cannot_accept_dispatch(self):
        wrong = CURRENT.replace("bounded synthetic task", "different task")
        self.assertFalse(await self.confirmation(wrong, IDLE))

    async def test_different_prompt_ending_in_expected_text_is_not_correlated(self):
        screen = CURRENT.replace("› bounded synthetic task", "› not bounded synthetic task")
        self.assertFalse(await self.confirmation(screen, IDLE))

    async def test_pending_input_identical_to_legacy_hint_is_not_acceptance(self):
        screen = CURRENT.replace(
            "› Ask Codex to do anything", "› Write tests for @main.py"
        ).replace(
            "GPT-6.1-Sol default · ~/synthetic",
            "gpt-5.3-codex high · 95% left · ~/synthetic",
        )
        self.assertFalse(
            await self.confirmation(screen, CURRENT, message="Write tests for @main.py")
        )

    async def test_unavailable_presend_capture_cannot_accept_dispatch(self):
        self.assertFalse(await self.confirmation(CURRENT, None))

    async def test_echoed_prompt_without_execution_cannot_accept_dispatch(self):
        screen = "› bounded synthetic task\n" + IDLE
        self.assertFalse(await self.confirmation(screen, IDLE))

    def test_fenced_native_looking_429_is_not_a_provider_failure(self):
        screen = (
            "› explain errors\n• Example:\n```text\n■ exceeded retry limit, last status: 429 Too Many Requests\n```\n"
            + IDLE
        )
        self.assertIsNone(self.provider().startup_failure_reason(screen))
        self.assertNotEqual(self.provider().get_status(screen), TerminalStatus.ERROR)

    def test_r3_quoted_transcript_user_marker_cannot_expose_native_429(self):
        for fence in ("`", "```", "~~~"):
            with self.subTest(fence=fence):
                screen = (
                    "› explain errors\n- Example transcript:\n" + fence + "text\n"
                    "› old task\n■ exceeded retry limit, last status: 429 Too Many Requests\n"
                    + fence
                    + "\n"
                    + IDLE
                )
                self.assertIsNone(self.provider().startup_failure_reason(screen))
                self.assertNotEqual(self.provider().get_status(screen), TerminalStatus.ERROR)

    def test_r3_quoted_dispatch_with_matching_prompt_is_unknown(self):
        for fence in ("`", "```", "~~~"):
            for bullet in ("-", "•"):
                with self.subTest(fence=fence, bullet=bullet):
                    screen = (
                        "› other task\n- Example transcript:\n" + fence + "text\n"
                        "› bounded synthetic task\n"
                        + bullet
                        + " Ran pwd\n └ /example\n"
                        + fence
                        + "\n"
                        + IDLE
                    )
                    self.assertEqual(
                        self.provider().startup_delivery_evidence(
                            screen, "bounded synthetic task", IDLE
                        ),
                        "unknown",
                    )

    def test_r3_unclosed_fence_with_native_markers_is_unknown(self):
        screen = "› other task\n```text\n" + CURRENT
        self.assertIsNone(self.provider().startup_failure_reason(screen))
        self.assertEqual(
            self.provider().startup_delivery_evidence(screen, "bounded synthetic task", IDLE),
            "unknown",
        )

    def test_r3_real_failure_after_closed_example_remains_visible(self):
        screen = (
            "› task\n• Example:\n```text\n› historical task\n```\n"
            "■ exceeded retry limit, last status: 429 Too Many Requests\n" + IDLE
        )
        self.assertEqual(self.provider().startup_failure_reason(screen), "rate_limited_429")

    async def test_new_identical_task_requires_a_new_submitted_turn(self):
        new_turn = CURRENT.removesuffix(IDLE) + CURRENT
        self.assertTrue(await self.confirmation(new_turn, CURRENT, cache=False))

    async def test_truncated_nonchronological_capture_is_unknown(self):
        before = "› unrelated earlier task\n• Completed earlier\n" + IDLE
        self.assertFalse(await self.confirmation(CURRENT, before))

    async def test_explicit_memory_prefix_preserves_complete_message_correlation(self):
        screen = CURRENT.replace(
            "› bounded synthetic task",
            "› <cao-memory>\n  synthetic note\n  </cao-memory>\n  bounded synthetic task",
        )
        self.assertTrue(await self.confirmation(screen, IDLE, cache=False))

    def test_shared_probe_blocks_replay_without_claiming_dispatch_acceptance(self):
        with (
            patch.object(
                ts,
                "get_terminal_metadata",
                return_value={"tmux_session": "s", "tmux_window": "w"},
            ),
            patch.object(ts, "get_backend") as backend,
        ):
            backend.return_value.get_history.return_value = CURRENT
            # The shared legacy guard may observe prior activity to block a
            # replay. It must never be used as proof of this dispatch.
            self.assertTrue(ts._worker_is_started_direct("synthetic", self.provider()))
        self.assertFalse(self.provider().has_task_acceptance(CURRENT))

    async def scheduled_capture(self, *, fail_before=False):
        provider = self.provider()
        provider.initialize = AsyncMock()
        provider.shell_baseline = None
        events = []
        sent = False

        def history(*args, **kwargs):
            events.append("capture_after" if sent else "capture_before")
            if not sent and fail_before:
                raise OSError("synthetic unavailable snapshot")
            return CURRENT if sent else IDLE

        def deliver(*args, **kwargs):
            nonlocal sent
            events.append("send")
            sent = True

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
            patch.object(ts, "wait_until_status", new=AsyncMock(return_value=True)),
            patch.object(ts, "send_input", side_effect=deliver) as send,
            patch.object(ts, "send_special_key") as enter,
            patch.object(ts, "_notify_caller_of_deferred_failure") as notify,
        ):
            backend.return_value.get_history.side_effect = history
            ts._schedule_deferred_init(provider, "synthetic", "bounded synthetic task", None, None)
            await asyncio.gather(*list(ts._deferred_init_tasks))
        self.assertEqual(send.call_count, 1)
        enter.assert_not_called()
        return events, notify

    async def test_scheduler_captures_baseline_before_initial_send(self):
        events, notify = await self.scheduled_capture()
        self.assertEqual(events[:2], ["capture_before", "send"])
        self.assertIn("capture_after", events[2:])
        notify.assert_not_called()

    async def test_scheduler_missing_baseline_retains_worker_despite_cached_success(
        self,
    ):
        _, notify = await self.scheduled_capture(fail_before=True)
        notify.assert_called_once()
        args, kwargs = notify.call_args
        self.assertFalse(kwargs.get("delete_worker", args[3] if len(args) > 3 else None))
        self.assertIn("startup_unknown", args[1])

    async def test_partial_snapshot_without_footer_is_not_a_delivery_baseline(self):
        self.assertFalse(await self.confirmation(CURRENT, "› bounded synthetic"))

    async def test_changed_word_boundaries_do_not_match_the_dispatched_message(self):
        screen = CURRENT.replace("bounded synthetic task", "boundedsynthetic task")
        self.assertFalse(await self.confirmation(screen, IDLE))

    async def test_new_correlated_tool_receipt_is_accepted(self):
        self.assertTrue(await self.confirmation(CURRENT, IDLE, cache=False))

    async def test_new_correlated_mcp_receipt_is_accepted(self):
        screen = CURRENT.replace("• Ran pwd", "• Called synthetic.read({})")
        self.assertTrue(await self.confirmation(screen, IDLE, cache=False))

    async def test_new_correlated_native_429_is_failure_not_acceptance(self):
        screen = CURRENT.replace(
            "• Ran pwd\n  └ /synthetic/worktree",
            "■ exceeded retry limit, last status: 429 Too Many Requests",
        )
        self.assertEqual(self.provider().get_status(screen), TerminalStatus.ERROR)
        self.assertFalse(await self.confirmation(screen, IDLE))


if __name__ == "__main__":
    unittest.main()
