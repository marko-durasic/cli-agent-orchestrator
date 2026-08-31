"""Tests for MCP server handoff logic.

Single-seam refactor (issue #312, N0): ``_handoff_impl`` was rewritten from a
six-call client-side loop into ONE call to ``POST /terminals/run-step``. These
tests preserve every OBSERVABLE behavior of the old suite (BR-8) — codex banner
content, no-banner for other providers, supervisor id from env, codex fast-fail
when CAO_TERMINAL_ID is unset, terminal_id surfacing, success on completion —
but assert them against the new single-call design rather than the old internal
mocks. (BR-8 explicitly makes observable behavior, not caller code, the
contract; the caller is deliberately rewritten.)
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.mcp_server.server import (
    HandoffContext,
    _handoff_impl,
    _shape_handoff_message,
)


def _ctx(provider, session_name=None, caller_id=None, allowed_tools=None):
    """Build a HandoffContext for mocking _resolve_handoff_provider."""
    return HandoffContext(
        provider=provider,
        session_name=session_name,
        caller_id=caller_id,
        allowed_tools=allowed_tools,
    )


def _ok_run_step_response(terminal_id="dev-term", last_message="task done"):
    """Build a mocked 200 response from POST /terminals/run-step."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "terminal_id": terminal_id,
        "last_message": last_message,
        "status": "completed",
    }
    resp.raise_for_status.return_value = None
    return resp


class TestShapeHandoffMessage:
    """The codex prompt-shaping that stays caller-side (was _send_direct_input_handoff)."""

    def test_codex_prepends_banner_with_supervisor_id(self):
        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            shaped = _shape_handoff_message("codex", "Implement hello world")
        assert shaped.startswith("[CAO Handoff]")
        assert "a1b2c3d4" in shaped
        assert "Implement hello world" in shaped
        assert "Do NOT use send_message" in shaped
        # Original message must appear in full AFTER the banner.
        assert shaped.endswith("Implement hello world")

    def test_non_codex_message_unchanged(self):
        for provider in ("claude_code", "kiro_cli"):
            assert _shape_handoff_message(provider, "Implement hello world") == (
                "Implement hello world"
            )

    def test_codex_no_env_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="CAO_TERMINAL_ID not set"):
                _shape_handoff_message("codex", "Do task")


class TestHandoffMessageContext:
    """Handoff sends the shaped prompt to the run-step endpoint."""

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_codex_provider_sends_banner_to_endpoint(self, mock_provider, _nudge):
        """Codex handoff posts the [CAO Handoff] banner as the prompt."""
        mock_provider.return_value = _ctx("codex")

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
                mock_requests.post.return_value = _ok_run_step_response()
                mock_requests.Timeout = Exception

                result = asyncio.run(_handoff_impl("developer", "Implement hello world"))

        assert result.success is True
        # Exactly one combined call replaces the former six round-trips.
        mock_requests.post.assert_called_once()
        url = mock_requests.post.call_args[0][0]
        assert url.endswith("/terminals/run-step")
        sent_prompt = mock_requests.post.call_args[1]["json"]["prompt"]
        assert sent_prompt.startswith("[CAO Handoff]")
        assert "a1b2c3d4" in sent_prompt
        assert "Implement hello world" in sent_prompt
        assert "Do NOT use send_message" in sent_prompt

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_claude_code_provider_no_banner(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("claude_code")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception

            result = asyncio.run(_handoff_impl("developer", "Implement hello world"))

        assert result.success is True
        sent_prompt = mock_requests.post.call_args[1]["json"]["prompt"]
        assert sent_prompt == "Implement hello world"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_kiro_cli_provider_no_banner(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("kiro_cli")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception

            result = asyncio.run(_handoff_impl("developer", "Implement hello world"))

        assert result.success is True
        sent_prompt = mock_requests.post.call_args[1]["json"]["prompt"]
        assert sent_prompt == "Implement hello world"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_codex_banner_supervisor_id_from_env(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("codex")

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "c0ffee01"}):
            with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
                mock_requests.post.return_value = _ok_run_step_response()
                mock_requests.Timeout = Exception

                asyncio.run(_handoff_impl("developer", "Build feature X"))

        sent_prompt = mock_requests.post.call_args[1]["json"]["prompt"]
        assert "c0ffee01" in sent_prompt
        assert "Build feature X" in sent_prompt

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_codex_fast_fail_when_no_env(self, mock_provider):
        """Codex handoff with no CAO_TERMINAL_ID fails visibly and never posts a
        step (issue #284) — never tell a worker its supervisor is 'unknown'."""
        mock_provider.return_value = _ctx("codex")

        with patch.dict(os.environ, {}, clear=True):
            with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
                mock_requests.Timeout = Exception
                result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is False
        assert "CAO_TERMINAL_ID not set" in result.message
        # Fast-fail: no step is run at all.
        mock_requests.post.assert_not_called()
        # No terminal was created, so none to surface.
        assert result.terminal_id is None

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_codex_original_message_preserved(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("codex")
        original = "Implement the task described in /path/to/task.md. Write tests."

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "deadbeef"}):
            with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
                mock_requests.post.return_value = _ok_run_step_response()
                mock_requests.Timeout = Exception
                asyncio.run(_handoff_impl("developer", original))

        sent_prompt = mock_requests.post.call_args[1]["json"]["prompt"]
        assert sent_prompt.endswith(original)

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_terminal_id_none_when_provider_resolution_fails(self, mock_provider):
        """When provider resolution fails (no terminal created), report none."""
        mock_provider.side_effect = Exception("session not found")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is False
        assert "Handoff failed" in result.message
        assert result.terminal_id is None
        mock_requests.post.assert_not_called()


class TestHandoffOutcomes:
    """Success/failure outcome semantics preserved through the single endpoint."""

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_success_returns_output_and_terminal_id(self, mock_provider, _nudge):
        """On success the worker output + terminal id are surfaced; the server
        owns teardown (the request asks for teardown=True)."""
        mock_provider.return_value = _ctx("kiro_cli")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response(
                terminal_id="dev-t1", last_message="done"
            )
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is True
        assert result.output == "done"
        assert result.terminal_id == "dev-t1"
        # The single combined call requests server-side teardown.
        assert mock_requests.post.call_args[1]["json"]["teardown"] is True

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_use_worktree_defaults_to_false_in_the_payload(self, mock_provider, _nudge):
        """issue #100 Phase 1: unconditionally present in the payload (unlike
        the Optional fields above) so the server always sees an explicit
        value, matching RunStepRequest's own unconditional default."""
        mock_provider.return_value = _ctx("kiro_cli")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            asyncio.run(_handoff_impl("developer", "Do task"))

        assert mock_requests.post.call_args[1]["json"]["use_worktree"] is False

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_use_worktree_true_reaches_the_payload(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("kiro_cli")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            asyncio.run(_handoff_impl("developer", "Do task", use_worktree=True))

        assert mock_requests.post.call_args[1]["json"]["use_worktree"] is True

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_endpoint_504_maps_to_timeout_result(self, mock_provider):
        """A 504 (worker ran long) becomes a timeout failure and reads the live
        terminal id from the STRUCTURED detail field (not a regex scrape)."""
        mock_provider.return_value = _ctx("kiro_cli")

        timeout_resp = MagicMock()
        timeout_resp.status_code = 504
        timeout_resp.json.return_value = {
            "detail": {
                "message": "step on terminal a1b2c3d4 did not complete within 600s",
                "kind": "timeout",
                "terminal_id": "a1b2c3d4",
            }
        }
        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = timeout_resp
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task", timeout=600))

        assert result.success is False
        assert "timed out after 600 seconds" in result.message
        assert result.terminal_id == "a1b2c3d4"

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_endpoint_502_maps_to_worker_errored_result(self, mock_provider):
        """A 502 (worker CRASHED) is reported as an error — NOT as a timeout —
        so a fast crash is not mislabeled as an N-second timeout."""
        mock_provider.return_value = _ctx("kiro_cli")

        crash_resp = MagicMock()
        crash_resp.status_code = 502
        crash_resp.json.return_value = {
            "detail": {
                "message": "terminal a1b2c3d4 reached ERROR status",
                "kind": "error",
                "terminal_id": "a1b2c3d4",
            }
        }
        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = crash_resp
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task", timeout=600))

        assert result.success is False
        assert "worker errored" in result.message
        assert "timed out" not in result.message
        assert result.terminal_id == "a1b2c3d4"

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_legacy_string_detail_still_scrapes_terminal_id(self, mock_provider):
        """Backward-compat: an older server returning a plain-string detail still
        yields the terminal id via the regex fallback."""
        mock_provider.return_value = _ctx("kiro_cli")

        legacy_resp = MagicMock()
        legacy_resp.status_code = 504
        legacy_resp.json.return_value = {
            "detail": "step on terminal a1b2c3d4 did not complete within 600s"
        }
        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = legacy_resp
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task", timeout=600))

        assert result.success is False
        assert result.terminal_id == "a1b2c3d4"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_malformed_200_surfaces_failure(self, mock_provider, _nudge):
        """A 200 with no last_message must be a failure, not a silent
        success-with-None."""
        mock_provider.return_value = _ctx("kiro_cli")

        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.json.return_value = {"terminal_id": "dev-t1"}  # no last_message
        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = bad_resp
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is False
        assert "malformed" in result.message
        assert result.terminal_id == "dev-t1"

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_endpoint_500_maps_to_failure_result(self, mock_provider):
        mock_provider.return_value = _ctx("kiro_cli")

        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.json.return_value = {"detail": "Failed to run step: boom"}
        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = err_resp
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is False
        assert "Handoff failed" in result.message
        assert "boom" in result.message


class TestHandoffContextPropagation:
    """Regression (PR #320): the run-step payload must carry the supervisor's
    session_name, caller_id and inherited allowed_tools so the worker is created
    in the SAME tmux session with #284 callback routing + tool inheritance — the
    observable behavior the old six-call _create_terminal path provided."""

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_supervisor_context_in_payload(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx(
            "kiro_cli",
            session_name="cao-a1b2c3d4",
            caller_id="sup-abc",
            allowed_tools=["fs_read", "fs_write"],
        )

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert payload["session_name"] == "cao-a1b2c3d4"
        assert payload["caller_id"] == "sup-abc"
        assert payload["allowed_tools"] == ["fs_read", "fs_write"]

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_no_supervisor_omits_session_and_caller(self, mock_provider, _nudge):
        """Outside a CAO terminal there is no supervisor: the payload omits
        session_name/caller_id/allowed_tools so the server auto-creates a fresh
        session (new_session=True)."""
        mock_provider.return_value = _ctx("kiro_cli")  # all context None

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert "session_name" not in payload
        assert "caller_id" not in payload
        assert "allowed_tools" not in payload


class TestHandoffModelOverride:
    """handoff's own `model` parameter -- an explicit per-call model override
    for the worker, threaded through to the run-step payload."""

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_mcode_omits_kiro_engine_and_forwards_model(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("mcode")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            result = asyncio.run(
                _handoff_impl(
                    "report_generator",
                    "Create a report template",
                    engine="v2",
                    model="MiniMax-M2.1",
                )
            )

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert "engine" not in payload
        assert payload["model"] == "MiniMax-M2.1"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_model_is_forwarded_in_payload(self, mock_provider, _nudge):
        mock_provider.return_value = _ctx("claude_code")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task", model="fable-5"))

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert payload["model"] == "fable-5"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    def test_omitted_model_is_absent_from_payload(self, mock_provider, _nudge):
        """No model given -> no 'model' key at all (not None), matching the
        existing convention for every other optional field on this payload
        (session_name/caller_id/allowed_tools/working_directory above)."""
        mock_provider.return_value = _ctx("claude_code")

        with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
            mock_requests.post.return_value = _ok_run_step_response()
            mock_requests.Timeout = Exception
            result = asyncio.run(_handoff_impl("developer", "Do task"))

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert "model" not in payload


class TestResolveHandoffProvider:
    """_resolve_handoff_provider extracts the full supervisor context (not just
    the provider) from the supervisor terminal metadata."""

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_child_allowed_tools")
    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider")
    def test_inside_cao_terminal_extracts_context(self, mock_resolve, mock_child_tools):
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        mock_resolve.return_value = "kiro_cli"
        mock_child_tools.return_value = "fs_read,fs_write"
        meta = MagicMock()
        meta.status_code = 200
        meta.json.return_value = {
            "provider": "kiro_cli",
            "session_name": "cao-sup",
            "allowed_tools": ["fs_read", "fs_write", "execute_bash"],
        }
        meta.raise_for_status.return_value = None

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "c0ffee01"}):
            with patch("cli_agent_orchestrator.mcp_server.server.requests") as mock_requests:
                mock_requests.get.return_value = meta
                ctx = _resolve_handoff_provider("developer")

        assert ctx.provider == "kiro_cli"
        assert ctx.session_name == "cao-sup"
        assert ctx.caller_id == "c0ffee01"
        assert ctx.allowed_tools == ["fs_read", "fs_write"]

    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider")
    def test_outside_cao_terminal_yields_empty_context(self, mock_resolve):
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        mock_resolve.return_value = "kiro_cli"
        with patch.dict(os.environ, {}, clear=True):
            ctx = _resolve_handoff_provider("developer")

        assert ctx.provider == "kiro_cli"
        assert ctx.session_name is None
        assert ctx.caller_id is None
        assert ctx.allowed_tools is None


class TestHandoffProviderOverride:
    """The explicit ``provider=`` override on handoff (and its absence).

    handoff's own description already claimed it would "Create a new terminal
    with the specified agent profile and provider" while its schema accepted no
    provider argument -- the tool documented a capability it did not expose.
    These tests pin the capability now that it exists: omitted inherits, named
    is honored, bad name fails loudly.
    """

    # --- provider omitted: unchanged behavior -----------------------------

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_omitted_provider_passes_no_override(self, mock_requests, mock_provider, _nudge):
        mock_provider.return_value = _ctx("claude_code")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(_handoff_impl("developer", "Do work"))

        assert result.success is True
        mock_provider.assert_called_once_with("developer", provider_override=None)
        _, kwargs = mock_requests.post.call_args
        assert kwargs["json"]["provider"] == "claude_code"

    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider", return_value="claude_code")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_resolve_context_without_override_still_inherits(
        self, mock_requests, mock_resolve_provider
    ):
        """provider_override=None leaves the profile/supervisor resolution untouched."""
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        metadata_response = MagicMock()
        metadata_response.json.return_value = {
            "provider": "kiro_cli",
            "session_name": "cao-session",
            "allowed_tools": None,
        }
        metadata_response.raise_for_status.return_value = None
        mock_requests.get.return_value = metadata_response

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            ctx = _resolve_handoff_provider("reviewer")

        assert ctx.provider == "claude_code"
        mock_resolve_provider.assert_called_once_with("reviewer", fallback_provider="kiro_cli")

    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider", return_value="kiro_cli")
    def test_resolve_context_without_terminal_and_without_override_is_unchanged(
        self, mock_resolve_provider
    ):
        """Outside a CAO terminal the DEFAULT_PROVIDER fallback still applies."""
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        with patch.dict(os.environ, {}, clear=True):
            ctx = _resolve_handoff_provider("reviewer")

        assert ctx.provider == "kiro_cli"
        assert ctx.session_name is None and ctx.caller_id is None

    # --- provider named: that provider is used ----------------------------

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._validate_provider_override")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_named_provider_reaches_the_run_step_payload(
        self, mock_requests, mock_provider, mock_validate, _nudge
    ):
        mock_validate.return_value = "codex"
        mock_provider.return_value = _ctx("codex", caller_id="a1b2c3d4")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(_handoff_impl("reviewer", "Review it", provider="codex"))

        assert result.success is True
        mock_validate.assert_called_once_with("codex")
        mock_provider.assert_called_once_with("reviewer", provider_override="codex")
        _, kwargs = mock_requests.post.call_args
        assert kwargs["json"]["provider"] == "codex"
        # The named provider also drives provider-specific shaping: asking for
        # codex must produce the codex banner, not the supervisor's plain prompt.
        assert kwargs["json"]["prompt"].startswith("[CAO Handoff]")
        assert "codex" in result.message

    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider", return_value="claude_code")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_resolve_context_override_beats_profile_and_supervisor(
        self, mock_requests, mock_resolve_provider
    ):
        """The override wins over BOTH inheritance sources, and short-circuits them."""
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        metadata_response = MagicMock()
        metadata_response.json.return_value = {
            "provider": "kiro_cli",
            "session_name": "cao-session",
            "allowed_tools": None,
        }
        metadata_response.raise_for_status.return_value = None
        mock_requests.get.return_value = metadata_response

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            ctx = _resolve_handoff_provider("reviewer", provider_override="grok_cli")

        assert ctx.provider == "grok_cli"
        assert ctx.session_name == "cao-session"  # still the supervisor's session
        assert ctx.caller_id == "a1b2c3d4"  # still #284 callback routing
        mock_resolve_provider.assert_not_called()

    @patch("cli_agent_orchestrator.mcp_server.server.resolve_provider", return_value="kiro_cli")
    def test_resolve_context_override_applies_outside_a_cao_terminal(self, mock_resolve_provider):
        from cli_agent_orchestrator.mcp_server.server import _resolve_handoff_provider

        with patch.dict(os.environ, {}, clear=True):
            ctx = _resolve_handoff_provider("reviewer", provider_override="grok_cli")

        assert ctx.provider == "grok_cli"
        mock_resolve_provider.assert_not_called()

    # --- provider unknown / unhealthy: loud failure ------------------------

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_unknown_provider_fails_before_resolving_anything(self, mock_requests, mock_provider):
        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(_handoff_impl("reviewer", "Review it", provider="claude-code"))

        assert result.success is False
        assert result.terminal_id is None
        assert "Handoff failed" in result.message
        assert "'claude-code'" in result.message
        assert "Not falling back" in result.message
        mock_provider.assert_not_called()
        mock_requests.post.assert_not_called()

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_uninstalled_provider_fails_before_resolving_anything(
        self, mock_requests, mock_provider
    ):
        health_response = MagicMock()
        health_response.json.return_value = [
            {"name": "grok_cli", "binary": "grok", "installed": False}
        ]
        health_response.raise_for_status.return_value = None
        mock_requests.get.return_value = health_response

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(_handoff_impl("reviewer", "Review it", provider="grok_cli"))

        assert result.success is False
        assert result.terminal_id is None
        assert "'grok'" in result.message  # names the missing binary
        assert "Not falling back" in result.message
        mock_provider.assert_not_called()
        mock_requests.post.assert_not_called()

    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_bad_provider_is_rejected_even_without_a_cao_terminal_id(
        self, mock_requests, mock_provider
    ):
        """Rejected as a bad argument, not conflated with the codex fast-fail."""
        with patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(_handoff_impl("reviewer", "Review it", provider="nonesuch"))

        assert result.success is False
        assert "'nonesuch'" in result.message
        assert "CAO_TERMINAL_ID" not in result.message
        mock_provider.assert_not_called()


class TestHandoffDocstringMatchesSchema:
    """The documentation defect: handoff promised a provider it did not accept."""

    def test_handoff_signature_accepts_provider(self):
        import inspect

        from cli_agent_orchestrator.mcp_server.server import handoff

        assert "provider" in inspect.signature(handoff).parameters

    def test_handoff_docstring_documents_the_provider_parameter(self):
        from cli_agent_orchestrator.mcp_server.server import handoff

        doc = handoff.__doc__ or ""
        assert "## Provider" in doc
        assert "provider: Optional explicit provider" in doc

    def test_handoff_docstring_distinguishes_provider_from_model(self):
        from cli_agent_orchestrator.mcp_server.server import handoff

        doc = handoff.__doc__ or ""
        assert "provider selects WHICH CLI runs the worker" in doc


class TestHandoffProviderOverrideWithOtherParams:
    """provider alongside the other worker parameters on the handoff path.

    Same ordering concern as assign: model is an opaque passthrough (CAO never
    checks a model id against a provider), so what must hold is that model,
    engine and use_worktree all ride the SAME run-step payload as the provider
    that selects the CLI.
    """

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._validate_provider_override")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_model_rides_the_same_payload_as_the_named_provider(
        self, mock_requests, mock_provider, mock_validate, _nudge
    ):
        mock_validate.return_value = "codex"
        mock_provider.return_value = _ctx("codex", caller_id="a1b2c3d4")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(
                _handoff_impl("reviewer", "Review it", model="gpt-5-codex", provider="codex")
            )

        assert result.success is True
        _, kwargs = mock_requests.post.call_args
        payload = kwargs["json"]
        assert payload["provider"] == "codex"
        assert payload["model"] == "gpt-5-codex"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._validate_provider_override")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_engine_gate_follows_the_overridden_provider(
        self, mock_requests, mock_provider, mock_validate, _nudge
    ):
        """engine is kiro-only; overriding to codex must drop it."""
        mock_validate.return_value = "codex"
        mock_provider.return_value = _ctx("codex", caller_id="a1b2c3d4")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(
                _handoff_impl("reviewer", "Review it", engine="v2", provider="codex")
            )

        assert result.success is True
        _, kwargs = mock_requests.post.call_args
        payload = kwargs["json"]
        assert payload["provider"] == "codex"
        assert "engine" not in payload

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._validate_provider_override")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_overriding_to_kiro_cli_re_enables_the_engine_field(
        self, mock_requests, mock_provider, mock_validate, _nudge
    ):
        mock_validate.return_value = "kiro_cli"
        mock_provider.return_value = _ctx("kiro_cli", caller_id="a1b2c3d4")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(
                _handoff_impl("reviewer", "Review it", engine="v2", provider="kiro_cli")
            )

        assert result.success is True
        _, kwargs = mock_requests.post.call_args
        payload = kwargs["json"]
        assert payload["provider"] == "kiro_cli"
        assert payload["engine"] == "v2"

    @patch("cli_agent_orchestrator.mcp_server.server._get_cleanup_nudge", return_value="")
    @patch("cli_agent_orchestrator.mcp_server.server._validate_provider_override")
    @patch("cli_agent_orchestrator.mcp_server.server._resolve_handoff_provider")
    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_use_worktree_rides_the_same_payload_as_the_named_provider(
        self, mock_requests, mock_provider, mock_validate, _nudge
    ):
        mock_validate.return_value = "codex"
        mock_provider.return_value = _ctx("codex", caller_id="a1b2c3d4")
        mock_requests.post.return_value = _ok_run_step_response()

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            result = asyncio.run(
                _handoff_impl("reviewer", "Review it", use_worktree=True, provider="codex")
            )

        assert result.success is True
        _, kwargs = mock_requests.post.call_args
        payload = kwargs["json"]
        assert payload["provider"] == "codex"
        assert payload["use_worktree"] is True
