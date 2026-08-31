"""Tests for the explicit ``provider=`` override on assign/handoff.

``assign`` and ``handoff`` historically had no way to name a provider: a worker
took the agent profile's own ``provider`` key, falling back to the provider of
the calling terminal. That fallback is silent, so a supervisor asking for a
cross-provider reviewer got a same-provider one and could not tell.
``_validate_provider_override`` is the gate that makes the difference visible --
these tests pin the "omitted inherits / named is honored / bad name fails loudly"
contract at the validator level. The tool-level cases live in test_assign.py and
test_handoff.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.constants import API_BASE_URL
from cli_agent_orchestrator.mcp_server.server import (
    ProviderOverrideError,
    _mcp_timeout,
    _provider_health_error,
    _validate_provider_override,
)


def _providers_response(entries):
    """Build a mocked 200 response from GET /agents/providers."""
    resp = MagicMock()
    resp.json.return_value = entries
    resp.raise_for_status.return_value = None
    return resp


def _healthy(*names):
    return _providers_response([{"name": n, "binary": n, "installed": True} for n in names])


class TestValidateProviderOverrideOmitted:
    """Omitted override must behave exactly as before: inherit, no health call."""

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_none_returns_none(self, mock_requests):
        assert _validate_provider_override(None) is None
        mock_requests.get.assert_not_called()

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_empty_string_returns_none(self, mock_requests):
        """A blank string is "not specified", not "a provider named ''"."""
        assert _validate_provider_override("") is None
        mock_requests.get.assert_not_called()

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_whitespace_only_returns_none(self, mock_requests):
        assert _validate_provider_override("   ") is None
        mock_requests.get.assert_not_called()


class TestValidateProviderOverrideNamed:
    """A named, known, installed provider is returned verbatim."""

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_known_installed_provider_is_returned(self, mock_requests):
        mock_requests.get.return_value = _healthy("codex", "claude_code")
        assert _validate_provider_override("codex") == "codex"
        mock_requests.get.assert_called_once_with(
            f"{API_BASE_URL}/agents/providers", timeout=_mcp_timeout()
        )

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_surrounding_whitespace_is_stripped(self, mock_requests):
        mock_requests.get.return_value = _healthy("codex")
        assert _validate_provider_override("  codex  ") == "codex"

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_provider_absent_from_health_listing_is_not_unhealthy(self, mock_requests):
        """mock_cli has no CLI binary by design; absence of evidence is not evidence.

        GET /agents/providers only enumerates providers that HAVE a binary, so a
        provider missing from that listing must not be rejected.
        """
        mock_requests.get.return_value = _healthy("codex", "claude_code")
        assert _validate_provider_override("mock_cli") == "mock_cli"


class TestValidateProviderOverrideFailsLoudly:
    """Unknown or unhealthy must raise -- never degrade into the supervisor's provider."""

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_unknown_provider_raises(self, mock_requests):
        with pytest.raises(ProviderOverrideError):
            _validate_provider_override("claude-code")  # hyphen typo for claude_code
        assert mock_requests.get.call_count == 0  # rejected before any health call

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_unknown_provider_message_names_it_and_lists_valid_ones(self, mock_requests):
        with pytest.raises(ProviderOverrideError) as exc:
            _validate_provider_override("gpt5")
        message = str(exc.value)
        assert "'gpt5'" in message
        assert "claude_code" in message and "codex" in message  # valid providers listed
        assert "Not falling back" in message

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_uninstalled_provider_raises(self, mock_requests):
        mock_requests.get.return_value = _providers_response(
            [{"name": "codex", "binary": "codex", "installed": False}]
        )
        with pytest.raises(ProviderOverrideError) as exc:
            _validate_provider_override("codex")
        message = str(exc.value)
        assert "'codex'" in message
        assert "not on PATH" in message
        assert "Not falling back" in message

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_uninstalled_message_names_the_missing_binary(self, mock_requests):
        mock_requests.get.return_value = _providers_response(
            [{"name": "cursor_cli", "binary": "agent", "installed": False}]
        )
        with pytest.raises(ProviderOverrideError) as exc:
            _validate_provider_override("cursor_cli")
        assert "'agent'" in str(exc.value)


class TestProviderHealthError:
    """The advisory health probe itself."""

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_installed_provider_has_no_error(self, mock_requests):
        mock_requests.get.return_value = _healthy("grok_cli")
        assert _provider_health_error("grok_cli") is None

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_unreachable_endpoint_does_not_block_the_override(self, mock_requests):
        """Health is advisory: an unreachable endpoint must not silently substitute.

        We cannot prove the provider is bad, so we let the explicit override
        stand. The request still goes out under the REQUESTED provider, so a
        real problem surfaces server-side as a failed terminal rather than as a
        worker quietly running on the supervisor's provider.
        """
        mock_requests.get.side_effect = RuntimeError("connection refused")
        assert _provider_health_error("codex") is None
        assert _validate_provider_override("codex") == "codex"

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_empty_listing_does_not_block_the_override(self, mock_requests):
        mock_requests.get.return_value = _providers_response([])
        assert _provider_health_error("codex") is None

    @patch("cli_agent_orchestrator.mcp_server.server.requests")
    def test_malformed_listing_does_not_raise_out_of_the_tool_call(self, mock_requests):
        """A garbage body degrades to "unknown health", like an unreachable server.

        Anything else would turn an advisory probe into an unhandled exception
        escaping assign/handoff.
        """
        resp = MagicMock()
        resp.json.return_value = {"unexpected": "shape"}
        resp.raise_for_status.return_value = None
        mock_requests.get.return_value = resp
        assert _provider_health_error("codex") is None
        assert _validate_provider_override("codex") == "codex"
