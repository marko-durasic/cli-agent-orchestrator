"""Unit tests for the ollama provider.

The fixtures are REAL output captured from ollama 0.33.2 on the agent host,
not invented. The CAO provider skill is explicit that guessed status patterns
are the usual failure mode for these adapters, and an earlier probe of this
same REPL returned an empty buffer — which would have produced a confidently
wrong detector had it been trusted.
"""

from pathlib import Path

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.ollama_cli import (
    DEFAULT_MODEL,
    OllamaCliProvider,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def provider(monkeypatch):
    p = OllamaCliProvider("t1", "sess", "win")
    # Isolate the detector: these tests are about the patterns, not the
    # backend plumbing that supplies the buffer.
    monkeypatch.setattr(p, "_resolve_native_status", lambda buffer: None)
    monkeypatch.setattr(p, "_resolve_buffer", lambda buffer: buffer)
    return p


class TestStatusDetection:
    def test_answer_then_prompt_is_completed(self, provider):
        assert provider.get_status(fixture("ollama_completed.txt")) == TerminalStatus.COMPLETED

    def test_bare_prompt_is_idle(self, provider):
        assert provider.get_status(fixture("ollama_idle.txt")) == TerminalStatus.IDLE

    def test_model_pull_is_processing_not_a_hang(self, provider):
        # A cold model downloads gigabytes before the prompt appears. Reading
        # that as IDLE would send a task into a shell; reading it as ERROR
        # would abandon a working start.
        assert provider.get_status(fixture("ollama_pulling.txt")) == TerminalStatus.PROCESSING

    def test_empty_buffer_is_unknown(self, provider):
        assert provider.get_status("") == TerminalStatus.UNKNOWN

    def test_missing_model_is_error(self, provider):
        out = ">>> \nError: model 'nope:1b' not found, try pulling it first\n"
        assert provider.get_status(out) == TerminalStatus.ERROR

    def test_open_multiline_is_not_idle(self, provider):
        # Ollama's triple-quote mode swallows the next input into the string,
        # so a task sent now would vanish rather than run.
        out = '>>> """\n'
        assert provider.get_status(out) == TerminalStatus.WAITING_USER_ANSWER

    def test_ansi_codes_do_not_break_detection(self, provider):
        coloured = "\x1b[32m>>> \x1b[0mreply\n\x1b[1mPONG\x1b[0m\n>>> "
        assert provider.get_status(coloured) == TerminalStatus.COMPLETED


class TestMessageExtraction:
    def test_extracts_the_answer_without_the_echoed_input(self, provider):
        assert provider.extract_last_message_from_script(
            fixture("ollama_completed.txt")
        ) == "PONG"

    def test_no_response_raises(self, provider):
        with pytest.raises(ValueError):
            provider.extract_last_message_from_script(fixture("ollama_idle.txt"))

    def test_multiline_answer_is_kept_whole(self, provider):
        out = ">>> explain\nline one\nline two\n>>> "
        assert provider.extract_last_message_from_script(out) == "line one\nline two"


class TestModelSelection:
    def test_defaults_when_nothing_configured(self, provider):
        assert provider.model == DEFAULT_MODEL

    def test_explicit_model_wins(self):
        p = OllamaCliProvider("t1", "sess", "win", model="qwen3.5:4b")
        assert p.model == "qwen3.5:4b"


class TestCliContract:
    def test_exit_command(self, provider):
        assert provider.exit_cli() == "/bye"

    def test_inline_repl_can_use_the_direct_status_probe(self, provider):
        # alternate_on was measured as 0: this is scrollback, not a TUI, so a
        # rendered snapshot is exactly what the patterns were calibrated on.
        assert provider.supports_direct_status_probe is True
