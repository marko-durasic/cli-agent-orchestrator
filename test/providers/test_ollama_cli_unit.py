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
from cli_agent_orchestrator.services.settings_service import get_server_settings

FIXTURES = Path(__file__).parent / "fixtures"

# What ollama actually draws when it hands control back, captured from a live
# session. A bare ">>> " is what the model can forge, so it is not a
# terminator; the tests use the real one.
RETURNED = ">>> Send a message (/? for help)"


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
        # Ollama's triple-quote mode swallows the next input into the string, so
        # a task sent now would vanish rather than run. This fixture is a real
        # capture: the opening quotes land on the prompt line after its hint,
        # and the continuation prompt is what marks the open state.
        assert provider.get_status(fixture("ollama_multiline_open_raw.txt")) == (
            TerminalStatus.WAITING_USER_ANSWER
        )

    def test_an_ellipsis_in_an_answer_does_not_forge_a_wait(self, provider):
        # Without an opener in the buffer, a line starting with "..." is just
        # the model trailing off mid-sentence.
        assert provider.get_status(">>> tell me a story\n... and then") == (
            TerminalStatus.PROCESSING
        )

    def test_ansi_codes_do_not_break_detection(self, provider):
        coloured = f"\x1b[32m>>> \x1b[0mreply\n\x1b[1mPONG\x1b[0m\n{RETURNED}"
        assert provider.get_status(coloured) == TerminalStatus.COMPLETED

    def test_raw_pty_capture_of_a_finished_turn_is_completed(self, provider):
        # The rolling buffer holds raw pipe-pane bytes, not a rendered screen.
        # This fixture is a real ollama 0.33.2 session captured through
        # tmux pipe-pane: spinner frames, synchronised-update and hide/show
        # cursor sequences, and a prompt hint drawn with a 28-column cursor-back
        # move. An SGR-only strip leaves those bytes in front of ">>> " and the
        # finished turn reads as PROCESSING forever.
        assert provider.get_status(fixture("ollama_completed_raw.txt")) == (
            TerminalStatus.COMPLETED
        )

    def test_prompt_shaped_model_output_is_not_a_returned_prompt(self, provider):
        # Review's carried-over finding: the model's own output ends the stream
        # on a line that is exactly ">>> ". A bare prompt is forgeable, so it is
        # not a terminator -- only the hint ollama actually draws is.
        assert provider.get_status(">>> show me an empty repl\n>>> ") == (TerminalStatus.PROCESSING)

    def test_interrupted_generation_does_not_deliver_a_partial_answer(self, provider):
        # Review's carried-over finding, now pinned to a real capture: Ctrl-C
        # mid-generation prints ^C and redraws the prompt on the same line, so
        # the buffer looks like a finished turn. COMPLETED here hands the caller
        # half an essay as the final answer.
        assert provider.get_status(fixture("ollama_interrupted_raw.txt")) == (TerminalStatus.IDLE)

    def test_raw_pty_capture_at_the_prompt_is_idle(self, provider):
        assert provider.get_status(fixture("ollama_idle_raw.txt")) == TerminalStatus.IDLE

    @pytest.mark.parametrize(
        "redraw",
        [
            pytest.param("\x1b[2K", id="erase_line"),
            pytest.param("\x1b[?25l\x1b[2K\x1b[?25h", id="hide_cursor_erase_show"),
            pytest.param("\r", id="carriage_return"),
            pytest.param("\x1b[1G", id="cursor_to_column_1"),
        ],
    )
    def test_a_redrawn_prompt_still_completes_the_turn(self, provider, redraw):
        # readline redraws its prompt with cursor-control and erase sequences,
        # not just colour. An SGR-only strip left those bytes in front of ">>> "
        # so the returned prompt failed the line match and a finished turn read
        # as PROCESSING forever.
        buffer = f">>> hello\nhi there\n{redraw}{RETURNED}"
        assert provider.get_status(buffer) == TerminalStatus.COMPLETED

    def test_repl_transcript_in_an_answer_does_not_forge_completion(self, provider):
        # Small local models answer "show me a python repl" with literal ">>>"
        # lines. Matching a prompt anywhere in the buffer read that as a
        # finished turn and handed the caller half an answer.
        streaming = (
            ">>> show me a python repl example\n"
            "Sure:\n"
            ">>> print(1)\n"
            "1\n"
            "and it keeps going"
        )
        assert provider.get_status(streaming) == TerminalStatus.PROCESSING

    def test_a_repl_line_at_the_buffer_edge_is_not_a_returned_prompt(self, provider):
        # The stream happens to stop on a ">>>" line from the model's own
        # transcript. Only the strict idle prompt (bare, or with the "Send a
        # message" hint) means ollama handed control back.
        assert provider.get_status(">>> repl please\nlike this:\n>>> print(1)") == (
            TerminalStatus.PROCESSING
        )

    def test_answer_still_streaming_is_processing_not_idle(self, provider):
        # IDLE would advertise the terminal as free and let a second task land
        # on top of the running one.
        assert provider.get_status(">>> hello\npartial ans") == TerminalStatus.PROCESSING

    def test_long_answer_that_rolled_the_buffer_is_still_completed(self, provider):
        # An answer larger than state_buffer_max drops the prompt that opened
        # the turn, leaving no positional evidence. grok_cli takes the same
        # at_rolling_capacity precaution; without it a big answer reports IDLE
        # and is never harvested.
        capacity = get_server_settings()["state_buffer_max"]
        rolled = ("answer text " * (capacity // 12 + 1)) + f"\n{RETURNED}"
        assert len(rolled) >= capacity
        assert provider.get_status(rolled) == TerminalStatus.COMPLETED

    def test_a_rolled_buffer_of_blank_lines_is_not_an_answer(self, provider):
        # At capacity but with nothing in it. Whitespace is not a completed
        # turn, and claiming one would hand the caller an empty message.
        capacity = get_server_settings()["state_buffer_max"]
        assert provider.get_status("\n" * capacity + RETURNED) == TerminalStatus.IDLE

    def test_headless_buffer_below_capacity_is_not_assumed_complete(self, provider):
        # Same shape, but the buffer never filled, so nothing was dropped and
        # there is no honest reason to claim a turn finished.
        assert provider.get_status(f"stray text\n{RETURNED}") == TerminalStatus.IDLE

    def test_a_redrawn_prompt_with_nothing_typed_stays_idle(self, provider):
        # The hint prompt then a bare redraw. Nothing was sent, so there is no
        # turn to complete.
        assert provider.get_status(f"{RETURNED}\n{RETURNED}") == TerminalStatus.IDLE

    def test_empty_answer_still_completes_the_turn(self, provider):
        # The model returned nothing, but the turn is over. IDLE here would
        # look like the task was never dispatched.
        assert provider.get_status(f">>> hi\n{RETURNED}") == TerminalStatus.COMPLETED


class TestMessageExtraction:
    def test_extracts_the_answer_without_the_echoed_input(self, provider):
        assert provider.extract_last_message_from_script(fixture("ollama_completed.txt")) == "PONG"

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
