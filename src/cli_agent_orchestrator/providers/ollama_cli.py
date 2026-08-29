"""Ollama provider — local models through ``ollama run``.

Ollama is the odd one in the fleet: it runs models locally, so it has no
quota, no auth and no network dependency. That makes it the cheapest provider
to exercise and the only one that still works when every subscription is
exhausted — worth having even though its models are far smaller.

The REPL was characterised on the agent host (ollama 0.33.2, llama3.2:1b)
rather than assumed, because the CAO provider skill is explicit that guessing
status patterns is the usual way these adapters fail:

    $ ollama run llama3.2:1b
    >>> reply with exactly: PONG
    PONG
    >>> Send a message (/? for help)

Measured properties that shape everything below:

* ``alternate_on`` is 0 — this is an INLINE scrollback REPL, not a full-screen
  TUI. No alt-screen compositing, so line-oriented matching on the rendered
  buffer is sound and ``supports_direct_status_probe`` is safe to enable.
* The idle prompt is ``>>> ``. When empty it renders with the hint
  ``>>> Send a message (/? for help)``; after input it is a bare ``>>> ``.
* A reply is plain text between the submitted line and the next ``>>>``.
  There is no response marker, no spinner glyph and no completion banner —
  which is why COMPLETED is inferred from "prompt returned with text since the
  last prompt" rather than from a marker.
* ``/bye`` exits.
"""

import logging
import re
import shlex
from typing import List, Optional

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.agent_profiles import load_agent_profile
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status

logger = logging.getLogger(__name__)

ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*m"

# The prompt, at start of line. Ollama emits ">>> " when it wants input.
IDLE_PROMPT_PATTERN = r"^>>>\s*$"
# The same prompt with its first-use hint. Matched separately so the empty
# prompt and the hinted prompt are both recognised as "waiting for input".
IDLE_PROMPT_HINT_PATTERN = r"^>>>\s+Send a message"
IDLE_PROMPT_ANY_PATTERN = r"^>>>"
IDLE_PROMPT_PATTERN_LOG = r">>>\s"

# Multi-line paste mode: ollama echoes '"""' and waits for the closing quotes.
# Treated as waiting-for-input rather than idle, because a task sent now would
# be swallowed into the open string instead of running.
#
# UNVERIFIED, unlike everything else here. The prompt, the answer format, the
# pull progress and the exit command were all captured from the live REPL; this
# branch was not — two attempts to type bare triple-quotes through nested shell
# quoting sent escaped literals instead. The pattern is therefore deliberately
# permissive (with or without the ">>> " prefix), and errs toward
# WAITING_USER_ANSWER, which costs a pause rather than a lost task. Worth
# confirming against a real multi-line session before relying on it.
# Matched with or without the prompt prefix: the opening quotes are typed on
# the prompt line (">>> \"\"\""), and later renders may show them alone.
MULTILINE_OPEN_PATTERN = r'^(?:>>>\s*)?"""\s*$'

# Model pull progress. A cold model downloads gigabytes before the prompt
# appears, and that is PROCESSING, not a hang.
PULLING_PATTERN = r"pulling\s+[0-9a-f]+|verifying sha256|writing manifest"

# Fatal startup failures. Anything else is left to the caller rather than
# guessed at, per the skill's warning against over-broad ERROR matching.
ERROR_PATTERN = (
    r"^Error:\s|"
    r"model\s+'[^']+'\s+not found|"
    r"could not connect to ollama"
)

DEFAULT_MODEL = "llama3.2:3b"


class OllamaCliProvider(BaseProvider):
    """Drive ``ollama run <model>`` as a CAO terminal."""

    BINARY_NAME = "ollama"

    # Inline scrollback, so a rendered capture-pane snapshot is exactly what
    # these patterns were calibrated against.
    supports_direct_status_probe = True

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(terminal_id, session_name, window_name, allowed_tools)
        self.agent_profile = agent_profile
        self._model = model

    @property
    def model(self) -> str:
        """Model to run: explicit argument, then the profile, then the default."""
        if self._model:
            return self._model
        if self.agent_profile:
            try:
                profile = load_agent_profile(self.agent_profile)
                configured = getattr(profile, "model", None)
                if configured:
                    return str(configured)
            except Exception:
                # A missing or malformed profile must not stop the terminal
                # coming up; the default model is a working fallback.
                logger.debug("ollama: no model in profile %s", self.agent_profile)
        return DEFAULT_MODEL

    async def initialize(self) -> bool:
        if not await wait_for_shell(self.terminal_id, timeout=15.0):
            raise TimeoutError("Shell initialization timed out after 15 seconds")

        command = shlex.join([self.BINARY_NAME, "run", self.model])
        get_backend().send_keys(self.session_name, self.window_name, command)

        # Generous: a model that is not resident is pulled on first run, and a
        # multi-gigabyte pull on a small instance is slow but not broken.
        if not await wait_until_status(
            self.terminal_id,
            {TerminalStatus.IDLE, TerminalStatus.COMPLETED},
            timeout=300.0,
        ):
            raise TimeoutError(
                f"ollama did not reach its prompt for model {self.model} within 300s"
            )

        self._initialized = True
        return True

    def get_status(self, buffer: str) -> TerminalStatus:
        native = self._resolve_native_status(buffer)
        if native is not None:
            return native

        buffer = self._resolve_buffer(buffer)
        if not buffer:
            return TerminalStatus.UNKNOWN

        clean = re.sub(ANSI_CODE_PATTERN, "", buffer)

        if re.search(ERROR_PATTERN, clean, re.MULTILINE | re.IGNORECASE):
            return TerminalStatus.ERROR

        # An open triple-quote swallows whatever is sent next, so it must not
        # look idle.
        tail = "\n".join(clean.splitlines()[-40:])
        if re.search(MULTILINE_OPEN_PATTERN, tail, re.MULTILINE):
            return TerminalStatus.WAITING_USER_ANSWER

        prompts = list(re.finditer(IDLE_PROMPT_ANY_PATTERN, clean, re.MULTILINE))
        if not prompts:
            # No prompt yet: either still pulling the model or still answering.
            return TerminalStatus.PROCESSING

        # Text between the previous prompt and the last one is a completed
        # answer. Ollama has no response marker, so this positional rule is the
        # only honest signal available.
        if len(prompts) >= 2:
            between = clean[prompts[-2].end() : prompts[-1].start()]
            if between.strip():
                return TerminalStatus.COMPLETED
        return TerminalStatus.IDLE

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Text between the last two prompts, minus the echoed input line."""
        clean = re.sub(ANSI_CODE_PATTERN, "", script_output)
        prompts = list(re.finditer(IDLE_PROMPT_ANY_PATTERN, clean, re.MULTILINE))
        if len(prompts) < 2:
            raise ValueError("No ollama response found in script output")

        block = clean[prompts[-2].end() : prompts[-1].start()]
        lines = block.splitlines()
        # The first line is the remainder of the prompt line the user typed on.
        if lines:
            lines = lines[1:]
        text = "\n".join(lines).strip()
        if not text:
            raise ValueError("No ollama response found in script output")
        return text

    def get_idle_pattern_for_log(self) -> str:
        return IDLE_PROMPT_PATTERN_LOG

    def exit_cli(self) -> str:
        return "/bye"

    def cleanup(self) -> None:
        return None
