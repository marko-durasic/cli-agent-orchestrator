"""Grok folder-trust inheritance.

CAO gives each terminal a private GROK_HOME with no trust decisions, so Grok
asks for trust on any directory holding repository-local MCP/LSP/hooks config
and initialize() refuses to answer. That refusal is correct — CAO must not
grant permission to run repository-defined code on the human's behalf.

The cost was that a directory the human had ALREADY trusted became unusable
from CAO: a workspace with an ordinary .cursor/mcp.json spawned fine under a
bare `grok` and returned HTTP 500 through CAO.

These tests pin the distinction that makes both true at once: an existing
decision is COPIED, a missing one is never INVENTED.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.providers.grok_cli import GrokCliProvider


@pytest.fixture
def provider(tmp_path):
    p = GrokCliProvider("t1", "sess", "win", "agentpick_dev")
    p._home_override = tmp_path / "private_home"
    return p


def _run_inherit(provider, home: Path, cwd: str):
    with patch("cli_agent_orchestrator.providers.grok_cli.get_backend") as backend:
        backend.return_value.get_pane_working_directory.return_value = cwd
        provider._inherit_folder_trust(home)


def test_existing_trust_is_inherited(provider, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{workdir}"]\ntrusted = true\ndecided_at = 1787204000\n'
    )
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    out = home / "trusted_folders.toml"
    assert out.is_file(), "an existing decision should be carried into the private home"
    assert str(workdir) in out.read_text()


def test_untrusted_directory_is_not_invented(provider, tmp_path, monkeypatch):
    # The decision exists but says NOT trusted. CAO must leave it alone so the
    # dialog still appears and initialize() still refuses.
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(f'[folders."{workdir}"]\ntrusted = false\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_unknown_directory_is_not_trusted(provider, tmp_path, monkeypatch):
    # A decision for a DIFFERENT directory must not leak into this one.
    other = tmp_path / "somewhere_else"
    other.mkdir()
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(f'[folders."{other}"]\ntrusted = true\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_missing_user_store_is_harmless(provider, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no_such_home"))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_a_later_false_entry_wins_over_an_earlier_true(provider, tmp_path, monkeypatch):
    # A store that says both things about one path is ambiguous. Refusing costs
    # a trust dialog; guessing could run repository-defined code the human
    # declined.
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{workdir}"]\ntrusted = true\n\n[folders."{workdir}"]\ntrusted = false\n'
    )
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_a_quoted_path_is_copied_verbatim_not_re_rendered(provider, tmp_path, monkeypatch):
    # Re-rendering the path into a quoted TOML key would emit a bare quote and
    # corrupt the private store. The matched text is copied as-is instead.
    workdir = tmp_path / 'we"ird'
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    escaped = str(workdir).replace('"', '\\"')
    (user_home / "trusted_folders.toml").write_text(f'[folders."{escaped}"]\ntrusted = true\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    import tomllib

    out = home / "trusted_folders.toml"
    assert out.is_file(), "the decision exists, so it should still be inherited"
    parsed = tomllib.loads(out.read_text())
    assert parsed["folders"][str(workdir)]["trusted"] is True


def test_a_commented_out_header_cannot_create_trust(provider, tmp_path, monkeypatch):
    # Review defeated the pattern-matching version with this exact store: the
    # commented header is not TOML, but a regex read it as one and paired it
    # with the real trusted = true on the next line, granting trust for a
    # directory the human never approved.
    approved = tmp_path / "workspace"
    approved.mkdir()
    never_approved = tmp_path / "never_approved"
    never_approved.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{approved}"]\n# [folders."{never_approved}"]\ntrusted = true\n'
    )
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(never_approved))

    assert not (home / "trusted_folders.toml").exists()


def test_a_table_header_inside_a_string_cannot_create_trust(provider, tmp_path, monkeypatch):
    # Same class as the comment: text that looks like a table header but is the
    # body of a multi-line string. Only a parser can tell the difference.
    approved = tmp_path / "workspace"
    approved.mkdir()
    never_approved = tmp_path / "never_approved"
    never_approved.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{approved}"]\ntrusted = true\n'
        f'note = """\n[folders."{never_approved}"]\ntrusted = true\n"""\n'
    )
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(never_approved))

    assert not (home / "trusted_folders.toml").exists()


def test_the_string_true_is_not_a_decision(provider, tmp_path, monkeypatch):
    # Only the literal boolean counts. A truthy-looking string must not stand in
    # for a decision the human made.
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(f'[folders."{workdir}"]\ntrusted = "true"\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_a_malformed_store_fails_closed(provider, tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    (user_home / "trusted_folders.toml").write_text(f'[folders."{workdir}"\ntrusted = true\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    assert not (home / "trusted_folders.toml").exists()


def test_a_path_that_cannot_be_rendered_never_corrupts_the_store(provider, tmp_path, monkeypatch):
    # A newline is legal in a POSIX directory name and legal in a TOML basic
    # string as \n, so the human's store can genuinely hold this decision. The
    # private store must never end up holding TOML that will not parse: writing
    # nothing is the acceptable outcome, writing garbage is not.
    import tomllib

    workdir = tmp_path / "we\nird"
    workdir.mkdir()
    user_home = tmp_path / "dot_grok"
    user_home.mkdir()
    escaped = str(workdir).replace("\\", "\\\\").replace("\n", "\\n")
    (user_home / "trusted_folders.toml").write_text(f'[folders."{escaped}"]\ntrusted = true\n')
    monkeypatch.setenv("GROK_HOME", str(user_home))

    home = tmp_path / "private_home"
    home.mkdir(parents=True, exist_ok=True)
    _run_inherit(provider, home, str(workdir))

    out = home / "trusted_folders.toml"
    if out.exists():
        parsed = tomllib.loads(out.read_text())
        assert parsed["folders"][str(workdir)]["trusted"] is True
