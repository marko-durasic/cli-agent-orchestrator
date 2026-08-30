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
    with patch(
        "cli_agent_orchestrator.providers.grok_cli.get_backend"
    ) as backend:
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
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{workdir}"]\ntrusted = false\n'
    )
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
    (user_home / "trusted_folders.toml").write_text(
        f'[folders."{other}"]\ntrusted = true\n'
    )
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
