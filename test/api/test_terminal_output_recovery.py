"""One explicit mutation route, guarded by existing write/admin scopes."""

from test.conftest import mint_test_token
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives import serialization

from cli_agent_orchestrator.security import auth
from cli_agent_orchestrator.services import terminal_output_recovery as recovery

URL = "/terminals/abcdef12/output/recover"
BODY = {"expected_session": "cao-existing", "expected_window": "worker-abcdef12"}


@pytest.fixture
def repair(monkeypatch):
    helper = MagicMock(
        return_value={
            "terminal_id": "abcdef12",
            "reader_recovered": True,
            "pipe_rearmed": True,
            "status": "unknown",
            "pane_id": "%3",
        }
    )
    monkeypatch.setattr(recovery, "recover_output", helper)
    return helper


@pytest.fixture
def signing(monkeypatch, rsa_keys):
    monkeypatch.setenv("AUTH0_DOMAIN", "test.local")
    monkeypatch.setenv("AUTH0_AUDIENCE", "cao://test")
    auth.get_jwks_cache().clear()
    private, _ = rsa_keys
    key = serialization.load_pem_private_key(private, password=None).public_key()
    monkeypatch.setattr(
        auth.get_jwks_cache(),
        "get_client",
        lambda _: SimpleNamespace(get_signing_key_from_jwt=lambda _: SimpleNamespace(key=key)),
    )
    yield private
    auth.get_jwks_cache().clear()


def test_requires_explicit_target_metadata(client, repair):
    response = client.post(URL, json={})
    assert response.status_code == 422
    repair.assert_not_called()


def test_invalid_terminal_id_has_no_effect(client, repair):
    assert client.post("/terminals/not-valid/output/recover", json=BODY).status_code == 422
    repair.assert_not_called()


def test_missing_auth_is_rejected(client, repair, signing):
    assert client.post(URL, json=BODY).status_code == 401
    repair.assert_not_called()


@pytest.mark.parametrize("scope", ["cao:read", "cao:metrics", ""])
def test_read_or_unrelated_scope_cannot_recover(client, repair, signing, scope):
    token = mint_test_token(signing, scopes=scope)
    response = client.post(URL, json=BODY, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    repair.assert_not_called()


@pytest.mark.parametrize("scope", ["cao:write", "cao:admin"])
def test_write_or_admin_can_recover_selected_terminal(client, repair, signing, scope):
    token = mint_test_token(signing, scopes=scope)
    response = client.post(URL, json=BODY, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "unknown"
    repair.assert_called_once_with("abcdef12", **BODY)


@pytest.mark.parametrize(
    "error,code",
    [
        (ValueError("missing"), 404),
        (recovery.OutputRecoveryConflict("changed"), 409),
        (OSError("reader open failed"), 503),
    ],
)
def test_failed_recovery_is_not_success(client, repair, error, code):
    repair.side_effect = error
    assert client.post(URL, json=BODY).status_code == code


def test_get_never_performs_recovery(client, repair):
    assert client.get(URL).status_code == 405
    repair.assert_not_called()
