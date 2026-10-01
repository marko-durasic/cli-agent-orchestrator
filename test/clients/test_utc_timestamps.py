"""last_active / inbox created_at are UTC, stored and served with an offset.

They used to be stamped with a naive ``datetime.now()``: server-local wall time
with nothing in the value saying which zone. Readers guessed, and guessed
differently (the web UI as browser-local, other clients as UTC), so on any
non-UTC server one of them was off by the offset. Retention and the inbox
reconciliation cutoff compared the same columns with a local ``now`` too.

EVERY test here pins a non-UTC zone itself (``TZ=Asia/Taipei``, +08:00). On a
UTC host the old local stamps and UTC coincide, which is exactly how the bug
went unnoticed; a test that relied on the runner's zone would pass on a UTC CI
runner whether or not the bug was present.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.encoders import jsonable_encoder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli_agent_orchestrator.clients.database import (
    Base,
    FlowModel,
    InboxModel,
    TerminalModel,
    create_inbox_message,
    get_flows_to_run,
    get_terminal_metadata,
    list_pending_receiver_ids_older_than,
    list_terminals_by_session,
    update_last_active,
)
from cli_agent_orchestrator.models.inbox import InboxMessage, MessageStatus
from cli_agent_orchestrator.models.terminal import Terminal
from cli_agent_orchestrator.services.ui_state_service import _isoformat

# Generous for a slow runner, and far below the 8 h a zone mix-up produces.
SLACK = timedelta(seconds=60)


@pytest.fixture(autouse=True)
def taipei(monkeypatch):
    """Run the code under test in +08:00, whatever zone the runner is in."""
    monkeypatch.setenv("TZ", "Asia/Taipei")
    time.tzset()
    # Guard the guard: if tzdata were missing the zone would silently be UTC
    # and every test below would prove nothing.
    assert datetime.now().astimezone().utcoffset() == timedelta(hours=8)
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)
    with patch("cli_agent_orchestrator.clients.database.SessionLocal", session):
        yield session


def _is_now_utc(value) -> bool:
    return (
        isinstance(value, datetime)
        and value.utcoffset() == timedelta(0)
        and abs(value - datetime.now(timezone.utc)) < SLACK
    )


def _seed_terminal(db, tid="t1", session="cao-s", last_active=None):
    with db() as s:
        s.add(
            TerminalModel(
                id=tid,
                tmux_session=session,
                tmux_window="w",
                provider="claude_code",
                last_active=last_active,
            )
        )
        s.commit()


class TestStampedUtc:
    def test_update_last_active_reads_back_as_aware_utc_now(self, db):
        _seed_terminal(db)
        assert update_last_active("t1")
        assert _is_now_utc(get_terminal_metadata("t1")["last_active"])

    def test_default_last_active_is_aware_utc_now(self, db):
        _seed_terminal(db)  # no last_active: the column default stamps it
        assert _is_now_utc(get_terminal_metadata("t1")["last_active"])

    def test_session_terminal_list_serializes_with_an_offset(self, db):
        # The dict endpoint (/sessions/{s}/terminals) bypasses the pydantic
        # models, so the offset has to come from the database layer itself.
        _seed_terminal(db)
        update_last_active("t1")
        [row] = list_terminals_by_session("cao-s")
        assert _is_now_utc(row["last_active"])
        served = jsonable_encoder(row)["last_active"]
        assert served.endswith("+00:00"), served

    def test_inbox_created_at_is_aware_utc_now(self, db):
        _seed_terminal(db, "rx")
        msg = create_inbox_message("tx", "rx", "hello")
        assert _is_now_utc(msg.created_at)
        with db() as s:
            stored = s.query(InboxModel).one().created_at
        assert _is_now_utc(stored)


class TestSerializedWithOffset:
    def test_terminal_model_marks_naive_last_active_as_utc(self):
        t = Terminal(
            id="abcdef12",
            name="w",
            provider="claude_code",
            session_name="cao-s",
            last_active=datetime(2026, 10, 1, 13, 38, 12),
        )
        assert json.loads(t.model_dump_json())["last_active"] in (
            "2026-10-01T13:38:12Z",
            "2026-10-01T13:38:12+00:00",
        )

    def test_inbox_model_marks_naive_created_at_as_utc(self):
        m = InboxMessage(
            id=1,
            sender_id="a",
            receiver_id="b",
            message="m",
            status=MessageStatus.PENDING,
            created_at=datetime(2026, 10, 1, 13, 38, 12),
        )
        assert json.loads(m.model_dump_json())["created_at"] in (
            "2026-10-01T13:38:12Z",
            "2026-10-01T13:38:12+00:00",
        )

    def test_ui_state_isoformat_marks_naive_as_utc(self):
        assert _isoformat(datetime(2026, 10, 1, 13, 38, 12)) == "2026-10-01T13:38:12+00:00"

    def test_an_explicit_offset_is_kept_as_given(self):
        # Mirror side: only NAIVE values are reinterpreted.
        taipei = timezone(timedelta(hours=8))
        assert _isoformat(datetime(2026, 10, 1, 21, 38, 12, tzinfo=taipei)) == (
            "2026-10-01T21:38:12+08:00"
        )


class TestCutoffsCompareInUtc:
    def test_reconciliation_cutoff_on_both_sides(self, db):
        now = datetime.now(timezone.utc)
        _seed_terminal(db, "rx-old")
        _seed_terminal(db, "rx-new")
        with db() as s:
            for rx, age in (("rx-old", 120), ("rx-new", 5)):
                s.add(
                    InboxModel(
                        sender_id="a",
                        receiver_id=rx,
                        message="m",
                        status=MessageStatus.PENDING.value,
                        created_at=now - timedelta(seconds=age),
                    )
                )
            s.commit()
        # 120s-old is past a 30s grace window; 5s-old is not. A local cutoff
        # in +08:00 sits 8 h ahead of the UTC rows and returns both.
        assert list_pending_receiver_ids_older_than(30) == ["rx-old"]

    @patch("cli_agent_orchestrator.services.cleanup_service.RETENTION_DAYS", 7)
    @patch("cli_agent_orchestrator.services.cleanup_service.TERMINAL_LOG_DIR")
    @patch("cli_agent_orchestrator.services.cleanup_service.LOG_DIR")
    @patch("cli_agent_orchestrator.services.cleanup_service.status_monitor")
    @patch("cli_agent_orchestrator.services.cleanup_service.fifo_manager")
    def test_retention_cutoff_on_both_sides(self, _fifo, _status, log_dir, term_log_dir, db):
        from cli_agent_orchestrator.services import cleanup_service

        log_dir.exists.return_value = False
        term_log_dir.exists.return_value = False
        now = datetime.now(timezone.utc)
        # One hour either side of the 7-day line: well inside the 8 h a local
        # cutoff would move it, so a zone mix-up deletes the "keep" row.
        _seed_terminal(db, "keep", last_active=now - timedelta(days=7) + timedelta(hours=1))
        _seed_terminal(db, "drop", last_active=now - timedelta(days=7) - timedelta(hours=1))
        with patch.object(cleanup_service, "SessionLocal", db):
            cleanup_service.cleanup_old_data()
        with db() as s:
            assert sorted(r.id for r in s.query(TerminalModel).all()) == ["keep"]


class TestFlowsStayLocal:
    def test_cron_next_run_is_still_compared_in_local_time(self, db):
        # flow_service computes next_run from a cron trigger against a LOCAL
        # datetime.now(), so "0 9 * * *" means 09:00 server-local. Only
        # last_active/created_at moved to UTC; moving this comparison alone
        # would shift every flow by the server's offset.
        local_now = datetime.now()
        with db() as s:
            for name, delta in (("due", -60), ("later", 3600)):
                s.add(
                    FlowModel(
                        name=name,
                        file_path=f"/x/{name}.md",
                        schedule="0 9 * * *",
                        agent_profile="dev",
                        provider="claude_code",
                        script="",
                        enabled=True,
                        next_run=local_now + timedelta(seconds=delta),
                    )
                )
            s.commit()
        assert [f.name for f in get_flows_to_run()] == ["due"]
