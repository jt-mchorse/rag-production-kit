"""`rag_kit.db.connect` cannot wait forever for a silent Postgres host (#280).

libpq waits forever by default, so a host that accepted TCP and never spoke
Postgres hung the indexer and the pgvector retriever: a probe was still
blocked when a 10 s alarm killed it. An operator's own `connect_timeout` (in
the DSN or `PGCONNECT_TIMEOUT`) still wins over the default.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

psycopg = pytest.importorskip("psycopg")

from rag_kit import db  # noqa: E402


def test_a_silent_host_fails_within_the_timeout() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(listener.accept()[0])
            except OSError:
                continue

    threading.Thread(target=accept, daemon=True).start()
    port = listener.getsockname()[1]
    started = time.monotonic()
    try:
        with pytest.raises(psycopg.OperationalError):
            db.connect(f"postgresql://u:p@127.0.0.1:{port}/x", connect_timeout=1)
        assert time.monotonic() - started < 4
    finally:
        stop.set()
        for c in held:
            c.close()
        listener.close()


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_connect(dsn: str, **kwargs: object) -> object:
        calls.append({"dsn": dsn, **kwargs})
        return object()

    monkeypatch.setattr(db.psycopg, "connect", fake_connect)
    monkeypatch.delenv("PGCONNECT_TIMEOUT", raising=False)
    return calls


def test_the_default_applies_when_nothing_sets_one(recorded: list[dict]) -> None:
    db.connect("postgresql://u:p@db.example/x")
    assert recorded[-1].get("connect_timeout") == db.DEFAULT_CONNECT_TIMEOUT_S


def test_a_dsn_connect_timeout_is_left_to_libpq(recorded: list[dict]) -> None:
    db.connect("postgresql://u:p@db.example/x?connect_timeout=3")
    assert "connect_timeout" not in recorded[-1]


def test_pgconnect_timeout_is_left_to_libpq(
    recorded: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PGCONNECT_TIMEOUT", "4")
    db.connect("postgresql://u:p@db.example/x")
    assert "connect_timeout" not in recorded[-1]


def test_an_explicit_argument_is_passed_through(recorded: list[dict]) -> None:
    db.connect("postgresql://u:p@db.example/x?connect_timeout=3", connect_timeout=7)
    assert recorded[-1]["connect_timeout"] == 7
