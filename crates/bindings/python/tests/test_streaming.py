"""Tests for db.execute_lazy() cursor-based query streaming.

Focus:
- Iterator protocol yields the same rows as db.execute()
- Early break drops the stream cleanly (subsequent queries still work)
- Columns are exposed before iteration starts
- Empty results behave like normal empty iterators
- Non-streamable queries (mutations, ORDER BY, session cmds, EXPLAIN) raise
- GIL is released during chunk pulls: other Python threads make progress
"""

import threading
import time

import pytest

pytestmark = pytest.mark.gql


@pytest.fixture
def people_db(db):
    """Fresh in-memory DB seeded with five Person nodes."""
    for name, age in (
        ("Alix", 32),
        ("Gus", 28),
        ("Vincent", 45),
        ("Jules", 40),
        ("Mia", 24),
    ):
        db.create_node(["Person"], {"name": name, "age": age})
    return db


def _rows_as_dict_set(result):
    """Convert an iterable of dicts into a frozenset of sorted tuples for set comparison."""
    return frozenset(tuple(sorted(row.items())) for row in result)


def test_streaming_matches_materialized(people_db):
    query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
    materialized = list(people_db.execute(query))
    streamed = list(people_db.execute_lazy(query))

    assert len(materialized) == len(streamed) == 5
    assert _rows_as_dict_set(materialized) == _rows_as_dict_set(streamed)


def test_streaming_yields_dicts_with_column_keys(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5
    assert all(isinstance(row, dict) for row in rows)
    assert all("p.name" in row for row in rows)


def test_stream_columns_exposed_before_iteration(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
    assert stream.columns == ["name", "age"]


def test_streaming_filter(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name"))
    names = {row["name"] for row in rows}
    assert names == {"Alix", "Vincent", "Jules"}


def test_streaming_empty_result(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 999 RETURN p.name"))
    assert rows == []


def test_streaming_early_break(people_db):
    # Pull one row, break, then make sure subsequent queries still work.
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    first = next(iter(stream))
    assert "p.name" in first
    del stream

    # Subsequent query on the same DB works.
    rows = list(people_db.execute("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5


def test_streaming_rejects_mutation(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("INSERT (:Person {name: 'Butch'})")
    msg = str(exc_info.value).lower()
    assert "mutat" in msg or "cannot be streamed" in msg or "execute() instead" in msg


def test_streaming_rejects_order_by(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
    msg = str(exc_info.value).lower()
    assert "push" in msg or "cannot be streamed" in msg


def test_streaming_rejects_session_command(db):
    with pytest.raises(Exception) as exc_info:
        db.execute_lazy("SESSION SET GRAPH analytics")
    assert "session" in str(exc_info.value).lower()


def test_streaming_rejects_explain(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("EXPLAIN MATCH (p:Person) RETURN p.name")
    msg = str(exc_info.value).lower()
    assert "explain" in msg or "cannot be streamed" in msg


def test_streaming_repr(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    assert "ResultStream" in repr(stream)


def test_gil_is_released_during_iteration(people_db):
    """A background Python thread must make progress while another thread
    iterates a stream. If execute_lazy held the GIL for the full query, the
    background thread's counter would not advance during iteration.
    """
    # Seed a larger result set so iteration takes measurable time.
    for i in range(2000):
        people_db.create_node(["Widget"], {"index": i})

    progress = {"count": 0, "stop": False}

    def ticker():
        while not progress["stop"]:
            progress["count"] += 1
            time.sleep(0.0001)

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()

    before = progress["count"]
    rows = 0
    for _row in people_db.execute_lazy("MATCH (w:Widget) RETURN w.index"):
        rows += 1
    after = progress["count"]

    progress["stop"] = True
    thread.join(timeout=1.0)

    assert rows == 2000
    # Very loose assertion: the ticker incremented at least once during iteration.
    # If the GIL were held the entire time, `after - before` could plausibly be 0.
    assert after - before >= 1, (
        f"background thread made no progress during iteration "
        f"(before={before}, after={after}); GIL may not be released"
    )
