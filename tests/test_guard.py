"""Tests for the Cypher guard-rails.

These run offline: no database, no API key, no tokens. The attack cases matter
more than the happy path - a guard that is only ever tested on well-formed input
tells you nothing about what it blocks.

    uv run python -m pytest tests/test_guard.py -q
"""

import pytest

from graphrag.guard import UnsafeCypher, enforce_limit, strip_noise, unwrap, validate

MAX = 50

SAFE_QUERY = "MATCH (t:Team) RETURN t.name AS name LIMIT 10"


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (t:Team) DETACH DELETE t",
        "MATCH (t:Team) SET t.name = 'hacked'",
        "CREATE (n:Evil) RETURN n",
        "MERGE (n:Evil {id: 1}) RETURN n",
        "MATCH (n) REMOVE n.name RETURN n",
        "DROP INDEX team_name",
        "MATCH (t:Team) FOREACH (x IN [1] | SET t.name = 'x')",
    ],
)
def test_write_clauses_are_rejected(cypher: str) -> None:
    """Every write clause named in the TP must be refused.

    Args:
        cypher: A query containing a write clause.
    """
    with pytest.raises(UnsafeCypher):
        validate(cypher, MAX)


@pytest.mark.parametrize(
    "cypher",
    [
        "CALL apoc.cypher.doIt('CREATE (n:Evil)', {}) YIELD value RETURN value",
        "CALL db.labels() YIELD label RETURN label",
        "CALL dbms.components() YIELD name RETURN name",
    ],
)
def test_procedure_calls_are_rejected(cypher: str) -> None:
    """Procedure calls are refused because apoc and dbms can reach writes.

    Args:
        cypher: A query calling a procedure.
    """
    with pytest.raises(UnsafeCypher):
        validate(cypher, MAX)


def test_call_subquery_is_allowed() -> None:
    """CALL { ... } is a read subquery and must not be confused with a procedure."""
    cypher = (
        "MATCH (t:Team) CALL { WITH t MATCH (t)-[:WON]->(m:Match) "
        "RETURN count(m) AS wins } RETURN t.name AS name, wins LIMIT 5"
    )
    assert validate(cypher, MAX).startswith("MATCH")


def test_second_statement_is_rejected() -> None:
    """A piggybacked second statement must not slip through."""
    with pytest.raises(UnsafeCypher):
        validate("MATCH (t:Team) RETURN t LIMIT 1; CREATE (n:Evil)", MAX)


def test_write_hidden_in_a_string_literal_is_allowed() -> None:
    """A team literally named "DELETE FC" is data, not a write clause.

    Scanning the raw text would reject this valid read query.
    """
    cypher = "MATCH (t:Team) WHERE t.name = 'DELETE FC' RETURN t.name AS name LIMIT 5"
    assert "DELETE FC" in validate(cypher, MAX)


def test_comment_cannot_hide_a_write() -> None:
    """Commented-out text is blanked, so it neither hides nor triggers a write."""
    assert "//" in validate(f"{SAFE_QUERY} // CREATE (n:Evil)", MAX)


def test_query_must_start_with_a_read_clause() -> None:
    """An opener outside the allow-list is refused."""
    with pytest.raises(UnsafeCypher):
        validate("SHOW DATABASES", MAX)


def test_literal_only_query_is_rejected() -> None:
    """A query that returns model-authored text instead of reading the graph.

    This is the grounding leak: it executes, returns a row, and the synthesiser
    would present that row as retrieved data.
    """
    with pytest.raises(UnsafeCypher):
        validate("RETURN 'I cannot answer that' AS answer", MAX)


def test_empty_query_is_rejected() -> None:
    """An empty reply from the model is a failure, not a query."""
    with pytest.raises(UnsafeCypher):
        validate("   ", MAX)


def test_missing_limit_is_added() -> None:
    """An unbounded query gets a LIMIT so it cannot return the whole graph."""
    result = validate("MATCH (t:Team) RETURN t.name AS name", MAX)
    assert result.endswith(f"LIMIT {MAX}")


def test_oversized_limit_is_clamped() -> None:
    """A LIMIT above the ceiling is lowered to it."""
    result = enforce_limit("MATCH (t:Team) RETURN t.name AS name LIMIT 100000", MAX)
    assert result.endswith(f"LIMIT {MAX}")


def test_small_limit_is_left_alone() -> None:
    """A LIMIT already under the ceiling is preserved."""
    assert enforce_limit(SAFE_QUERY, MAX) == SAFE_QUERY


def test_markdown_fences_are_stripped() -> None:
    """Models wrap queries in fences despite the prompt; unwrap removes them."""
    assert unwrap(f"```cypher\n{SAFE_QUERY}\n```") == SAFE_QUERY


def test_strip_noise_preserves_keywords_outside_literals() -> None:
    """Blanking literals must not blank a real write clause."""
    assert "DELETE" in strip_noise("MATCH (n) WHERE n.x = 'DELETE' DELETE n").upper()
