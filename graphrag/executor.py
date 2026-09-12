"""Server-enforced execution of generated Cypher.

The static scan in guard.py is a text heuristic. This module is where the real
guarantees live, because both are enforced by Neo4j rather than by us:

- READ_ACCESS: a write inside a read transaction is refused by the server with
  Neo.ClientError.Statement.AccessMode, whatever the text scan missed.
- timeout: a query that runs too long is killed server-side, so a generated
  cartesian product cannot hold the connection open.
"""

from typing import Any

import neo4j
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError, Neo4jError

from graphrag.config import database_name

QUERY_TIMEOUT_SECONDS = 10.0


class QueryFailed(RuntimeError):
    """Raised when Neo4j refuses or fails to run a query.

    Attributes:
        detail: A short, readable reason suitable for feeding back to the model.
    """

    def __init__(self, detail: str) -> None:
        """Store the readable failure reason.

        Args:
            detail: A short description of what went wrong.
        """
        super().__init__(detail)
        self.detail = detail


def run_read_only(
    driver: Driver, cypher: str, timeout: float = QUERY_TIMEOUT_SECONDS
) -> list[dict[str, Any]]:
    """Run a query in a read-only transaction with a server-side timeout.

    Args:
        driver: An open Neo4j driver.
        cypher: The validated query to run.
        timeout: Seconds after which the server aborts the query.

    Returns:
        One dictionary per returned record.

    Raises:
        QueryFailed: If the query is syntactically invalid, attempts a write, or
            fails for any other Neo4j reason. The raw driver exception is never
            propagated, so internal details do not reach the user.
    """
    try:
        with driver.session(
            database=database_name(), default_access_mode=neo4j.READ_ACCESS
        ) as session:
            with session.begin_transaction(timeout=timeout) as tx:
                return [record.data() for record in tx.run(cypher)]
    except CypherSyntaxError as exc:
        raise QueryFailed(f"invalid Cypher syntax: {_first_line(exc)}") from exc
    except Neo4jError as exc:
        if "AccessMode" in str(getattr(exc, "code", "")):
            raise QueryFailed("query attempted a write; only reads are allowed") from exc
        raise QueryFailed(f"query failed: {_first_line(exc)}") from exc


def _first_line(exc: Exception) -> str:
    """Reduce a driver exception to one short line.

    Args:
        exc: The exception raised by the driver.

    Returns:
        The first line of the message, truncated.
    """
    message = str(exc).strip().splitlines()
    return (message[0] if message else exc.__class__.__name__)[:200]
