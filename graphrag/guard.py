"""Static validation of LLM-generated Cypher, before it reaches the database.

This is the guard-rail layer the TP requires. It is defence in depth, and the
order matters - the weakest check is the one that runs first:

1. This module: reject write clauses, procedure calls and multi-statement input.
2. A forced LIMIT, so a valid but unbounded query cannot pull the whole graph.
3. A read-mode transaction with a timeout (see executor.py), which is enforced by
   the server, not by the text scan below.

Layer 3 is the one that actually holds. Any regex over a language as flexible as
Cypher can in principle be worked around, so the text scan is there to catch the
common cases early and give the model a readable error to repair from - it is not
the thing standing between a prompt injection and your data.

The TP also asks for a read-only *account*. Aura Free rejects SHOW ROLES and
SHOW USERS, so a separate least-privilege user cannot be created on this tier;
the read-mode transaction is the available substitute and is noted as such.
"""

import re

# Clauses that write, or that can reach code able to write.
FORBIDDEN_CLAUSES = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "FOREACH",
    "LOAD CSV",
    "USING PERIODIC COMMIT",
)

# Clause a read query is allowed to start with. RETURN is deliberately absent:
# see require_graph_access below.
ALLOWED_OPENERS = ("MATCH", "OPTIONAL MATCH", "CALL {")

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL)
_TRAILING_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:cypher)?\s*|\s*```\s*$", re.IGNORECASE)


class UnsafeCypher(ValueError):
    """Raised when generated Cypher fails validation."""


def strip_noise(cypher: str) -> str:
    """Blank out comments and string literals so keyword scanning is reliable.

    Literals are replaced rather than deleted, keeping offsets roughly intact.
    Scanning the raw text instead would flag a query that merely mentions a team
    called "Created FC" in a string, while still missing nothing real: keywords
    outside literals survive this step untouched.

    Args:
        cypher: The query text.

    Returns:
        The query with comments and literal contents blanked out.
    """
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", cypher))
    return _STRING_LITERAL.sub(lambda m: "'" + " " * max(len(m.group()) - 2, 0) + "'", without_comments)


def unwrap(raw: str) -> str:
    """Remove markdown fences and surrounding whitespace from a model reply.

    The prompt forbids fences, but models add them often enough that stripping
    them is cheaper than a repair round-trip.

    Args:
        raw: The model's raw reply.

    Returns:
        The bare query text.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text)
    return text.strip().rstrip(";").strip()


def validate(cypher: str, max_limit: int) -> str:
    """Check generated Cypher and return a bounded, read-only query.

    Args:
        cypher: The query proposed by the model, already unwrapped.
        max_limit: The largest row count the query may return.

    Returns:
        The query, with LIMIT added or clamped so it cannot exceed max_limit.

    Raises:
        UnsafeCypher: If the query writes, calls a procedure, contains more than
            one statement, or does not start with a read clause.
    """
    if not cypher.strip():
        raise UnsafeCypher("empty query")

    scannable = strip_noise(cypher)

    if ";" in scannable.strip().rstrip(";"):
        raise UnsafeCypher("multiple statements are not allowed")

    upper = scannable.upper()
    for clause in FORBIDDEN_CLAUSES:
        if re.search(rf"(?<![A-Z_]){re.escape(clause)}(?![A-Z_])", upper):
            raise UnsafeCypher(f"write clause {clause} is not allowed")

    # CALL { ... } is a read subquery and stays allowed; CALL procedure.name() is
    # not, because it reaches apoc, dbms and db procedures.
    for match in re.finditer(r"(?<![A-Z_])CALL(?![A-Z_])", upper):
        remainder = scannable[match.end():].lstrip()
        if not remainder.startswith("{"):
            raise UnsafeCypher("procedure calls are not allowed")

    if not upper.lstrip().startswith(tuple(o.upper() for o in ALLOWED_OPENERS)):
        raise UnsafeCypher("query must start with MATCH, OPTIONAL MATCH or CALL {")

    require_graph_access(upper)

    return enforce_limit(cypher, max_limit)


def require_graph_access(upper: str) -> None:
    """Reject a query that returns literals without reading the graph.

    Asked something the schema cannot answer, a model will sometimes write
    `RETURN 'I cannot answer that' AS answer`. It runs, it returns one row, and
    the synthesiser then presents that row as data retrieved from the graph -
    text the model wrote about itself, laundered into the grounded context.

    Requiring a MATCH closes that path. A question the graph cannot answer then
    fails validation until the retry budget runs out, and the pipeline says so
    through its own failure message rather than through a fabricated row.

    Args:
        upper: The uppercased, literal-stripped query text.

    Raises:
        UnsafeCypher: If the query contains no MATCH clause.
    """
    if not re.search(r"(?<![A-Z_])MATCH(?![A-Z_])", upper):
        raise UnsafeCypher("query must read the graph with MATCH, not return literals")


def enforce_limit(cypher: str, max_limit: int) -> str:
    """Ensure the query ends with a LIMIT no larger than max_limit.

    A syntactically valid query with no LIMIT can still return every row in the
    graph, so this is a cost and latency control rather than a security one.

    Args:
        cypher: The validated query text.
        max_limit: The largest row count the query may return.

    Returns:
        The query with a trailing LIMIT added or clamped.
    """
    text = cypher.strip().rstrip(";").strip()
    match = _TRAILING_LIMIT.search(strip_noise(text))

    if match is None:
        return f"{text}\nLIMIT {max_limit}"

    if int(match.group(1)) <= max_limit:
        return text

    return _TRAILING_LIMIT.sub(f"LIMIT {max_limit}", text)
