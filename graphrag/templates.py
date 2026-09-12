"""Pre-written Cypher templates, one per supported intent.

This module is the security boundary of the V1 pipeline. Every query that can
ever reach Neo4j is written here, by hand, and reviewed. The LLM chooses which
template runs and supplies values for its named parameters - it never writes,
extends or concatenates Cypher. A prompt-injected question can therefore change
*which* of these four read-only queries runs, and with what values, but cannot
introduce a fifth query or a write clause.
"""

from dataclasses import dataclass, field
from typing import Any

from graphrag.config import MAX_LIMIT


@dataclass(frozen=True)
class IntentTemplate:
    """One supported question type and the query that answers it.

    Attributes:
        cypher: The parameterised Cypher query.
        required: Parameter names the router must supply, excluding limit.
        description: Sent to the router so it can tell the intents apart.
        takes_limit: Whether the query exposes a $limit parameter.
    """

    cypher: str
    required: tuple[str, ...]
    description: str
    takes_limit: bool = True
    examples: tuple[str, ...] = field(default=())


INTENT_TEMPLATES: dict[str, IntentTemplate] = {
    "top_scorer": IntentTemplate(
        cypher="""
        MATCH (p:Player)-[s:SCORED_IN]->(m:Match)-[:PART_OF]->(t:Tournament)
        WHERE t.name = $tournament
        RETURN p.name AS player, count(s) AS goals
        ORDER BY goals DESC LIMIT $limit
        """,
        required=("tournament",),
        description="Top goal scorers of one tournament.",
        examples=("Who scored the most goals in the FIFA World Cup?",),
    ),
    "most_successful_team": IntentTemplate(
        cypher="""
        MATCH (team:Team)-[:WON]->(m:Match)-[:PART_OF]->(t:Tournament)
        WHERE t.name = $tournament
        RETURN team.name AS team, count(m) AS wins
        ORDER BY wins DESC LIMIT $limit
        """,
        required=("tournament",),
        description="Teams with the most wins in one tournament.",
        examples=("Which team has won the most Copa America matches?",),
    ),
    "head_to_head": IntentTemplate(
        cypher="""
        MATCH (a:Team)-[:PLAYED_HOME|PLAYED_AWAY]->(m:Match)
        MATCH (b:Team)-[:PLAYED_HOME|PLAYED_AWAY]->(m)
        WHERE a.name = $team_a AND b.name = $team_b
        OPTIONAL MATCH (winner:Team)-[:WON]->(m)
        OPTIONAL MATCH (m)-[:PART_OF]->(t:Tournament)
        RETURN m.date AS date, m.home_score AS home_score, m.away_score AS away_score,
               winner.name AS winner, t.name AS tournament
        ORDER BY date DESC LIMIT $limit
        """,
        required=("team_a", "team_b"),
        description="Match history between two given teams.",
        examples=("What is the record between Brazil and Argentina?",),
    ),
    "matches_in_country": IntentTemplate(
        cypher="""
        MATCH (team:Team)-[:PLAYED_HOME|PLAYED_AWAY]->(m:Match)
        MATCH (m)-[:PLAYED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
        WHERE team.name = $team AND c.name = $country
        OPTIONAL MATCH (m)-[:PART_OF]->(t:Tournament)
        RETURN m.date AS date, m.home_score AS home_score, m.away_score AS away_score,
               t.name AS tournament
        ORDER BY date DESC LIMIT $limit
        """,
        required=("team", "country"),
        description="Matches played by one team in one country.",
        examples=("Which matches did France play in Brazil?",),
    ),
}

# Returned when the question falls outside the four templates above. Answering
# "I cannot answer that" is a correct outcome for a closed-world pipeline; making
# something up is not.
UNKNOWN_INTENT = "unknown"


class InvalidRouting(ValueError):
    """Raised when the router output cannot be turned into a safe query."""


def validate_routing(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Check a router decision and return the parameters ready to bind.

    This runs between the LLM and the database. The TP's sample code indexes
    INTENT_TEMPLATES directly with the model's output, which raises KeyError the
    first time the model returns something unexpected.

    Args:
        intent: The intent name proposed by the router.
        params: The parameters proposed by the router.

    Returns:
        Parameters limited to those the template declares, with limit clamped to
        the range 1..MAX_LIMIT.

    Raises:
        InvalidRouting: If the intent is unknown or a required parameter is
            missing or empty.
    """
    template = INTENT_TEMPLATES.get(intent)
    if template is None:
        raise InvalidRouting(f"unsupported intent: {intent!r}")

    clean: dict[str, Any] = {}
    for name in template.required:
        value = params.get(name)
        if not isinstance(value, str) or not value.strip():
            raise InvalidRouting(f"intent {intent!r} requires a non-empty {name!r}")
        clean[name] = value.strip()

    if template.takes_limit:
        raw_limit = params.get("limit", 5)
        limit = raw_limit if isinstance(raw_limit, int) else 5
        clean["limit"] = max(1, min(limit, MAX_LIMIT))

    return clean


def describe_intents() -> str:
    """Render the intent catalogue for the router prompt.

    Returns:
        One line per intent, giving its name, required parameters and purpose.
    """
    lines = []
    for name, template in INTENT_TEMPLATES.items():
        params = ", ".join(template.required) or "none"
        lines.append(f"- {name}: {template.description} Required params: {params}.")
    lines.append(f"- {UNKNOWN_INTENT}: the question cannot be answered by the graph.")
    return "\n".join(lines)
