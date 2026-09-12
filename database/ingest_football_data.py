"""Ingest the international football dataset (1872-2026) into Neo4j AuraDB.

Adapted from the DataCamp tutorial script (Bex Tuychiev, 2024):
- reads the CSV files locally from data/ instead of fetching them from GitHub
- renames the Cypher variable `as` (a reserved keyword) to `aws`
- reports node and relationship counts once the ingestion is done

Run this exactly once against an empty database: goals are inserted with CREATE,
so a second run would duplicate every SCORED_FOR / SCORED_IN relationship.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Session
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
GOALSCORERS_CSV = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"

BATCH_SIZE = 5000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_driver() -> Driver:
    """Create a Neo4j driver from the credentials stored in .env.

    Returns:
        A driver whose connectivity has already been verified.

    Raises:
        ValueError: If any of the three required environment variables is missing.
    """
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not (uri and user and password):
        raise ValueError(
            "Missing credentials: set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD in .env"
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three source CSV files with the date column parsed.

    Returns:
        The results, goalscorers and shootouts dataframes, in that order.
    """
    logger.info("Loading data...")
    return (
        pd.read_csv(RESULTS_CSV, parse_dates=["date"]),
        pd.read_csv(GOALSCORERS_CSV, parse_dates=["date"]),
        pd.read_csv(SHOOTOUTS_CSV, parse_dates=["date"]),
    )


def make_match_id(row: pd.Series) -> str:
    """Build the synthetic join key shared by the three CSV files.

    The source data has no match identifier, so date plus both team names is used as
    a composite key. It is what lets goals and shootouts find their match node.

    Args:
        row: A row holding the date, home_team and away_team fields.

    Returns:
        An identifier such as "1872-11-30 00:00:00_Scotland_England".
    """
    return f"{row['date']}_{row['home_team']}_{row['away_team']}"


def create_indexes(session: Session) -> None:
    """Create the indexes backing every MERGE key.

    Without them each MERGE scans all nodes carrying the label, which turns the
    ingestion from seconds into hours.

    Args:
        session: An open Neo4j session.
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (t:Team) ON (t.name)",
        "CREATE INDEX IF NOT EXISTS FOR (m:Match) ON (m.id)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Player) ON (p.name)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Tournament) ON (t.name)",
        "CREATE INDEX IF NOT EXISTS FOR (c:City) ON (c.name)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Country) ON (c.name)",
    ]
    for index in indexes:
        session.run(index)
    print("Indexes created.")


def ingest_matches(session: Session, df: pd.DataFrame) -> None:
    """Ingest matches along with their teams, tournament, city and country.

    Every entity is MERGEd so that a team or a tournament ends up as a single shared
    node. WON / LOST / DREW relationships are derived from the score using the
    FOREACH-over-a-conditional-list idiom, since Cypher has no IF statement.

    Args:
        session: An open Neo4j session.
        df: The results dataframe.
    """
    query = """
    UNWIND $batch AS row
    MERGE (m:Match {id: row.id})
    SET m.date = date(row.date), m.home_score = row.home_score,
        m.away_score = row.away_score, m.neutral = row.neutral
    MERGE (home:Team {name: row.home_team})
    MERGE (away:Team {name: row.away_team})
    MERGE (t:Tournament {name: row.tournament})
    MERGE (c:City {name: row.city})
    MERGE (country:Country {name: row.country})
    MERGE (home)-[:PLAYED_HOME]->(m)
    MERGE (away)-[:PLAYED_AWAY]->(m)
    MERGE (m)-[:PART_OF]->(t)
    MERGE (m)-[:PLAYED_IN]->(c)
    MERGE (c)-[:LOCATED_IN]->(country)
    WITH m, home, away, row.home_score AS hs, row.away_score AS aws
    FOREACH(_ IN CASE WHEN hs > aws THEN [1] ELSE [] END |
        MERGE (home)-[:WON]->(m)
        MERGE (away)-[:LOST]->(m)
    )
    FOREACH(_ IN CASE WHEN hs < aws THEN [1] ELSE [] END |
        MERGE (away)-[:WON]->(m)
        MERGE (home)-[:LOST]->(m)
    )
    FOREACH(_ IN CASE WHEN hs = aws THEN [1] ELSE [] END |
        MERGE (home)-[:DREW]->(m)
        MERGE (away)-[:DREW]->(m)
    )
    """
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Ingesting matches"):
        batch = df.iloc[i : i + BATCH_SIZE]
        data = [
            {
                "id": make_match_id(row),
                "date": row["date"].strftime("%Y-%m-%d"),
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "neutral": bool(row["neutral"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "tournament": row["tournament"],
                "city": row["city"],
                "country": row["country"],
            }
            for _, row in batch.iterrows()
        ]
        session.run(query, batch=data)


def ingest_goals(session: Session, df: pd.DataFrame) -> None:
    """Ingest goals as SCORED_FOR (player to team) and SCORED_IN (player to match).

    Goals use CREATE rather than MERGE on purpose: each goal is a distinct event
    carrying its own minute, penalty and own_goal flags, so a hat-trick must stay
    three separate relationships.

    Args:
        session: An open Neo4j session.
        df: The goalscorers dataframe.
    """
    query = """
    UNWIND $batch AS row
    MATCH (m:Match {id: row.id})
    MERGE (p:Player {name: row.scorer})
    MERGE (t:Team {name: row.team})
    CREATE (p)-[s:SCORED_FOR]->(t)
    SET s.own_goal = row.own_goal, s.penalty = row.penalty
    FOREACH(_ IN CASE WHEN row.minute IS NOT NULL THEN [1] ELSE [] END |
        SET s.minute = row.minute
    )
    CREATE (p)-[r:SCORED_IN]->(m)
    SET r.own_goal = row.own_goal, r.penalty = row.penalty
    FOREACH(_ IN CASE WHEN row.minute IS NOT NULL THEN [1] ELSE [] END |
        SET r.minute = row.minute
    )
    """
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Ingesting goals"):
        batch = df.iloc[i : i + BATCH_SIZE]
        data = [
            {
                "id": make_match_id(row),
                "scorer": row["scorer"] if pd.notna(row["scorer"]) else "Unnamed Player",
                "team": row["team"],
                "minute": float(row["minute"]) if pd.notnull(row["minute"]) else None,
                "own_goal": bool(row["own_goal"]),
                "penalty": bool(row["penalty"]),
            }
            for _, row in batch.iterrows()
        ]
        if data:
            session.run(query, batch=data)


def ingest_shootouts(session: Session, df: pd.DataFrame) -> None:
    """Ingest penalty shootouts as a HAD_SHOOTOUT relationship to the winning team.

    Args:
        session: An open Neo4j session.
        df: The shootouts dataframe.
    """
    query = """
    UNWIND $batch AS row
    MATCH (m:Match {id: row.id})
    MATCH (w:Team {name: row.winner})
    MERGE (m)-[s:HAD_SHOOTOUT]->(w)
    SET s.winner = row.winner
    FOREACH(_ IN CASE WHEN row.first_shooter IS NOT NULL THEN [1] ELSE [] END |
        SET s.first_shooter = row.first_shooter
    )
    """
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Ingesting shootouts"):
        batch = df.iloc[i : i + BATCH_SIZE]
        data = [
            {
                "id": make_match_id(row),
                "winner": row["winner"],
                "first_shooter": (
                    row["first_shooter"] if pd.notnull(row["first_shooter"]) else None
                ),
            }
            for _, row in batch.iterrows()
        ]
        session.run(query, batch=data)


def verify(session: Session) -> tuple[int, int]:
    """Print node and relationship counts, broken down by label and by type.

    The TP expects roughly 64k nodes and 340k relationships for the 2024 dataset;
    an archive running to 2026 legitimately reports more.

    Args:
        session: An open Neo4j session.

    Returns:
        The total node count and the total relationship count.
    """
    nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"\nNodes         : {nodes:,}")
    print(f"Relationships : {rels:,}")

    print("\nBy label:")
    for record in session.run(
        "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC"
    ):
        print(f"  {record['label']:<12} {record['n']:>8,}")

    print("\nBy relationship type:")
    for record in session.run(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"
    ):
        print(f"  {record['t']:<14} {record['n']:>8,}")

    return nodes, rels


def main() -> None:
    """Run the full ingestion pipeline: indexes, matches, goals, shootouts, checks."""
    results_df, goalscorers_df, shootouts_df = load_dataframes()

    driver = build_driver()
    print("Connected to Neo4j instance successfully!")

    try:
        with driver.session() as session:
            create_indexes(session)
            ingest_matches(session, results_df)
            ingest_goals(session, goalscorers_df)
            ingest_shootouts(session, shootouts_df)
            verify(session)
    finally:
        driver.close()

    print("\nData ingestion completed!")


if __name__ == "__main__":
    main()
