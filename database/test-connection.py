"""Smoke test for the Neo4j AuraDB connection.

Reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD from the project .env file and
verifies that the driver can reach the instance.
"""

import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


def build_driver() -> Driver:
    """Create a Neo4j driver from the credentials stored in .env.

    Returns:
        A configured Neo4j driver. Connectivity is not checked here.

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

    return GraphDatabase.driver(uri, auth=(user, password))


def main() -> None:
    """Open a driver, verify connectivity, then close it."""
    driver = build_driver()
    try:
        driver.verify_connectivity()
        print("Connexion OK")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
