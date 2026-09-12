"""Shared configuration: Neo4j connection and Bedrock model selection.

Two notes specific to this environment, both verified against the account rather
than assumed:

- The Bedrock *Mantle* client (AnthropicBedrockMantle) needs the IAM action
  bedrock-mantle:CreateInference, which the SSO role here does not have. The
  legacy AnthropicBedrock client goes through bedrock-runtime:InvokeModel, which
  it does have, so that is the client used below.
- Model ids use the `eu.` inference-profile prefix. Inference then stays inside
  the EU, and the bare `anthropic.*` ids are rejected for on-demand throughput.
"""

import os

from anthropic import AnthropicBedrock
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

AWS_REGION = "eu-west-1"

# Router: classification + entity extraction. A small model is the right tool for
# a constrained extraction task, and it is the cheaper half of the V1/V2 energy
# comparison the TP asks for.
ROUTER_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

# Synthesiser: turns records into prose. claude-opus-5 is the intended default but
# is denied on this account (INVALID_PAYMENT_INSTRUMENT), so Opus 4.8 stands in.
ANSWER_MODEL = "eu.anthropic.claude-opus-4-8"

# Cypher generation (V2). Writing correct Cypher against a schema is a reasoning
# task, not an extraction task, so this is the one place V2 cannot use the small
# model that V1 routes with - which is most of why V2 costs more per question.
CYPHER_MODEL = "eu.anthropic.claude-opus-4-8"

# Hard ceiling applied to every $limit, whatever the LLM proposes.
MAX_LIMIT = 50


def build_driver() -> Driver:
    """Create a Neo4j driver from the credentials stored in .env.

    Returns:
        A configured Neo4j driver.

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


def database_name() -> str | None:
    """Return the Neo4j database to query, or None for the instance default.

    Recent AuraDB instances name their database after the instance id rather than
    "neo4j", so a hardcoded "neo4j" raises DatabaseNotFound.

    Returns:
        The value of NEO4J_DATABASE, or None when it is unset.
    """
    load_dotenv()
    return os.getenv("NEO4J_DATABASE")


def build_llm_client() -> AnthropicBedrock:
    """Create the Bedrock client used by both the router and the synthesiser.

    Credentials come from the ambient AWS chain (SSO profile, environment
    variables, instance role) - nothing is read from .env here.

    Returns:
        An Anthropic client bound to the Bedrock runtime in AWS_REGION.
    """
    return AnthropicBedrock(aws_region=AWS_REGION)
