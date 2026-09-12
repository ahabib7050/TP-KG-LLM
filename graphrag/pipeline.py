"""The V1 Graph-RAG pipeline, built as a LangGraph StateGraph.

                    START
                      |
                   [route]          LLM call 1: intent + parameters
                      |
                  [validate]        no LLM: is this safe to run?
                   /      \\
            [reject]      [query]   pre-written Cypher, parameters bound
                |            |
                |       [synthesize]  LLM call 2: grounded answer
                \\          /
                     END

Making the stages graph nodes rather than statements in a function buys three
things the TP cares about: the reject path is a visible edge instead of an early
return, token usage accumulates through state reducers, and the intermediate
state of every run can be inspected or streamed.

The synthesiser is told to use only the records handed to it. Without that
constraint the model answers from its own parametric knowledge, which is exactly
what a retrieval-augmented pipeline exists to avoid - and on football trivia it
would often be right, which makes the failure hard to notice.
"""

import json
import operator
from typing import Annotated, Any, Literal

from anthropic import AnthropicBedrock
from langgraph.graph import END, START, StateGraph
from neo4j import Driver
from typing_extensions import TypedDict

from graphrag.config import ANSWER_MODEL, database_name
from graphrag.router import IntentRouter
from graphrag.templates import INTENT_TEMPLATES, InvalidRouting, validate_routing

SYNTHESIS_SYSTEM = (
    "You answer questions about international football using ONLY the query "
    "results provided.\n\n"
    "Rules:\n"
    "- Use only the data given. Never add facts from your own knowledge, even if "
    "you are confident they are correct.\n"
    "- If the results are empty, say plainly that the graph holds no data for that "
    "question. Do not guess.\n"
    "- Cite the actual figures from the data.\n"
    "- Answer in the language the question was asked in.\n"
    "- Be concise: two or three sentences unless the data warrants more."
)

REJECTION_TEXT = (
    "I cannot answer that from this graph. It covers top scorers and wins per "
    "tournament, head-to-head records between two teams, and matches played by a "
    "team in a given country."
)


class PipelineState(TypedDict, total=False):
    """State shared by every node of the graph.

    input_tokens and output_tokens carry operator.add reducers, so the router and
    synthesiser nodes each add their own usage instead of overwriting the other's.

    Attributes:
        question: The user's question in natural language.
        intent: The intent chosen by the router.
        raw_params: Parameters as proposed by the router, before validation.
        params: Validated parameters actually bound to the query.
        records: Rows returned by Neo4j.
        answer: The final natural-language answer.
        error: Why the question was rejected, when it was.
        input_tokens: Input tokens billed across all model calls.
        output_tokens: Output tokens billed across all model calls.
    """

    question: str
    intent: str
    raw_params: dict[str, Any]
    params: dict[str, Any]
    records: list[dict[str, Any]]
    answer: str
    error: str | None
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def build_pipeline(
    router: IntentRouter, driver: Driver, client: AnthropicBedrock
) -> Any:
    """Compile the V1 pipeline graph around its three dependencies.

    The dependencies are closed over rather than stored in state: they are not
    data about the run, and keeping them out of state means the state stays
    serialisable for checkpointing later.

    Args:
        router: The intent router to use (rule-based or LLM-backed).
        driver: An open Neo4j driver.
        client: A Bedrock-backed Anthropic client for the synthesis call.

    Returns:
        The compiled LangGraph application, ready for invoke or stream.
    """

    def route(state: PipelineState) -> dict[str, Any]:
        """Classify the question and extract its parameters.

        Args:
            state: The current pipeline state.

        Returns:
            The chosen intent, the raw parameters, and the router's token usage.
        """
        routing = router.route(state["question"])
        return {
            "intent": routing.intent,
            "raw_params": routing.params,
            "input_tokens": routing.input_tokens,
            "output_tokens": routing.output_tokens,
        }

    def validate(state: PipelineState) -> dict[str, Any]:
        """Check the routing before anything reaches the database.

        Args:
            state: The current pipeline state.

        Returns:
            The validated parameters, or the reason the routing was rejected.
        """
        try:
            params = validate_routing(state["intent"], state.get("raw_params", {}))
        except InvalidRouting as exc:
            return {"error": str(exc), "params": {}}
        return {"params": params, "error": None}

    def is_runnable(state: PipelineState) -> Literal["query", "reject"]:
        """Decide whether the validated routing can be executed.

        Args:
            state: The current pipeline state.

        Returns:
            The name of the next node.
        """
        return "reject" if state.get("error") else "query"

    def query(state: PipelineState) -> dict[str, Any]:
        """Run the pre-written Cypher template for the chosen intent.

        Args:
            state: The current pipeline state.

        Returns:
            The rows returned by Neo4j.
        """
        cypher = INTENT_TEMPLATES[state["intent"]].cypher
        records, _, _ = driver.execute_query(
            cypher, database_=database_name(), **state["params"]
        )
        return {"records": [record.data() for record in records]}

    def synthesize(state: PipelineState) -> dict[str, Any]:
        """Turn the rows into an answer grounded in those rows only.

        Args:
            state: The current pipeline state.

        Returns:
            The answer text and the synthesiser's token usage.
        """
        records = state.get("records", [])
        payload = json.dumps(records, ensure_ascii=False, default=str, indent=2)
        content = (
            f"Question: {state['question']}\n\n"
            f"These rows come from the '{state['intent']}' query with parameters "
            f"{state['params']}.\n"
            f"Query results ({len(records)} rows):\n{payload}"
        )

        response = client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYNTHESIS_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return {
            "answer": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def reject(state: PipelineState) -> dict[str, Any]:
        """Explain what the graph can answer, without calling the model.

        Refusing costs nothing here: there is no point paying for a model call to
        say "I don't know".

        Args:
            state: The current pipeline state.

        Returns:
            The canned rejection message.
        """
        return {"answer": REJECTION_TEXT, "records": []}

    return (
        StateGraph(PipelineState)
        .add_node("route", route)
        .add_node("validate", validate)
        .add_node("query", query)
        .add_node("synthesize", synthesize)
        .add_node("reject", reject)
        .add_edge(START, "route")
        .add_edge("route", "validate")
        .add_conditional_edges("validate", is_runnable, ["query", "reject"])
        .add_edge("query", "synthesize")
        .add_edge("synthesize", END)
        .add_edge("reject", END)
        .compile()
    )


def answer_v1(question: str, pipeline: Any) -> PipelineState:
    """Answer one question with the template-based Graph-RAG pipeline.

    Args:
        question: The user's question in natural language.
        pipeline: A graph compiled by build_pipeline.

    Returns:
        The final state, including the intermediate routing and records so the
        pipeline can be inspected rather than treated as a black box.
    """
    return pipeline.invoke(
        {"question": question, "input_tokens": 0, "output_tokens": 0}
    )
