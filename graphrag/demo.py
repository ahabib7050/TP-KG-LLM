"""Runnable demonstration of the V1 template-based Graph-RAG pipeline.

    uv run python -m graphrag.demo              # LLM router
    uv run python -m graphrag.demo --regex      # rule-based router, no API calls
    uv run python -m graphrag.demo --trace      # show each graph node as it runs
    uv run python -m graphrag.demo "your question here"
"""

import sys
from typing import Any

from graphrag.config import ANSWER_MODEL, ROUTER_MODEL, build_driver, build_llm_client
from graphrag.pipeline import PipelineState, answer_v1, build_pipeline
from graphrag.router import IntentRouter, LLMRouter, RegexRouter, load_vocabulary

DEMO_QUESTIONS = [
    "Qui a marque le plus de buts en Coupe du Monde ?",
    "Which team has won the most matches in the Copa America?",
    "What is the head to head record between Brazil and Argentina?",
    "Which matches did France play in Brazil?",
    "What is the average salary of a Premier League goalkeeper?",
]


def print_state(state: PipelineState) -> None:
    """Print one pipeline result, including the intermediate stages.

    Args:
        state: The final state returned by the graph.
    """
    print(f"\nQ: {state['question']}")
    print(f"   intent : {state.get('intent')}")
    if state.get("params"):
        print(f"   params : {state['params']}")
    if state.get("error"):
        print(f"   error  : {state['error']}")
    records = state.get("records", [])
    print(f"   rows   : {len(records)}")
    if records:
        print(f"   first  : {records[0]}")
    print(f"   tokens : in {state.get('input_tokens', 0)}, out {state.get('output_tokens', 0)}")
    print(f"   A: {state.get('answer')}")


def trace(pipeline: Any, question: str) -> PipelineState:
    """Run one question, printing each node as it completes.

    Args:
        pipeline: A graph compiled by build_pipeline.
        question: The user's question in natural language.

    Returns:
        The final state once the graph reaches END.
    """
    state: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    for step in pipeline.stream(
        {"question": question, "input_tokens": 0, "output_tokens": 0},
        stream_mode="updates",
    ):
        for node, update in step.items():
            keys = ", ".join(k for k in update if update[k] not in (None, [], {}, 0))
            print(f"   -> {node:<11} {keys}")
            for key, value in update.items():
                # Mirror the operator.add reducers: stream_mode "updates" yields
                # per-node deltas, so overwriting would drop the router's usage.
                if key in ("input_tokens", "output_tokens"):
                    state[key] = state.get(key, 0) + value
                else:
                    state[key] = value

    state["question"] = question
    return state  # type: ignore[return-value]


def main() -> None:
    """Run the demo questions through the pipeline and print each stage."""
    flags = {"--regex", "--trace"}
    args = [a for a in sys.argv[1:] if a not in flags]
    use_regex = "--regex" in sys.argv
    use_trace = "--trace" in sys.argv
    questions = args or DEMO_QUESTIONS

    driver = build_driver()
    client = build_llm_client()

    try:
        vocabulary = load_vocabulary(driver)
        print(
            f"Vocabulary: {len(vocabulary.tournaments)} tournaments, "
            f"{len(vocabulary.teams)} teams, {len(vocabulary.countries)} countries"
        )

        router: IntentRouter
        if use_regex:
            router = RegexRouter(vocabulary)
            print("Router: rules (no API call)")
        else:
            router = LLMRouter(client, vocabulary)
            print(f"Router: {ROUTER_MODEL}")
        print(f"Answer: {ANSWER_MODEL}")

        pipeline = build_pipeline(router, driver, client)

        total = 0
        for question in questions:
            if use_trace:
                print(f"\nQ: {question}")
                state = trace(pipeline, question)
                print(f"   A: {state.get('answer')}")
            else:
                state = answer_v1(question, pipeline)
                print_state(state)
            total += state.get("input_tokens", 0) + state.get("output_tokens", 0)

        print(f"\nTotal tokens across {len(questions)} questions: {total}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
