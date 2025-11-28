import json
from agent.graph_hybrid import get_agent
from typing import Any, Dict, List

import click

app = get_agent()
def load_questions(batch_path: str) -> List[Dict[str, Any]]:
    """Load questions from a JSONL file."""
    questions = []
    with open(batch_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def run_agent(question_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the hybrid retail analytics agent (LangGraph) on a single question.
    """
    qid = question_item["id"]
    question = question_item["question"]
    format_hint = question_item.get("format_hint", "")

    # Initial state for the graph
    state: Dict[str, Any] = {
        "id": qid,
        "question": question,
        "format_hint": format_hint,
        "trace": [],
        "attempt": 0,
    }

    # Invoke the compiled LangGraph app
    final_state = app.invoke(state)

    # Extract fields with safe defaults
    final_answer = final_state.get("final_answer")
    sql = final_state.get("sql", "")
    confidence = float(final_state.get("confidence", 0.0))
    explanation = final_state.get("explanation", "")
    citations = final_state.get("citations", [])

    return {
        "id": qid,
        "final_answer": final_answer,
        "sql": sql,
        "confidence": confidence,
        "explanation": explanation,
        "citations": citations,
    }


@click.command()
@click.option(
    "--batch",
    "batch_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input JSONL file with questions.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(),
    help="Path to output JSONL file for answers.",
)
def main(batch_path: str, out_path: str) -> None:
    """Main entrypoint for the hybrid retail analytics agent."""
    questions = load_questions(batch_path)

    outputs: List[Dict[str, Any]] = []
    for q in questions:
        result = run_agent(q)
        outputs.append(result)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(outputs)} answers to {out_path}")


if __name__ == "__main__":
    main()
