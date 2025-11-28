import dspy
from typing import List, Dict, Any

# ====== SIGNATURES ======
class RouterSignature(dspy.Signature):
    """
    Decide whether to use RAG, SQL, or a hybrid approach for a question.
    """
    question = dspy.InputField(desc="User question in natural language.")
    format_hint = dspy.InputField(desc="Required output format hint (e.g. int, float, object).")

    route = dspy.OutputField(
        desc="One of: 'rag', 'sql', or 'hybrid'.",
    )


class NLToSQLSignature(dspy.Signature):
    """
    Generate a valid SQLite query from the question, constraints, and schema.
    """
    question = dspy.InputField(desc="User question in natural language.")
    format_hint = dspy.InputField(desc="Required output format hint.")
    plan = dspy.InputField(desc="Structured constraints and notes extracted by planner.")
    schema = dspy.InputField(desc="Database schema info (tables + columns) as a compact text string.")

    sql = dspy.OutputField(
        desc="A valid SQLite query using the provided schema.",
    )


class SynthesizerSignature(dspy.Signature):
    """
    Synthesize the final typed answer from docs + SQL result + constraints.
    """
    question = dspy.InputField(desc="User question in natural language.")
    format_hint = dspy.InputField(desc="Required output format hint.")
    retrieved_docs = dspy.InputField(desc="Relevant doc chunks with ids and scores.")
    plan = dspy.InputField(desc="Structured constraints and notes from planner.")
    sql = dspy.InputField(desc="The final executed SQL query (if any).")
    sql_result = dspy.InputField(desc="Execution result: columns + rows + error flag.")

    final_answer = dspy.OutputField(
        desc="Answer that strictly matches the format_hint (int, float, object, list, etc.)."
    )
    explanation = dspy.OutputField(
        desc="Short explanation (<= 2 sentences)."
    )
    citations = dspy.OutputField(
        desc="List of citations: DB tables used and doc chunk IDs, e.g. ['Orders', 'product_policy::chunk0']."
    )
    confidence = dspy.OutputField(
        desc="Numeric confidence between 0 and 1."
    )


# ====== MODULE WRAPPERS ======


class RouterModule(dspy.Module):
    """
    DSPy module that wraps RouterSignature.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RouterSignature)

    def forward(self, question: str, format_hint: str) -> dspy.Prediction:
        return self.predict(question=question, format_hint=format_hint)


class NLToSQLModule(dspy.Module):
    """
    DSPy module that wraps NLToSQLSignature.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(NLToSQLSignature)

    def forward(self, question: str, format_hint: str, plan: dict, schema_text: str) -> dspy.Prediction:
        return self.predict(
            question=question,
            format_hint=format_hint,
            plan=plan,
            schema=schema_text,
        )


class SynthesizerModule(dspy.Module):
    """
    DSPy module that wraps SynthesizerSignature.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SynthesizerSignature)

    def forward(
        self,
        question: str,
        format_hint: str,
        retrieved_docs: List[Dict[str, Any]],
        plan: dict,
        sql: str,
        sql_result: Dict[str, Any],
    ) -> dspy.Prediction:
        return self.predict(
            question=question,
            format_hint=format_hint,
            retrieved_docs=retrieved_docs,
            plan=plan,
            sql=sql,
            sql_result=sql_result,
        )
