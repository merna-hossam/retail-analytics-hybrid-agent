import dspy
from dspy.teleprompt import MIPROv2

from agent.dspy_signatures import RouterModule
from agent.graph_hybrid import _router_heuristic

# ========= 1. CONFIGURE DSPY WITH LOCAL LLM =========
lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)


# ========= 2. BUILD A TINY LABELED DATASET FOR ROUTING =========
train_examples = [
    # --- Pure RAG examples ---
    dspy.Example(
        question="According to the product policy, what is the return window for unopened Beverages?",
        format_hint="int",
        route="rag",
    ),
    dspy.Example(
        question="Per the product policy docs, how many days can unopened Beverages be returned?",
        format_hint="int",
        route="rag",
    ),

    # --- Pure SQL examples ---
    dspy.Example(
        question="What are the top 3 products by total revenue across all time?",
        format_hint="list[{product:str, revenue:float}]",
        route="sql",
    ),
    dspy.Example(
        question="Using the sales database only, list the three products with the highest revenue.",
        format_hint="list[{product:str, revenue:float}]",
        route="sql",
    ),

    # --- Hybrid examples (docs + SQL) ---
    dspy.Example(
        question="What was the Average Order Value during the Winter Classics 1997 campaign?",
        format_hint="float",
        route="hybrid",
    ),
    dspy.Example(
        question="During Summer Beverages 1997, what was the total revenue from the Beverages category?",
        format_hint="float",
        route="hybrid",
    ),
    dspy.Example(
        question="In Summer Beverages 1997, which product category sold the most units?",
        format_hint="{category:str, quantity:int}",
        route="hybrid",
    ),
    dspy.Example(
        question="Per the KPI definition of gross margin, who was the top customer by gross margin in 1997?",
        format_hint="{customer:str, margin:float}",
        route="hybrid",
    ),
]

# Tell DSPy which fields are inputs
trainset = [ex.with_inputs("question", "format_hint") for ex in train_examples]


# ========= 3. DEFINE METRIC FOR ROUTER ACCURACY =========

def router_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Metric for MIPROv2: 1.0 if predicted route matches gold route, else 0.0.
    """
    gold = (example.route or "").strip().lower()
    got = (pred.route or "").strip().lower()
    return 1.0 if gold == got else 0.0


def evaluate_router(module: RouterModule, dataset) -> float:
    """
    Utility to compute average accuracy of a router module over a dataset.
    """
    scores = []
    for ex in dataset:
        pred = module(question=ex.question, format_hint=ex.format_hint)
        scores.append(router_metric(ex, pred))
    return sum(scores) / len(scores) if scores else 0.0


# ========= 4. BASELINE (HEURISTIC) ACCURACY =========

def baseline_heuristic_accuracy(dataset) -> float:
    """
    Evaluate the simple heuristic router directly (without DSPy).
    """
    correct = 0
    for ex in dataset:
        qid = ""  # we don't need an ID here, so pass empty string
        predicted_route = _router_heuristic(ex.question, qid)
        gold = (ex.route or "").strip().lower()
        if predicted_route.strip().lower() == gold:
            correct += 1
    return correct / len(dataset) if dataset else 0.0


# ========= 5. TRAIN ROUTER WITH MIPROv2 =========

def main():
    # 5.1 Baseline heuristic accuracy
    base_heuristic_acc = baseline_heuristic_accuracy(trainset)
    print(f"[Heuristic router] accuracy on trainset: {base_heuristic_acc:.2f}")

    # 5.2 Baseline DSPy router (unoptimized)
    base_router = RouterModule()
    base_dspy_acc = evaluate_router(base_router, trainset)
    print(f"[DSPy RouterModule (unoptimized)] accuracy on trainset: {base_dspy_acc:.2f}")

    # 5.3 Optimize router with MIPROv2
    print("\n[Training] Running MIPROv2 optimization on RouterModule...")
    tp = MIPROv2(
        metric=router_metric,
        auto="light",        # cheap mode
    )

    optimized_router = tp.compile(base_router, trainset=trainset)

    # 5.4 Accuracy after optimization
    optimized_acc = evaluate_router(optimized_router, trainset)
    print(f"[DSPy RouterModule (optimized)] accuracy on trainset: {optimized_acc:.2f}")


if __name__ == "__main__":
    main()
