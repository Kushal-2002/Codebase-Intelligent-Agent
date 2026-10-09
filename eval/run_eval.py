"""
Compare three retrieval setups on eval/questions.json.

    vector  - top-k semantic search only
    graph   - top-k semantic search, expanded with 1-hop callers and dependencies
    agent   - the planner-driven tool loop in agent.py

Metrics per question:
    retrieval_recall  - fraction of expected function groups that were retrieved
                        (a group is a list of alternatives, any one counts)
    fact_recall       - fraction of expected facts found in the final answer
                        (a fact is a list of alternative strings, case-insensitive)
    context_chars     - size of the context handed to the answer LLM
    latency_s         - wall time for retrieval + answer generation

Run from the project root (Ollama must be running for answers):

    python eval/run_eval.py
    python eval/run_eval.py --retrieval-only
    python eval/run_eval.py --systems vector,agent --ids q01,q14
    LLM_MODEL=llama3.2 python eval/run_eval.py
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tools import search_code, get_callers, get_dependencies, get_source  # noqa: E402
from agent import run_agent, generate_answer  # noqa: E402
from planner import MODEL  # noqa: E402

REPO_PREFIX = "repository/"
TOP_K = 3


# ---------------- retrieval setups ----------------


def node_from_metadata(metadata):

    return f"{metadata['path']}::{metadata['function_name'].split('(')[0]}"


def build_context(nodes):

    context = ""

    for node in nodes:
        source = get_source(node)

        if source:
            context += f"\nFUNCTION ({node}):\n{source}\n"

    return context


def vector_retrieve(question):

    results = search_code(question, n_results=TOP_K)

    return [node_from_metadata(m) for m in results["metadatas"][0]]


def graph_retrieve(question):

    seeds = vector_retrieve(question)

    nodes = list(seeds)

    for seed in seeds:
        for neighbour in get_callers(seed) + get_dependencies(seed):
            if neighbour not in nodes:
                nodes.append(neighbour)

    return nodes


def run_retrieval_system(retrieve, question, with_answer):

    start = time.time()

    nodes = retrieve(question)
    context = build_context(nodes)

    answer = None

    if with_answer:
        answer = generate_answer(question, f"Observation:\n{context}")

    return {
        "nodes": nodes,
        "answer": answer,
        "context_chars": len(context),
        "latency_s": round(time.time() - start, 2),
    }


def run_agent_system(question):

    start = time.time()

    def on_step(step, action, function_name):
        # stderr, because stdout is silenced below
        target = f"({function_name})" if function_name else ""
        print(f"\n      step {step}: {action}{target}", end="", file=sys.stderr, flush=True)

    # the planner prints every step; keep the eval output readable
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_agent(question, verbose=False, on_step=on_step)

    print(file=sys.stderr)

    result["latency_s"] = round(time.time() - start, 2)

    return result


# ---------------- scoring ----------------


def retrieval_recall(expected_groups, nodes):

    retrieved = set(nodes)

    hits = 0

    for group in expected_groups:
        if any(REPO_PREFIX + node in retrieved for node in group):
            hits += 1

    return hits / len(expected_groups)


def fact_recall(expected_facts, answer):

    if answer is None:
        return None

    answer = answer.lower()

    hits = 0

    for alternatives in expected_facts:
        if any(alt.lower() in answer for alt in alternatives):
            hits += 1

    return hits / len(expected_facts)


def mean(values):

    values = [v for v in values if v is not None]

    return sum(values) / len(values) if values else None


def fmt(value, pct=True):

    if value is None:
        return "-"

    return f"{value * 100:.0f}%" if pct else f"{value:,.0f}"


def print_summary(rows, systems):

    print("\n## Overall\n")
    print("| System | Retrieval recall | Fact recall | Avg context (chars) | Avg latency (s) |")
    print("|---|---|---|---|---|")

    for system in systems:
        sys_rows = [r for r in rows if r["system"] == system]

        print(
            f"| {system} "
            f"| {fmt(mean(r['retrieval_recall'] for r in sys_rows))} "
            f"| {fmt(mean(r['fact_recall'] for r in sys_rows))} "
            f"| {fmt(mean(r['context_chars'] for r in sys_rows), pct=False)} "
            f"| {mean(r['latency_s'] for r in sys_rows):.1f} |"
        )

    categories = sorted({r["category"] for r in rows})

    print("\n## Retrieval recall / fact recall by category\n")
    print("| Category | " + " | ".join(systems) + " |")
    print("|---|" + "---|" * len(systems))

    for category in categories:
        cells = []

        for system in systems:
            cat_rows = [r for r in rows if r["system"] == system and r["category"] == category]
            cells.append(
                f"{fmt(mean(r['retrieval_recall'] for r in cat_rows))} / "
                f"{fmt(mean(r['fact_recall'] for r in cat_rows))}"
            )

        print(f"| {category} | " + " | ".join(cells) + " |")


# ---------------- main ----------------


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="vector,graph,agent")
    parser.add_argument("--ids", default=None, help="comma separated question ids")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip answer generation (no Ollama needed); agent is skipped",
    )
    args = parser.parse_args()

    systems = args.systems.split(",")

    if args.retrieval_only and "agent" in systems:
        systems.remove("agent")

    with open("eval/questions.json") as f:
        questions = json.load(f)

    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    rows = []

    total = len(questions) * len(systems)

    print(f"Model: {MODEL} | {len(questions)} questions x {len(systems)} systems = {total} runs")
    print("Press Ctrl+C to stop early; finished runs are still saved.\n", flush=True)

    run_start = time.time()

    try:
        for q in questions:
            for system in systems:
                done = len(rows)
                elapsed = time.time() - run_start
                eta = f", ~{elapsed / done * (total - done) / 60:.0f} min left" if done else ""

                print(f"[{done + 1}/{total}{eta}] {q['id']} {system:<6} ...", end="", flush=True)

                if system == "vector":
                    result = run_retrieval_system(vector_retrieve, q["question"], not args.retrieval_only)
                elif system == "graph":
                    result = run_retrieval_system(graph_retrieve, q["question"], not args.retrieval_only)
                elif system == "agent":
                    result = run_agent_system(q["question"])
                else:
                    raise ValueError(f"unknown system {system}")

                row = {
                    "id": q["id"],
                    "category": q["category"],
                    "system": system,
                    "model": MODEL,
                    "question": q["question"],
                    "retrieval_recall": retrieval_recall(q["expected_functions"], result["nodes"]),
                    "fact_recall": fact_recall(q["expected_facts"], result["answer"]),
                    **result,
                }

                rows.append(row)

                print(
                    f" retrieval={fmt(row['retrieval_recall'])} "
                    f"facts={fmt(row['fact_recall'])} ({row['latency_s']}s)",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(f"\n\nStopped early after {len(rows)}/{total} runs.")

    if not rows:
        return

    os.makedirs("eval/results", exist_ok=True)

    out_path = f"eval/results/run-{MODEL.replace(':', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    # only summarise systems that actually ran (matters after Ctrl+C)
    print_summary(rows, [s for s in systems if any(r["system"] == s for r in rows)])

    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
