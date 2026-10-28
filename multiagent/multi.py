"""
Single-agent vs Multi-agent token cost comparison (Groq).

Setup:
    pip install groq ddgs
    export GROQ_API_KEY="your_key"
    python agent_token_comparison.py

Single agent : 1 LLM with web_search + calculator tools.
Multi agent  : ask_manager_llm (no tools, only routes) ->
                 web_search_llm (web_search tool only)
                 calculator_llm (calculator tool only)
"""

import ast
import json
import operator
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 8  # safety cap per agent loop

# The task needs >= 1 search call and >= 2 calculator calls.
TASK = (
    "Search the web for the height of Mount Everest in meters. Then use the "
    "calculator to (1) convert that height to feet (multiply by 3.28084) and "
    "(2) compute half of the height in meters. Give a short final answer."
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ----------------------------------------------------------------------------
# Token tracking
# ----------------------------------------------------------------------------
class UsageTracker:
    """Collects token usage per agent name."""

    def __init__(self):
        self.stats = defaultdict(
            lambda: {"llm_calls": 0, "prompt": 0, "completion": 0, "total": 0,
                     "search_calls": 0, "calculator_calls": 0}
        )

    def add_llm(self, agent, usage):
        s = self.stats[agent]
        s["llm_calls"] += 1
        s["prompt"] += usage.prompt_tokens
        s["completion"] += usage.completion_tokens
        s["total"] += usage.total_tokens

    def add_tool(self, agent, tool_name):
        key = "search_calls" if tool_name == "web_search" else "calculator_calls"
        self.stats[agent][key] += 1

    def totals(self):
        t = defaultdict(int)
        for s in self.stats.values():
            for k, v in s.items():
                t[k] += v
        return dict(t)


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # older package name
    try:
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
    )


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        return str(_eval(ast.parse(expression.replace(",", ""), mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}"


SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}

CALC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(1.4e9 - 1.2e8) / 2'. "
                       "Supports + - * / ** % and parentheses. No commas in numbers.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

TOOL_IMPLS = {"web_search": web_search, "calculator": calculator}


# ----------------------------------------------------------------------------
# Generic agent loop (an LLM + a set of tools)
# ----------------------------------------------------------------------------
def run_tool_agent(name, system_prompt, user_input, tools, tracker):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools,
            tool_choice="auto", temperature=0,
        )
        tracker.add_llm(name, resp.usage)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = TOOL_IMPLS[tc.function.name](**args)
            tracker.add_tool(name, tc.function.name)
            print(f"  [{name}] {tc.function.name}({args}) -> {result[:120]!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Stopped: max steps reached."


# ----------------------------------------------------------------------------
# 1) Single agent system
# ----------------------------------------------------------------------------
def run_single_agent(task, tracker):
    system = (
        "You are a helpful assistant with two tools: web_search and calculator. "
        "Always use web_search for facts and calculator for any arithmetic. "
        "Do not do math yourself."
    )
    return run_tool_agent("single_agent", system, task, [SEARCH_SCHEMA, CALC_SCHEMA], tracker)


# ----------------------------------------------------------------------------
# 2) Multi agent system
# ----------------------------------------------------------------------------
MANAGER_SYSTEM = """You are ask_manager_llm, a manager with NO tools.
You decide which helper should work next:
- "web_search_llm": finds facts on the web.
- "calculator_llm": does arithmetic using a calculator.
Give each helper one focused instruction containing all numbers it needs.
When you have everything, finish.

Reply ONLY with a JSON object, one of:
{"delegate_to": "web_search_llm", "instruction": "..."}
{"delegate_to": "calculator_llm", "instruction": "..."}
{"delegate_to": "final", "answer": "..."}"""

WEB_SYSTEM = ("You are web_search_llm. Use the web_search tool to answer the "
              "instruction. Reply with a brief factual answer including the numbers.")
CALC_SYSTEM = ("You are calculator_llm. Use the calculator tool for the arithmetic "
               "in the instruction. Reply with the result and a one-line explanation.")


def run_multi_agent(task, tracker):
    manager_msgs = [
        {"role": "system", "content": MANAGER_SYSTEM},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_STEPS + 4):
        resp = client.chat.completions.create(
            model=MODEL, messages=manager_msgs, temperature=0,
            response_format={"type": "json_object"},
        )
        tracker.add_llm("ask_manager_llm", resp.usage)
        raw = resp.choices[0].message.content
        manager_msgs.append({"role": "assistant", "content": raw})

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            manager_msgs.append({"role": "user", "content": "Invalid JSON. Reply with JSON only."})
            continue

        target = decision.get("delegate_to")
        print(f"  [ask_manager_llm] -> {target}: "
              f"{decision.get('instruction') or decision.get('answer', '')[:100]}")

        if target == "final":
            return decision.get("answer", "")
        if target == "web_search_llm":
            out = run_tool_agent("web_search_llm", WEB_SYSTEM,
                                 decision["instruction"], [SEARCH_SCHEMA], tracker)
        elif target == "calculator_llm":
            out = run_tool_agent("calculator_llm", CALC_SYSTEM,
                                 decision["instruction"], [CALC_SCHEMA], tracker)
        else:
            manager_msgs.append({"role": "user", "content": "Unknown delegate_to value. Use the allowed JSON formats."})
            continue

        manager_msgs.append({"role": "user", "content": f"Result from {target}: {out}"})
    return "Stopped: max manager steps reached."


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def print_system_report(title, tracker):
    print(f"\n{title}")
    print("-" * 86)
    print(f"{'Agent':<18}{'LLM calls':>10}{'Prompt':>10}{'Completion':>12}{'Total':>10}"
          f"{'Search':>8}{'Calc':>6}")
    for agent, s in tracker.stats.items():
        print(f"{agent:<18}{s['llm_calls']:>10}{s['prompt']:>10}{s['completion']:>12}"
              f"{s['total']:>10}{s['search_calls']:>8}{s['calculator_calls']:>6}")
    t = tracker.totals()
    print("-" * 86)
    print(f"{'TOTAL':<18}{t['llm_calls']:>10}{t['prompt']:>10}{t['completion']:>12}"
          f"{t['total']:>10}{t['search_calls']:>8}{t['calculator_calls']:>6}")


def print_comparison(single, multi):
    s, m = single.totals(), multi.totals()
    print("\n" + "=" * 86)
    print("COMPARISON: SINGLE vs MULTI AGENT")
    print("=" * 86)
    print(f"{'Metric':<20}{'Single':>12}{'Multi':>12}{'Diff (M-S)':>14}{'Multi/Single':>16}")
    for label, key in [("LLM calls", "llm_calls"), ("Prompt tokens", "prompt"),
                       ("Completion tokens", "completion"), ("Total tokens", "total")]:
        ratio = m[key] / s[key] if s[key] else float("nan")
        print(f"{label:<20}{s[key]:>12}{m[key]:>12}{m[key]-s[key]:>14}{ratio:>15.2f}x")


def main():
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Please set the GROQ_API_KEY environment variable.")

    print(f"Model: {MODEL}\nTask : {TASK}\n")

    single = UsageTracker()
    print("=== SINGLE AGENT ===")
    single_answer = run_single_agent(TASK, single)
    print(f"\nAnswer: {single_answer}")

    multi = UsageTracker()
    print("\n=== MULTI AGENT ===")
    multi_answer = run_multi_agent(TASK, multi)
    print(f"\nAnswer: {multi_answer}")

    print_system_report("TOKEN USAGE - SINGLE AGENT", single)
    print_system_report("TOKEN USAGE - MULTI AGENT", multi)
    print_comparison(single, multi)

    with open("token_usage_report.json", "w") as f:
        json.dump({
            "model": MODEL, "task": TASK,
            "single_agent": {"answer": single_answer, "per_agent": single.stats,
                             "totals": single.totals()},
            "multi_agent": {"answer": multi_answer, "per_agent": multi.stats,
                            "totals": multi.totals()},
        }, f, indent=2)
    print("\nSaved report to token_usage_report.json")


if __name__ == "__main__":
    main()