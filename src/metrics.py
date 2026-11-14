"""Metric helpers for retrieval and agent evaluation."""
from __future__ import annotations

import math
import statistics
from typing import Any

from .retrieval import normalize_text


def evidence_in_chunk(chunk: dict[str, Any], gold: dict[str, Any]) -> bool:
    if chunk.get("source_id") != gold.get("source_id"):
        return False
    try:
        if int(chunk.get("page")) != int(gold.get("page")):
            return False
    except (TypeError, ValueError):
        return False
    evidence = normalize_text(gold.get("evidence"))
    return bool(evidence) and evidence in normalize_text(chunk.get("text"))


def result_hit(result: dict[str, Any], gold_items: list[dict[str, Any]]) -> bool:
    return any(evidence_in_chunk(result, gold) for gold in gold_items)


def context_has_all_evidence(context_chunks: list[dict[str, Any]], gold_items: list[dict[str, Any]]) -> bool:
    if not gold_items:
        return False
    return all(any(evidence_in_chunk(chunk, gold) for chunk in context_chunks) for gold in gold_items)


def retrieval_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    hit1 = sum(1 for trace in traces if trace.get("hit1"))
    hit3 = sum(1 for trace in traces if trace.get("hit3"))
    context_hit = sum(1 for trace in traces if trace.get("context_hit"))
    reciprocal = 0.0
    for trace in traces:
        rank = trace.get("first_hit_rank")
        if rank:
            reciprocal += 1.0 / rank
    latencies = [trace.get("elapsed_ms", 0.0) for trace in traces if trace.get("elapsed_ms") is not None]
    context_chars = [trace.get("context_chars", 0) for trace in traces]
    cost = sum(float(trace.get("list_cost_usd") or 0.0) for trace in traces)
    llm_calls = sum(int(trace.get("llm_calls") or 0) for trace in traces)
    return {
        "n": n,
        "hit1": hit1,
        "hit1_rate": round(hit1 / n, 4) if n else None,
        "hit3": hit3,
        "hit3_rate": round(hit3 / n, 4) if n else None,
        "mrr": round(reciprocal / n, 4) if n else None,
        "context_hit": context_hit,
        "context_rate": round(context_hit / n, 4) if n else None,
        "avg_context_chars": round(statistics.mean(context_chars), 1) if context_chars else 0,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "llm_calls": llm_calls,
        "list_cost_usd": round(cost, 6),
    }


def agent_summary(
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mirror the starter aggregator, with explicit error accounting."""
    trace_by_id = {trace["query_id"]: trace for trace in traces}
    review_by_id = {review["query_id"]: review for review in reviews}
    output: dict[str, Any] = {
        "cases": len(cases),
        "provider_errors": 0,
        "tool_use": {"n": 0, "N": 0},
        "tool_calls_total": 0,
        "tool_calls_mean": None,
        "grounded_answer": {"0": 0, "1": 0, "2": 0, "N": 0, "mean": None},
        "citation_validity": {"n": 0, "N": 0},
        "answer_citation_coverage": {"n": 0, "N": 0},
        "abstention": {"n": 0, "N": 0},
        "false_abstention": {"n": 0, "N": 0},
        "context_grades": {"0": 0, "1": 0, "2": 0, "N": 0, "mean": None},
        "latency_ms_total": 0.0,
        "llm_calls": 0,
        "tokens": 0,
        "list_cost_usd": 0.0,
    }
    for case in cases:
        case_id = case["id"]
        trace = trace_by_id.get(case_id)
        if trace is None:
            raise ValueError(f"{case_id}: missing trace")
        if trace.get("status") == "error":
            output["provider_errors"] += 1
            continue
        review = review_by_id.get(case_id)
        if review is None:
            raise ValueError(f"{case_id}: missing review")
        citations = trace.get("citations") or []
        support = review.get("citation_support")
        if not isinstance(support, list) or len(support) != len(citations):
            raise ValueError(f"{case_id}: citation_support must match citations")
        output["tool_use"]["N"] += 1
        output["tool_use"]["n"] += int(trace.get("searches", 0) > 0)
        output["tool_calls_total"] += int(trace.get("tool_calls") or 0)
        output["citation_validity"]["n"] += sum(bool(value) for value in support)
        output["citation_validity"]["N"] += len(citations)
        if trace.get("status") == "answered":
            output["answer_citation_coverage"]["N"] += 1
            output["answer_citation_coverage"]["n"] += int(bool(citations))
        output["latency_ms_total"] += float(trace.get("latency_ms") or 0.0)
        output["llm_calls"] += int(trace.get("llm_calls") or 0)
        usage = trace.get("usage") or {}
        output["tokens"] += int(usage.get("total_tokens") or 0)
        output["list_cost_usd"] += float(usage.get("list_cost_usd") or 0.0)
        if case.get("answerable"):
            grade = review.get("answer_grade")
            if grade not in (0, 1, 2):
                raise ValueError(f"{case_id}: answer_grade must be 0, 1 or 2")
            if trace.get("status") == "abstain" and grade != 0:
                raise ValueError(f"{case_id}: abstention on an answerable question must be graded 0")
            output["grounded_answer"][str(grade)] += 1
            output["grounded_answer"]["N"] += 1
            output["false_abstention"]["N"] += 1
            output["false_abstention"]["n"] += int(trace.get("status") == "abstain")
        else:
            correct = review.get("abstention_correct")
            if type(correct) is not bool:
                raise ValueError(f"{case_id}: abstention_correct must be boolean")
            if correct and trace.get("status") != "abstain":
                raise ValueError(f"{case_id}: correct abstention label does not match status")
            output["abstention"]["N"] += 1
            output["abstention"]["n"] += int(correct)
        context_grade = review.get("context_grade")
        if context_grade is not None:
            if context_grade not in (0, 1, 2):
                raise ValueError(f"{case_id}: context_grade must be 0, 1 or 2")
            output["context_grades"][str(context_grade)] += 1
            output["context_grades"]["N"] += 1
    graded = output["grounded_answer"]
    if graded["N"]:
        graded["mean"] = round((graded["1"] + 2 * graded["2"]) / graded["N"], 4)
    contexts = output["context_grades"]
    if contexts["N"]:
        contexts["mean"] = round((contexts["1"] + 2 * contexts["2"]) / contexts["N"], 4)
    if output["tool_use"]["N"]:
        output["tool_calls_mean"] = round(output["tool_calls_total"] / output["tool_use"]["N"], 3)
    output["list_cost_usd"] = round(output["list_cost_usd"], 6)
    output["latency_ms_mean"] = (
        round(output["latency_ms_total"] / output["tool_use"]["N"], 1)
        if output["tool_use"]["N"]
        else None
    )
    return output
