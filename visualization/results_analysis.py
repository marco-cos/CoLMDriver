import argparse
import csv
import json
import math
import os
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from tabulate import tabulate


CATEGORY_RULES = [
    ("IC", lambda r: any(key in r for key in ("ins_ss", "ins_sl", "ins_oppo", "ins_chaos"))),
    ("LM", lambda r: any(key in r for key in ("ins_sr", "ins_c", "ins_rl", "hw_merge"))),
    ("LC", lambda r: any(key in r for key in ("crosschange", "hw_c"))),
]
CATEGORY_ORDER = ["Total", "IC", "LM", "LC"]
ROUTE_PREFIXES = {
    "Interdrive_no_npc_": "No NPC",
    "Interdrive_npc_": "NPC",
}
MODE_ORDER = {"No NPC": 0, "NPC": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def classify_route(route_name: str) -> str:
    for category, predicate in CATEGORY_RULES:
        if predicate(route_name):
            return category
    return "Other"


def init_metrics():
    return {
        "score_route": [],
        "score_penalty": [],
        "score_composed": [],
        "success_rate": [],
        "game_time": [],
        "count": 0,
    }


def update_metrics(bucket, score_route, score_penalty, score_composed, success_flag, game_time):
    bucket["score_route"].append(score_route)
    bucket["score_penalty"].append(score_penalty)
    bucket["score_composed"].append(score_composed)
    bucket["success_rate"].append(success_flag)
    bucket["game_time"].append(game_time)
    bucket["count"] += 1


def average(values):
    return sum(values) / len(values) if values else 0.0


def format_row(metrics):
    return [
        metrics["count"],
        round(average(metrics["score_composed"]), 2),
        round(average(metrics["score_route"]), 2),
        round(average(metrics["score_penalty"]), 3),
        round(average(metrics["success_rate"]), 3),
        round(average(metrics["game_time"]), 2),
    ]


def build_category_metrics(entries):
    metrics = {cat: init_metrics() for cat in CATEGORY_ORDER}
    metrics["Other"] = init_metrics()
    for entry in entries:
        update_metrics(
            metrics["Total"],
            entry["score_route"],
            entry["score_penalty"],
            entry["score_composed"],
            entry["success_flag"],
            entry["duration_game"],
        )
        target_key = entry["category"] if entry["category"] in metrics else "Other"
        update_metrics(
            metrics[target_key],
            entry["score_route"],
            entry["score_penalty"],
            entry["score_composed"],
            entry["success_flag"],
            entry["duration_game"],
        )
    return metrics


def summarize_categories(entries):
    if not entries:
        return []
    metrics = build_category_metrics(entries)
    summary_rows = []
    for cat in CATEGORY_ORDER:
        metric = metrics.get(cat)
        if not metric:
            continue
        summary_rows.append(
            {
                "Category": cat,
                "Routes": metric["count"],
                "DS": average(metric["score_composed"]),
                "RC": average(metric["score_route"]),
                "IS": average(metric["score_penalty"]),
                "SR": average(metric["success_rate"]),
                "Avg Game Time (s)": average(metric["game_time"]),
            }
        )
    return summary_rows


def extract_route_index(route_name: str) -> int:
    head = route_name.split("_", 1)[0]
    if head.startswith("r"):
        number = head[1:]
        if number.isdigit():
            return int(number)
    return sys.maxsize


def route_entry_sort_key(entry):
    return (
        extract_route_index(entry["route_name"]),
        entry["route_name"].lower(),
        MODE_ORDER.get(entry["mode"], len(MODE_ORDER)),
    )


def load_ego_records(route_path: str):
    records = []
    for ego_dir in sorted(d for d in os.listdir(route_path) if d.startswith("ego_vehicle")):
        result_path = os.path.join(route_path, ego_dir, "results.json")
        data = safe_load_json(result_path)
        if not data:
            continue
        ego_records = data.get("_checkpoint", {}).get("records", [])
        if ego_records:
            records.append(ego_records[-1])
    return records


def aggregate_route_records(records):
    score_route_vals = []
    score_penalty_vals = []
    score_composed_vals = []
    success_flags = []
    duration_game_vals = []
    duration_system_vals = []
    infractions_totals = defaultdict(int)

    for record in records:
        scores = record.get("scores", {})
        meta = record.get("meta", {})

        score_route_vals.append(scores.get("score_route", 0.0))
        score_penalty_vals.append(scores.get("score_penalty", 0.0))
        score_composed_vals.append(scores.get("score_composed", 0.0))
        duration_game_vals.append(meta.get("duration_game", 0.0))
        duration_system_vals.append(meta.get("duration_system", 0.0))
        success_flags.append(1 if scores.get("score_composed", 0.0) > 99.95 else 0)

        infractions = record.get("infractions", {})
        if infractions:
            for key, events in infractions.items():
                if isinstance(events, list):
                    if events:
                        infractions_totals[key] += len(events)
                elif isinstance(events, (int, float)):
                    infractions_totals[key] += events
                elif isinstance(events, dict):
                    infractions_totals[key] += len(events)

    aggregated = {
        "score_route": average(score_route_vals),
        "score_penalty": average(score_penalty_vals),
        "score_composed": average(score_composed_vals),
        "success_flag": average(success_flags),
        "duration_game": average(duration_game_vals),
        "duration_system": average(duration_system_vals),
    }
    infractions_clean = {k: v for k, v in infractions_totals.items() if v > 0}
    if infractions_clean:
        aggregated["infractions"] = infractions_clean
    aggregated["ego_runs"] = len(records)
    return aggregated


def load_negotiation_logs(base_path: str):
    stats: Dict[str, Dict] = {}
    detailed_records: List[Dict] = []

    image_root = os.path.join(os.path.dirname(base_path), "image", os.path.basename(base_path))
    candidate_roots = []
    if os.path.isdir(image_root):
        candidate_roots.append(image_root)

    legacy_logs = os.path.join(base_path, "collected_nego_logs")
    if os.path.isdir(legacy_logs):
        candidate_roots.append(legacy_logs)

    if not candidate_roots:
        return stats, detailed_records

    def ingest(summary: Dict, negotiation: Dict, route_key: str, run_id: str, stage_key: str, event_key: str):
        content = negotiation.get("content") or []
        if not content:
            return
        rounds_len = len(content)
        speakers = {
            msg.get("id")
            for msg in content
            if isinstance(msg, dict) and msg.get("id") is not None
        }
        suggestion_flag = bool(negotiation.get("suggestion"))

        summary["count"] += 1
        summary["rounds"].append(rounds_len)
        summary["suggestions"] += 1 if suggestion_flag else 0
        summary["unique_speakers"].update(speakers)

        for key, bucket in (
            ("cons_score", "cons_scores"),
            ("safety_score", "safety_scores"),
            ("efficiency_score", "efficiency_scores"),
            ("total_score", "total_scores"),
            ("min_distance", "min_distances"),
        ):
            value = negotiation.get(key)
            if isinstance(value, (int, float)):
                summary[bucket].append(float(value))

        detailed_records.append(
            {
                "route_id": route_key,
                "run_id": run_id,
                "stage": stage_key,
                "event": event_key,
                "rounds": rounds_len,
                "unique_speakers": len(speakers),
                "suggestion": suggestion_flag,
                "cons_score": negotiation.get("cons_score"),
                "safety_score": negotiation.get("safety_score"),
                "efficiency_score": negotiation.get("efficiency_score"),
                "total_score": negotiation.get("total_score"),
                "min_distance": negotiation.get("min_distance"),
            }
        )

    for root in candidate_roots:
        for entry in os.listdir(root):
            if entry.startswith("."):
                continue

            route_path = os.path.join(root, entry)
            route_key = entry[:-5] if entry.endswith(".json") else entry

            summary = stats.setdefault(
                route_key,
                {
                    "count": 0,
                    "rounds": [],
                    "suggestions": 0,
                    "unique_speakers": set(),
                    "cons_scores": [],
                    "safety_scores": [],
                    "efficiency_scores": [],
                    "total_scores": [],
                    "min_distances": [],
                },
            )

            if os.path.isdir(route_path) and not entry.endswith(".json"):
                for run_dir in sorted(os.listdir(route_path)):
                    log_path = os.path.join(route_path, run_dir, "log", "nego.json")
                    if not os.path.isfile(log_path):
                        continue
                    data = safe_load_json(log_path)
                    if not data:
                        continue
                    for stage_key, stage_val in data.items():
                        if stage_key == "action" or not isinstance(stage_val, dict):
                            continue
                        for event_key, negotiation in stage_val.items():
                            if isinstance(negotiation, dict):
                                ingest(summary, negotiation, route_key, run_dir, stage_key, event_key)
            elif entry.endswith(".json"):
                data = safe_load_json(route_path)
                if not data:
                    continue
                for stage_key, stage_val in data.items():
                    if stage_key == "action" or not isinstance(stage_val, dict):
                        continue
                    for event_key, negotiation in stage_val.items():
                        if isinstance(negotiation, dict):
                            ingest(summary, negotiation, route_key, "", stage_key, event_key)

    for route_key, summary in stats.items():
        rounds = summary["rounds"]
        summary["total_rounds"] = sum(rounds)
        summary["avg_rounds"] = sum(rounds) / summary["count"] if summary["count"] else 0.0
        summary["median_rounds"] = statistics.median(rounds) if rounds else 0.0
        summary["max_rounds"] = max(rounds) if rounds else 0
        summary["unique_speakers"] = len(summary["unique_speakers"])
        summary["avg_consensus"] = average(summary["cons_scores"])
        summary["avg_safety"] = average(summary["safety_scores"])
        summary["avg_efficiency"] = average(summary["efficiency_scores"])
        summary["avg_total_score"] = average(summary["total_scores"])
        summary["avg_min_distance"] = average(summary["min_distances"])

    return stats, detailed_records


def analyze_results(main_path: str):
    discovered_routes = []
    for route_dir in os.listdir(main_path):
        for prefix, mode in ROUTE_PREFIXES.items():
            if route_dir.startswith(prefix):
                discovered_routes.append((mode, prefix, route_dir))
                break

    if not discovered_routes:
        return None

    route_entries = []
    infractions_summary = []
    unmatched_routes = []
    total_ego_runs = 0
    infraction_totals = defaultdict(int)
    negotiation_stats, negotiation_records = load_negotiation_logs(main_path)
    negotiation_rounds_all: List[int] = []
    negotiation_cons_scores_all: List[float] = []
    negotiation_safety_scores_all: List[float] = []
    negotiation_efficiency_scores_all: List[float] = []
    negotiation_total_scores_all: List[float] = []
    negotiation_min_distances_all: List[float] = []
    negotiation_suggestions_total = 0
    total_negotiations = 0

    for mode, prefix, route_dir in sorted(discovered_routes, key=lambda x: x[2]):
        route_path = os.path.join(main_path, route_dir)
        records = load_ego_records(route_path)
        if not records:
            continue

        agent_count = len([d for d in os.listdir(route_path) if d.startswith("ego_vehicle")])

        route_name = route_dir.replace(prefix, "", 1)
        category = classify_route(route_name)
        if category == "Other":
            unmatched_routes.append((route_name, mode))

        aggregated = aggregate_route_records(records)
        score_route = aggregated["score_route"]
        score_penalty = aggregated["score_penalty"]
        score_composed = aggregated["score_composed"]
        duration_game = aggregated["duration_game"]
        duration_system = aggregated["duration_system"]
        success_flag = aggregated["success_flag"]
        total_ego_runs += aggregated["ego_runs"]

        infractions = aggregated.get("infractions")
        if infractions:
            for key, count in infractions.items():
                infraction_totals[key] += count
            infractions_summary.append(
                {
                    "route_name": route_name,
                    "mode": mode,
                    "details": infractions,
                }
            )

        negotiation_info = negotiation_stats.get(route_dir, {})
        negotiation_count = negotiation_info.get("count", 0)
        negotiation_rounds = negotiation_info.get("rounds", [])
        negotiation_avg_rounds = negotiation_info.get("avg_rounds", 0.0)
        negotiation_median_rounds = negotiation_info.get("median_rounds", 0.0)
        negotiation_max_rounds = negotiation_info.get("max_rounds", 0)
        negotiation_suggestions = negotiation_info.get("suggestions", 0)
        negotiation_speakers = negotiation_info.get("unique_speakers", 0)
        negotiation_cons_scores = negotiation_info.get("cons_scores", [])
        negotiation_safety_scores = negotiation_info.get("safety_scores", [])
        negotiation_efficiency_scores = negotiation_info.get("efficiency_scores", [])
        negotiation_total_scores = negotiation_info.get("total_scores", [])
        negotiation_min_distances = negotiation_info.get("min_distances", [])
        negotiation_avg_consensus = average(negotiation_cons_scores)
        negotiation_avg_safety = average(negotiation_safety_scores)
        negotiation_avg_efficiency = average(negotiation_efficiency_scores)
        negotiation_avg_total = average(negotiation_total_scores)
        negotiation_avg_min_distance = average(negotiation_min_distances)

        if negotiation_count:
            total_negotiations += negotiation_count
            negotiation_rounds_all.extend(negotiation_rounds)
            negotiation_cons_scores_all.extend(negotiation_cons_scores)
            negotiation_safety_scores_all.extend(negotiation_safety_scores)
            negotiation_efficiency_scores_all.extend(negotiation_efficiency_scores)
            negotiation_total_scores_all.extend(negotiation_total_scores)
            negotiation_min_distances_all.extend(negotiation_min_distances)
            negotiation_suggestions_total += negotiation_suggestions

        route_entries.append(
            {
                "route_id": route_dir,
                "route_name": route_name,
                "mode": mode,
                "category": category,
                "score_composed": score_composed,
                "score_route": score_route,
                "score_penalty": score_penalty,
                "success_flag": success_flag,
                "duration_game": duration_game,
                "duration_system": duration_system,
                "ego_runs": aggregated["ego_runs"],
                "infractions": infractions or {},
                "agent_count": agent_count,
                "negotiations_count": negotiation_count,
                "negotiation_rounds": negotiation_rounds,
                "negotiation_avg_rounds": negotiation_avg_rounds,
                "negotiation_median_rounds": negotiation_median_rounds,
                "negotiation_max_rounds": negotiation_max_rounds,
                "negotiation_suggestions": negotiation_suggestions,
                "negotiation_speakers": negotiation_speakers,
                "negotiation_cons_scores": negotiation_cons_scores,
                "negotiation_safety_scores": negotiation_safety_scores,
                "negotiation_efficiency_scores": negotiation_efficiency_scores,
                "negotiation_total_scores": negotiation_total_scores,
                "negotiation_min_distances": negotiation_min_distances,
                "negotiation_avg_consensus": negotiation_avg_consensus,
                "negotiation_avg_safety": negotiation_avg_safety,
                "negotiation_avg_efficiency": negotiation_avg_efficiency,
                "negotiation_avg_total": negotiation_avg_total,
                "negotiation_avg_min_distance": negotiation_avg_min_distance,
            }
        )

    if not route_entries:
        return None

    npc_entries = [entry for entry in route_entries if entry["mode"] == "NPC"]
    no_npc_entries = [entry for entry in route_entries if entry["mode"] == "No NPC"]

    category_summaries = {
        "NPC Only": summarize_categories(npc_entries),
        "No NPC": summarize_categories(no_npc_entries),
        "Combined": summarize_categories(route_entries),
    }

    stats = {
        "total_route_entries": len(route_entries),
        "unique_routes": len({entry["route_name"] for entry in route_entries}),
        "npc_route_entries": len(npc_entries),
        "no_npc_route_entries": len(no_npc_entries),
        "with_infractions": len(infractions_summary),
        "unmatched_routes": len(unmatched_routes),
        "total_ego_runs": total_ego_runs,
        "avg_ds": average([entry["score_composed"] for entry in route_entries]),
        "avg_success": average([entry["success_flag"] for entry in route_entries]),
        "total_negotiations": total_negotiations,
        "routes_with_negotiations": sum(1 for entry in route_entries if entry["negotiations_count"] > 0),
        "avg_rounds_per_negotiation": (sum(negotiation_rounds_all) / total_negotiations) if total_negotiations else 0.0,
        "median_rounds_per_negotiation": statistics.median(negotiation_rounds_all) if negotiation_rounds_all else 0.0,
        "max_rounds_per_negotiation": max(negotiation_rounds_all) if negotiation_rounds_all else 0,
        "negotiation_suggestions": negotiation_suggestions_total,
        "avg_consensus": average(negotiation_cons_scores_all),
        "avg_safety": average(negotiation_safety_scores_all),
        "avg_efficiency": average(negotiation_efficiency_scores_all),
        "avg_negotiation_score": average(negotiation_total_scores_all),
        "avg_negotiation_min_distance": average(negotiation_min_distances_all),
    }

    communication_summary: Dict[str, Dict] = {
        "overall": {
            "routes": stats["total_route_entries"],
            "negotiations": stats["total_negotiations"],
            "routes_with_negotiations": stats["routes_with_negotiations"],
            "rounds": sum(negotiation_rounds_all),
            "rounds_list": negotiation_rounds_all,
            "avg_rounds": stats["avg_rounds_per_negotiation"],
            "median_rounds": stats["median_rounds_per_negotiation"],
            "max_rounds": stats["max_rounds_per_negotiation"],
            "suggestions": negotiation_suggestions_total,
            "cons_scores": negotiation_cons_scores_all,
            "avg_consensus": stats["avg_consensus"],
            "safety_scores": negotiation_safety_scores_all,
            "avg_safety": stats["avg_safety"],
            "efficiency_scores": negotiation_efficiency_scores_all,
            "avg_efficiency": stats["avg_efficiency"],
            "total_scores": negotiation_total_scores_all,
            "avg_total_score": stats["avg_negotiation_score"],
            "min_distances": negotiation_min_distances_all,
            "avg_min_distance": stats["avg_negotiation_min_distance"],
        },
        "by_mode": {},
    }

    mode_summaries: Dict[str, Dict] = {}
    for entry in route_entries:
        mode = entry["mode"]
        summary = mode_summaries.setdefault(
            mode,
            {
                "routes": 0,
                "negotiations": 0,
                "routes_with_negotiations": 0,
                "rounds_list": [],
                "suggestions": 0,
                "cons_scores": [],
                "safety_scores": [],
                "efficiency_scores": [],
                "total_scores": [],
                "min_distances": [],
            },
        )
        summary["routes"] += 1
        if entry["negotiations_count"] > 0:
            summary["negotiations"] += entry["negotiations_count"]
            summary["routes_with_negotiations"] += 1
            summary["rounds_list"].extend(entry["negotiation_rounds"])
            summary["suggestions"] += entry["negotiation_suggestions"]
            summary["cons_scores"].extend(entry["negotiation_cons_scores"])
            summary["safety_scores"].extend(entry["negotiation_safety_scores"])
            summary["efficiency_scores"].extend(entry["negotiation_efficiency_scores"])
            summary["total_scores"].extend(entry["negotiation_total_scores"])
            summary["min_distances"].extend(entry["negotiation_min_distances"])

    for mode, info in mode_summaries.items():
        rounds_list = info["rounds_list"]
        total_mode_negs = info["negotiations"]
        communication_summary["by_mode"][mode] = {
            "routes": info["routes"],
            "negotiations": total_mode_negs,
            "routes_with_negotiations": info["routes_with_negotiations"],
            "rounds": sum(rounds_list),
            "rounds_list": rounds_list,
            "avg_rounds": (sum(rounds_list) / total_mode_negs) if total_mode_negs else 0.0,
            "median_rounds": statistics.median(rounds_list) if rounds_list else 0.0,
            "max_rounds": max(rounds_list) if rounds_list else 0,
            "suggestions": info["suggestions"],
            "cons_scores": info["cons_scores"],
            "avg_consensus": average(info["cons_scores"]),
            "safety_scores": info["safety_scores"],
            "avg_safety": average(info["safety_scores"]),
            "efficiency_scores": info["efficiency_scores"],
            "avg_efficiency": average(info["efficiency_scores"]),
            "total_scores": info["total_scores"],
            "avg_total_score": average(info["total_scores"]),
            "min_distances": info["min_distances"],
            "avg_min_distance": average(info["min_distances"]),
        }

    return {
        "route_entries": route_entries,
        "infractions": infractions_summary,
        "unmatched_routes": unmatched_routes,
        "category_summaries": category_summaries,
        "stats": stats,
        "infraction_totals": dict(sorted(infraction_totals.items(), key=lambda item: item[0])),
        "communication_summary": communication_summary,
        "negotiation_records": negotiation_records,
    }


def route_entries_for_table(entries: List[Dict]) -> List[List]:
    rows = []
    for entry in sorted(entries, key=route_entry_sort_key):
        category_label = entry["category"] if entry["category"] in ("IC", "LM", "LC") else entry["category"]
        rows.append(
            [
                entry["route_name"],
                entry["mode"],
                entry.get("agent_count", 0),
                category_label,
                entry["score_composed"],
                entry["score_route"],
                entry["score_penalty"],
                entry["success_flag"],
                entry["duration_game"],
                entry["duration_system"],
                entry["ego_runs"],
                infractions_to_string(entry.get("infractions", {})),
                entry.get("negotiations_count", 0),
                entry.get("negotiation_avg_rounds", 0.0),
                entry.get("negotiation_median_rounds", 0.0),
                entry.get("negotiation_max_rounds", 0),
                entry.get("negotiation_suggestions", 0),
                entry.get("negotiation_speakers", 0),
                entry.get("negotiation_avg_consensus", 0.0),
                entry.get("negotiation_avg_safety", 0.0),
                entry.get("negotiation_avg_efficiency", 0.0),
                entry.get("negotiation_avg_total", 0.0),
                entry.get("negotiation_avg_min_distance", 0.0),
            ]
        )
    return rows


def compute_route_pairs(entries: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    pairs = defaultdict(dict)
    for entry in entries:
        pairs[entry["route_name"]][entry["mode"]] = entry
    return pairs


def infractions_to_string(infractions: Dict[str, int]) -> str:
    if not infractions:
        return ""
    return ", ".join(f"{k}: {v}" for k, v in sorted(infractions.items()))



def slugify(value: str) -> str:
    cleaned = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in value)
    cleaned = cleaned.strip('_')
    return cleaned or 'experiment'


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def render_console_report(label: str, result: Dict):
    stats = result["stats"]
    route_entries = result["route_entries"]
    infractions = result["infractions"]
    comm_summary = result.get("communication_summary", {})

    print(f"\n===== {label} =====")
    print(
        textwrap.fill(
            (
                f"Routes evaluated: {stats['total_route_entries']} "
                f"(No NPC: {stats['no_npc_route_entries']}, NPC: {stats['npc_route_entries']}). "
                f"Unique scenarios: {stats['unique_routes']}. "
                f"Aggregated ego runs: {stats['total_ego_runs']}."
            ),
            width=100,
        )
    )
    print(
        textwrap.fill(
            (
                f"Average DS: {round(stats['avg_ds'], 2)} · "
                f"Average success rate: {round(stats['avg_success'] * 100, 1)}%."
            ),
            width=100,
        )
    )

    top_routes = sorted(route_entries, key=lambda e: e["score_composed"], reverse=True)[:5]
    if top_routes:
        table = [
            [
                entry["route_name"],
                entry["mode"],
                round(entry["score_composed"], 2),
                round(entry["score_route"], 2),
                round(entry["score_penalty"], 3),
                f"{round(entry['success_flag'] * 100, 1)}%",
            ]
            for entry in top_routes
        ]
        print("\nTop 5 Routes by Driving Score")
        print(tabulate(table, headers=["Route", "Mode", "DS", "RC", "IS", "Success"], tablefmt="github"))

    route_pairs = compute_route_pairs(route_entries)
    penalty_rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        delta = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        if delta > 0.1:
            penalty_rows.append(
                [
                    route_name,
                    round(no_npc_entry["score_composed"], 2),
                    round(npc_entry["score_composed"], 2),
                    round(delta, 2),
                ]
            )

    if penalty_rows:
        penalty_rows.sort(key=lambda row: row[3], reverse=True)
        print("\nRoutes most impacted by NPC traffic")
        print(tabulate(penalty_rows[:5], headers=["Route", "No NPC DS", "NPC DS", "Delta DS"], tablefmt="github"))

    agent_summary = defaultdict(lambda: {"routes": 0, "modes": defaultdict(int), "ds": [], "negs": 0})
    for entry in route_entries:
        count = entry.get("agent_count", 0)
        summary = agent_summary[count]
        summary["routes"] += 1
        summary["modes"][entry.get("mode", "")] += 1
        summary["ds"].append(entry.get("score_composed", 0.0))
        summary["negs"] += entry.get("negotiations_count", 0)

    if agent_summary:
        rows = []
        for count in sorted(agent_summary.keys()):
            summary = agent_summary[count]
            routes = summary["routes"]
            avg_ds = average(summary["ds"])
            avg_negs = summary["negs"] / routes if routes else 0.0
            rows.append(
                [
                    count,
                    routes,
                    summary["modes"].get("No NPC", 0),
                    summary["modes"].get("NPC", 0),
                    round(avg_ds, 2),
                    round(avg_negs, 2),
                ]
            )
        print("\nAgent count summary")
        print(tabulate(rows, headers=["Agents", "Routes", "No NPC", "NPC", "Avg DS", "Avg Negotiations"], tablefmt="github"))

    overall_comm = comm_summary.get("overall", {})
    total_negotiations = overall_comm.get("negotiations", 0)
    if total_negotiations:
        avg_rounds = stats.get("avg_rounds_per_negotiation", 0.0)
        median_rounds = stats.get("median_rounds_per_negotiation", 0.0)
        avg_consensus = stats.get("avg_consensus", 0.0)
        avg_total = stats.get("avg_negotiation_score", 0.0)
        print(
            f"\nNegotiations: {total_negotiations} total "
            f"(routes with negotiations: {stats.get('routes_with_negotiations', 0)}). "
            f"Average rounds: {round(avg_rounds, 2)}, median rounds: {round(median_rounds, 2)}. "
            f"Avg consensus score: {round(avg_consensus, 2)}, avg negotiation total: {round(avg_total, 2)}."
        )
        mode_rows = []
        for mode, summary in comm_summary.get("by_mode", {}).items():
            mode_rows.append(
                [
                    mode,
                    summary["routes"],
                    summary["negotiations"],
                    summary["routes_with_negotiations"],
                    round(summary.get("avg_rounds", 0.0), 2),
                    round(summary.get("median_rounds", 0.0), 2),
                    summary.get("suggestions", 0),
                    round(summary.get("avg_consensus", 0.0), 2),
                    round(summary.get("avg_efficiency", 0.0), 2),
                    round(summary.get("avg_total_score", 0.0), 2),
                ]
            )
        if mode_rows:
            print(
                tabulate(
                    mode_rows,
                    headers=[
                        "Mode",
                        "Routes",
                        "Negotiations",
                        "Routes w/ Neg",
                        "Avg Rounds",
                        "Median",
                        "Suggestions",
                        "Avg Cons",
                        "Avg Eff",
                        "Avg Total",
                    ],
                    tablefmt="github",
                )
            )

    if infractions:
        print("\nRoutes with Infractions")
        inf_rows = [
            [item["route_name"], item["mode"], infractions_to_string(item["details"])]
            for item in infractions
        ]
        print(tabulate(inf_rows, headers=["Route", "Mode", "Details"], tablefmt="github"))
    else:
        print("\nNo infractions detected ✅")

    if result["unmatched_routes"]:
        print("\nRoutes not mapped to IC/LM/LC:")
        for route_name, mode in sorted(result["unmatched_routes"], key=lambda item: (item[0].lower(), item[1])):
            print(f"- {route_name} ({mode})")


def generate_category_charts(figures_dir: Path, category_summaries: Dict[str, List[Dict]], experiment_dir: Path):
    categories = [row["Category"] for row in category_summaries.get("Combined", []) if row["Category"] != "Total"]
    if not categories:
        for summary in category_summaries.values():
            if summary:
                categories = [row["Category"] for row in summary if row["Category"] != "Total"]
                if categories:
                    break
    modes = [name for name in ("No NPC", "NPC Only", "Combined") if category_summaries.get(name)]
    if not categories or not modes:
        return {}

    figure_paths = {}
    x = np.arange(len(categories))
    width = 0.8 / max(len(modes), 1)

    metrics = [
        ("DS", "Driving Score (DS)", False, "category_scores.png"),
        ("SR", "Success Rate (%)", True, "category_success_rate.png"),
    ]

    for metric_key, metric_label, as_percent, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, mode in enumerate(modes):
            summary_map = {row["Category"]: row for row in category_summaries.get(mode, [])}
            values = []
            for category in categories:
                value = summary_map.get(category, {}).get(metric_key, 0.0)
                if as_percent:
                    value *= 100
                values.append(value)
            offset = (idx - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=mode)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Scenario Category")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        ensure_directory(figures_dir)
        path = figures_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figure_paths[filename] = path.relative_to(experiment_dir).as_posix()

    return figure_paths


def generate_npc_gap_chart(figures_dir: Path, route_pairs: Dict[str, Dict[str, Dict]], experiment_dir: Path):
    rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        diff = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        rows.append((route_name, no_npc_entry, npc_entry, diff))

    if not rows:
        return None

    rows.sort(key=lambda item: abs(item[3]), reverse=True)
    rows = rows[:15]

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.45 * len(rows)))
    y_positions = np.arange(len(rows))
    diffs = np.array([item[3] for item in rows])
    colors = np.where(diffs >= 0, "#d62728", "#2ca02c")
    ax.barh(y_positions, diffs, color=colors, height=0.6)

    ax.text(
        0.98,
        1.02,
        "Δ (No NPC - NPC)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#222222",
    )

    for pos, diff, row in zip(y_positions, diffs, rows):
        npc_score = row[2]["score_composed"]
        no_npc_score = row[1]["score_composed"]
        offset = 2 if diff >= 0 else -2
        align = "left" if diff >= 0 else "right"
        ax.text(diff + offset, pos + 0.15, f"{diff:+.1f}", va="center", ha=align, fontsize=9, color="#222222")
        ax.text(
            diff + offset,
            pos - 0.25,
            f"No NPC: {no_npc_score:.1f} | NPC: {npc_score:.1f}",
            va="center",
            ha=align,
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Driving Score difference (No NPC - NPC)")
    ax.set_title("Routes Most Impacted by NPC Traffic")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "npc_gap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_score_distribution(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    data = defaultdict(list)
    for entry in route_entries:
        data[entry["mode"]].append(entry["score_composed"])

    available_modes = [mode for mode in ("No NPC", "NPC") if data.get(mode)]
    if not available_modes:
        available_modes = list(data.keys())
    if not available_modes:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [data[mode] for mode in available_modes]
    box = ax.boxplot(box_data, patch_artist=True)
    palette = {"No NPC": "#1f77b4", "NPC": "#d62728"}
    for patch, mode in zip(box["boxes"], available_modes):
        patch.set_facecolor(palette.get(mode, "#7f7f7f"))
        patch.set_alpha(0.6)
    for whisker in box["whiskers"]:
        whisker.set_color("#444444")
    for cap in box["caps"]:
        cap.set_color("#444444")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(2)

    ax.set_xticks(np.arange(1, len(available_modes) + 1))
    ax.set_xticklabels(available_modes)
    ax.set_ylabel("Driving Score (DS)")
    ax.set_title("Driving Score Distribution by Traffic Setting")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "score_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()

def generate_negotiation_volume_chart(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    rows = [
        (entry["route_name"], entry["mode"], entry.get("negotiations_count", 0))
        for entry in route_entries
        if entry.get("negotiations_count", 0) > 0
    ]
    if not rows:
        return None

    rows.sort(key=lambda item: item[2], reverse=True)
    rows = rows[:15]
    y_positions = np.arange(len(rows))
    counts = np.array([row[2] for row in rows])
    palette = {"No NPC": "#1f77b4", "NPC": "#ff7f0e"}
    colors = [palette.get(row[1], "#7f7f7f") for row in rows]

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.45 * len(rows)))
    ax.barh(y_positions, counts, color=colors, height=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row[0]} ({row[1]})" for row in rows])
    ax.set_xlabel("Negotiations Started")
    ax.set_title("Top Routes by Negotiation Count")
    for y, count in zip(y_positions, counts):
        ax.text(count + 0.1, y, str(int(count)), va="center", fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    handles = [Patch(facecolor=color, label=mode) for mode, color in palette.items()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "negotiations_per_route.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_negotiation_rounds_chart(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    rounds = [rounds for entry in route_entries for rounds in entry.get("negotiation_rounds", [])]
    if not rounds:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = range(1, max(rounds) + 2)
    ax.hist(rounds, bins=bins, color="#9467bd", edgecolor="#ffffff", alpha=0.85, align="left")
    ax.set_xlabel("Rounds per Negotiation")
    ax.set_ylabel("Number of Negotiations")
    ax.set_title("Negotiation Round Distribution")
    ax.set_xticks(range(1, max(rounds) + 1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "negotiation_rounds_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_agent_count_distribution(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    counts = defaultdict(lambda: {"No NPC": 0, "NPC": 0})
    for entry in route_entries:
        agent_count = entry.get("agent_count", 0)
        mode = entry.get("mode", "")
        counts[agent_count][mode] += 1

    if not counts:
        return None

    sorted_counts = sorted(counts.keys())
    modes = ["No NPC", "NPC"]
    x = np.arange(len(sorted_counts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, mode in enumerate(modes):
        values = [counts[count].get(mode, 0) for count in sorted_counts]
        offset = (idx - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in sorted_counts])
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Number of routes")
    ax.set_title("Routes by Agent Count and Traffic Setting")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "agent_count_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_agent_ds_boxplot(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    groups = defaultdict(list)
    for entry in route_entries:
        agent_count = entry.get("agent_count")
        score = entry.get("score_composed")
        if agent_count is None or score is None:
            continue
        groups[agent_count].append(score)

    if not groups:
        return None

    sorted_counts = sorted(groups.keys())
    data = [groups[count] for count in sorted_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_counts)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box["medians"]:
        median.set_color("#2f2f2f")
        median.set_linewidth(2)

    ax.set_xticks(np.arange(1, len(sorted_counts) + 1))
    ax.set_xticklabels([str(count) for count in sorted_counts])
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Driving Score (DS)")
    ax.set_title("Driving Score Distribution by Agent Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "agent_ds_boxplot.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_infractions_chart(figures_dir: Path, infractions: List[Dict], experiment_dir: Path):
    rows = []
    for item in infractions:
        total = sum(item["details"].values())
        if total <= 0:
            continue
        rows.append((item["route_name"], item["mode"], item["details"], total))

    if not rows:
        return None

    rows.sort(key=lambda item: item[3], reverse=True)
    rows = rows[:12]

    infraction_types = sorted({key for _, _, details, _ in rows for key in details})
    if not infraction_types:
        return None

    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.45 * len(rows)))
    y_positions = np.arange(len(rows))
    cumulative = np.zeros(len(rows))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(infraction_types), 1)) for i in range(len(infraction_types))]

    for idx, inf_type in enumerate(infraction_types):
        segment = np.array([details.get(inf_type, 0) for _, _, details, _ in rows])
        if not segment.any():
            continue
        ax.barh(
            y_positions,
            segment,
            left=cumulative,
            height=0.55,
            color=colors[idx],
            label=inf_type,
        )
        cumulative += segment

    for pos, (_, mode, _, total) in enumerate(rows):
        ax.text(total + 0.1, pos, f"{int(total)}", va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{route} ({mode})" for route, mode, _, _ in rows])
    ax.set_xlabel("Infraction events")
    ax.set_title("Most Frequent Infractions")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "infractions_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()

def write_csv_file(path: Path, headers: List[str], rows: List[List]):
    ensure_directory(path.parent)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def generate_figures(experiment_dir: Path, result: Dict):
    figures_dir = experiment_dir / "figures"
    ensure_directory(figures_dir)

    figure_paths = generate_category_charts(figures_dir, result["category_summaries"], experiment_dir)

    npc_gap_path = generate_npc_gap_chart(figures_dir, compute_route_pairs(result["route_entries"]), experiment_dir)
    if npc_gap_path:
        figure_paths["npc_gap.png"] = npc_gap_path

    score_dist_path = generate_score_distribution(figures_dir, result["route_entries"], experiment_dir)
    if score_dist_path:
        figure_paths["score_distribution.png"] = score_dist_path

    neg_vol_path = generate_negotiation_volume_chart(figures_dir, result["route_entries"], experiment_dir)
    if neg_vol_path:
        figure_paths["negotiations_per_route.png"] = neg_vol_path

    neg_rounds_path = generate_negotiation_rounds_chart(figures_dir, result["route_entries"], experiment_dir)
    if neg_rounds_path:
        figure_paths["negotiation_rounds_distribution.png"] = neg_rounds_path

    agent_dist_path = generate_agent_count_distribution(figures_dir, result["route_entries"], experiment_dir)
    if agent_dist_path:
        figure_paths["agent_count_distribution.png"] = agent_dist_path

    agent_ds_path = generate_agent_ds_boxplot(figures_dir, result["route_entries"], experiment_dir)
    if agent_ds_path:
        figure_paths["agent_ds_boxplot.png"] = agent_ds_path

    infractions_path = generate_infractions_chart(figures_dir, result["infractions"], experiment_dir)
    if infractions_path:
        figure_paths["infractions_breakdown.png"] = infractions_path

    return figure_paths

def create_markdown_report(label: str, result: Dict, resources: Optional[Dict] = None) -> str:
    stats = result["stats"]
    route_entries = result["route_entries"]
    category_summaries = result["category_summaries"]
    infractions = result["infractions"]

    per_route_csv = None
    category_csvs: Dict[str, str] = {}
    infractions_csv = None
    unmatched_csv = None
    negotiation_summary_csv = None
    negotiation_detailed_csv = None
    figures: Dict[str, str] = {}

    if resources:
        per_route_csv = resources.get("per_route_csv")
        category_csvs = resources.get("category_csvs", {})
        infractions_csv = resources.get("infractions_csv")
        unmatched_csv = resources.get("unmatched_csv")
        negotiation_summary_csv = resources.get("negotiation_summary_csv")
        negotiation_detailed_csv = resources.get("negotiation_detailed_csv")
        figures = resources.get("figures", {})

    lines: List[str] = []
    lines.append(f"# Experiment Report · {label}")
    lines.append("")
    lines.append("## Overview")
    lines.extend(
        [
            f"- Total route evaluations: **{stats['total_route_entries']}** (No NPC: {stats['no_npc_route_entries']}, NPC: {stats['npc_route_entries']})",
            f"- Unique route scenarios: **{stats['unique_routes']}**",
            f"- Aggregated ego runs: **{stats['total_ego_runs']}**",
            f"- Average driving score (DS): **{round(stats['avg_ds'], 2)}**",
            f"- Average success rate: **{round(stats['avg_success'] * 100, 1)}%**",
        ]
    )
    lines.append("")

    if figures:
        lines.append("## Visual Highlights")
        ordered_figures = [
            ("category_scores.png", "Driving scores by scenario category and traffic condition."),
            ("category_success_rate.png", "Success rate by scenario category and traffic condition."),
            ("npc_gap.png", "Routes where NPC traffic changes the driving score (positive values mean NPCs make the scenario harder)."),
            ("negotiations_per_route.png", "Top routes by number of negotiations started."),
            ("negotiation_rounds_distribution.png", "Distribution of negotiation lengths measured in rounds."),
            ("agent_count_distribution.png", "Routes per agent count split by traffic setting."),
            ("agent_ds_boxplot.png", "Driving score distribution grouped by agent count."),
            ("score_distribution.png", "Distribution of driving scores for NPC and no-NPC runs."),
            ("infractions_breakdown.png", "Stacked view of infractions per route; repeated colors show multiple events of the same type."),
        ]
        consumed = set()
        for filename, caption in ordered_figures:
            path = figures.get(filename)
            if not path:
                continue
            lines.append(f"![{caption}]({path})")
            lines.append("")
            consumed.add(filename)
        for filename, path in figures.items():
            if filename in consumed:
                continue
            lines.append(f"![{filename}]({path})")
            lines.append("")

    lines.append("## Per-Route Summary")
    if per_route_csv:
        lines.append(f"[Download full per-route dataset]({per_route_csv})")

    top_routes = sorted(route_entries, key=lambda e: e["score_composed"], reverse=True)[:15]
    if top_routes:
        table = [
            [
                entry["route_name"],
                entry["mode"],
                entry["category"] if entry["category"] in ("IC", "LM", "LC") else entry["category"],
                round(entry["score_composed"], 2),
                round(entry["score_route"], 2),
                round(entry["score_penalty"], 3),
                f"{round(entry['success_flag'] * 100, 1)}%",
                round(entry["duration_game"], 2),
            ]
            for entry in top_routes
        ]
        lines.append("")
        lines.append("Top 15 routes by driving score:")
        lines.append(
            tabulate(
                table,
                headers=["Route", "Mode", "Category", "DS", "RC", "IS", "Success", "Game Time (s)"],
                tablefmt="github",
            )
        )
    else:
        lines.append("No route data available.")
    lines.append("")

    route_pairs = compute_route_pairs(route_entries)
    penalty_rows = []
    benefit_rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        delta = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        record = {
            "route": route_name,
            "category": npc_entry.get("category") if npc_entry.get("category") != "Other" else no_npc_entry.get("category"),
            "no_npc": no_npc_entry["score_composed"],
            "npc": npc_entry["score_composed"],
            "delta": delta,
        }
        if delta > 0:
            penalty_rows.append(record)
        elif delta < 0:
            benefit_rows.append(record)

    penalty_rows.sort(key=lambda item: item["delta"], reverse=True)
    benefit_rows.sort(key=lambda item: item["delta"])

    if penalty_rows:
        lines.append("Routes most impacted by NPC traffic (positive delta = score drop with NPCs present):")
        penalty_table = [
            [
                row["route"],
                row["category"] if row["category"] in ("IC", "LM", "LC") else "",
                round(row["no_npc"], 2),
                round(row["npc"], 2),
                round(row["delta"], 2),
            ]
            for row in penalty_rows[:10]
        ]
        lines.append(
            tabulate(
                penalty_table,
                headers=["Route", "Category", "No NPC DS", "NPC DS", "Delta DS"],
                tablefmt="github",
            )
        )
        lines.append("")

    if benefit_rows:
        lines.append("Routes where NPC traffic improved the score (negative delta):")
        benefit_table = [
            [
                row["route"],
                row["category"] if row["category"] in ("IC", "LM", "LC") else "",
                round(row["no_npc"], 2),
                round(row["npc"], 2),
                round(row["delta"], 2),
            ]
            for row in benefit_rows[:5]
        ]
        lines.append(
            tabulate(
                benefit_table,
                headers=["Route", "Category", "No NPC DS", "NPC DS", "Delta DS"],
                tablefmt="github",
            )
        )
        lines.append("")

    lowest_routes = sorted(route_entries, key=lambda entry: entry["score_composed"])[:10]
    if lowest_routes:
        lines.append("Routes with the lowest driving scores:")
        lines.append(
            tabulate(
                [
                    [
                        entry["route_name"],
                        entry["mode"],
                        round(entry["score_composed"], 2),
                        f"{round(entry['success_flag'] * 100, 1)}%",
                        round(entry["duration_game"], 2),
                    ]
                    for entry in lowest_routes
                ],
                headers=["Route", "Mode", "DS", "Success", "Game Time (s)"],
                tablefmt="github",
            )
        )
        lines.append("")

    agent_summary = defaultdict(lambda: {"routes": 0, "modes": defaultdict(int), "ds": [], "negs": 0})
    for entry in route_entries:
        count = entry.get("agent_count", 0)
        summary = agent_summary[count]
        summary["routes"] += 1
        summary["modes"][entry.get("mode", "")] += 1
        summary["ds"].append(entry.get("score_composed", 0.0))
        summary["negs"] += entry.get("negotiations_count", 0)

    if agent_summary:
        lines.append("## Agent Composition")
        rows = []
        for count in sorted(agent_summary.keys()):
            summary = agent_summary[count]
            routes = summary["routes"]
            avg_ds = average(summary["ds"])
            avg_negs = summary["negs"] / routes if routes else 0.0
            rows.append(
                [
                    count,
                    routes,
                    summary["modes"].get("No NPC", 0),
                    summary["modes"].get("NPC", 0),
                    round(avg_ds, 2),
                    round(avg_negs, 2),
                ]
            )
        lines.append(
            tabulate(
                rows,
                headers=["Agents", "Routes", "No NPC", "NPC", "Avg DS", "Avg Negotiations"],
                tablefmt="github",
            )
        )
        lines.append("")

    comm_summary = result.get("communication_summary", {})
    overall_comm = comm_summary.get("overall", {})
    lines.append("## Communication Analysis")
    total_negs = overall_comm.get("negotiations", 0)
    if total_negs:
        avg_rounds = round(stats.get("avg_rounds_per_negotiation", 0.0), 2)
        median_rounds = round(stats.get("median_rounds_per_negotiation", 0.0), 2)
        avg_cons = round(stats.get("avg_consensus", 0.0), 2)
        avg_total = round(stats.get("avg_negotiation_score", 0.0), 2)
        avg_min = round(stats.get("avg_negotiation_min_distance", 0.0), 2)
        lines.append(
            f"- Negotiations started: **{total_negs}** across {overall_comm.get('routes_with_negotiations', 0)} routes"
        )
        lines.append(f"- Average rounds per negotiation: **{avg_rounds}** (median {median_rounds})")
        lines.append(f"- Negotiations containing suggestions: **{overall_comm.get('suggestions', 0)}**")
        lines.append(
            f"- Average consensus score: **{avg_cons}**, average negotiation total: **{avg_total}**, average min distance: **{avg_min}**"
        )
        lines.append("")
        mode_rows = []
        for mode, summary in comm_summary.get("by_mode", {}).items():
            mode_rows.append(
                [
                    mode,
                    summary["routes"],
                    summary["negotiations"],
                    summary["routes_with_negotiations"],
                    round(summary.get("avg_rounds", 0.0), 2),
                    round(summary.get("median_rounds", 0.0), 2),
                    summary.get("suggestions", 0),
                    round(summary.get("avg_consensus", 0.0), 2),
                    round(summary.get("avg_efficiency", 0.0), 2),
                    round(summary.get("avg_total_score", 0.0), 2),
                ]
            )
        if mode_rows:
            lines.append(
                tabulate(
                    mode_rows,
                    headers=[
                        "Mode",
                        "Routes",
                        "Negotiations",
                        "Routes w/ Neg",
                        "Avg Rounds",
                        "Median",
                        "Suggestions",
                        "Avg Cons",
                        "Avg Eff",
                        "Avg Total",
                    ],
                    tablefmt="github",
                )
            )
            lines.append("")
        if negotiation_summary_csv:
            lines.append(f"[Download negotiation summary]({negotiation_summary_csv})")
        if negotiation_detailed_csv:
            lines.append(f"[Download negotiation detail]({negotiation_detailed_csv})")
        lines.append("")
    else:
        lines.append("No negotiations were recorded in these runs.")
        lines.append("")

    lines.append("## Category Summaries")
    for section_name in ("No NPC", "NPC Only", "Combined"):
        summary = category_summaries.get(section_name, [])
        if not summary:
            continue
        lines.append(f"### {section_name}")
        lines.append(
            tabulate(
                [
                    [
                        row["Category"],
                        row["Routes"],
                        round(row["DS"], 2),
                        round(row["RC"], 2),
                        round(row["IS"], 3),
                        f"{round(row['SR'] * 100, 1)}%",
                        round(row["Avg Game Time (s)"], 2),
                    ]
                    for row in summary
                ],
                headers=["Category", "Routes", "DS", "RC", "IS", "Success", "Game Time (s)"],
                tablefmt="github",
            )
        )
        csv_path = category_csvs.get(section_name)
        if csv_path:
            lines.append(f"[Download CSV]({csv_path})")
        lines.append("")

    lines.append("## Infractions")
    if infractions:
        if infractions_csv:
            lines.append(f"[Download detailed infractions]({infractions_csv})")
        lines.append("")
        lines.append(
            "Each value represents the number of times that infraction occurred. "
            "Multiple counts for the same infraction indicate repeated events within the same route run."
        )
        lines.append("")
        totals = result.get("infraction_totals", {})
        if totals:
            lines.append("Total infraction events observed across all evaluations:")
            for key, count in sorted(totals.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"- **{key}**: {count}")
            lines.append("")
        for item in infractions:
            lines.append(
                f"- **{item['route_name']}** ({item['mode']}): {infractions_to_string(item['details'])}"
            )
    else:
        lines.append("No infractions recorded.")
    lines.append("")

    if result["unmatched_routes"]:
        lines.append("## Unmatched Routes")
        if unmatched_csv:
            lines.append(f"[Download unmatched routes list]({unmatched_csv})")
        lines.extend([f"- {route} ({mode})" for route, mode in sorted(result["unmatched_routes"])])
    else:
        lines.append("## Unmatched Routes")
        lines.append("All routes were categorized into IC/LM/LC.")

    lines.append("")
    return "\n".join(lines).strip() + "\n"

def write_experiment_outputs(base_dir: Path, label: str, result: Dict, *, nest: bool):
    experiment_dir = base_dir / slugify(label) if nest else base_dir
    ensure_directory(experiment_dir)

    tables_dir = experiment_dir / "tables"
    ensure_directory(tables_dir)

    route_rows = route_entries_for_table(result["route_entries"])
    route_csv_path = tables_dir / "per_route_summary.csv"
    write_csv_file(
        route_csv_path,
        [
            "Route",
            "Mode",
            "Agent Count",
            "Category",
            "DS",
            "RC",
            "IS",
            "Success Rate",
            "Game Time (s)",
            "System Time (s)",
            "Ego Runs",
            "Infractions",
            "Negotiations",
            "Avg Rounds",
            "Median Rounds",
            "Max Rounds",
            "Suggestions",
            "Unique Speakers",
            "Avg Consensus",
            "Avg Safety",
            "Avg Efficiency",
            "Avg Total Score",
            "Avg Min Distance",
        ],
        [
            [
                row[0],
                row[1],
                str(int(row[2])) if isinstance(row[2], (int, float)) else row[2],
                row[3],
                f"{row[4]:.3f}",
                f"{row[5]:.3f}",
                f"{row[6]:.4f}",
                f"{row[7]:.3f}",
                f"{row[8]:.2f}",
                f"{row[9]:.2f}",
                str(int(row[10])) if isinstance(row[10], (int, float)) else row[10],
                row[11],
                str(int(row[12])) if isinstance(row[12], (int, float)) else row[12],
                f"{row[13]:.2f}",
                f"{row[14]:.2f}",
                str(int(row[15])) if isinstance(row[15], (int, float)) else row[15],
                str(int(row[16])) if isinstance(row[16], (int, float)) else row[16],
                f"{row[17]:.2f}",
                f"{row[18]:.2f}",
                f"{row[19]:.2f}",
                f"{row[20]:.2f}",
                f"{row[21]:.2f}",
            ]
            for row in route_rows
        ],
    )

    category_csv_paths = {}
    for section_name, summary in result["category_summaries"].items():
        if not summary:
            continue
        csv_path = tables_dir / f"category_summary_{slugify(section_name)}.csv"
        write_csv_file(
            csv_path,
            ["Category", "Routes", "DS", "RC", "IS", "Success Rate", "Avg Game Time (s)"],
            [
                [
                    row["Category"],
                    row["Routes"],
                    f"{row['DS']:.3f}",
                    f"{row['RC']:.3f}",
                    f"{row['IS']:.4f}",
                    f"{row['SR']:.3f}",
                    f"{row['Avg Game Time (s)']:.2f}",
                ]
                for row in summary
            ],
        )
        category_csv_paths[section_name] = csv_path.relative_to(experiment_dir).as_posix()

    infractions_csv_path = None
    if result["infractions"]:
        infractions_rows = []
        for item in result["infractions"]:
            for key, count in item["details"].items():
                infractions_rows.append([item["route_name"], item["mode"], key, count])
        infractions_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        infractions_csv = tables_dir / "infractions.csv"
        write_csv_file(
            infractions_csv,
            ["Route", "Mode", "Infraction", "Count"],
            infractions_rows,
        )
        infractions_csv_path = infractions_csv.relative_to(experiment_dir).as_posix()

    unmatched_csv_path = None
    if result["unmatched_routes"]:
        unmatched_csv = tables_dir / "unmatched_routes.csv"
        write_csv_file(
            unmatched_csv,
            ["Route", "Mode"],
            [[route, mode] for route, mode in sorted(result["unmatched_routes"])],
        )
        unmatched_csv_path = unmatched_csv.relative_to(experiment_dir).as_posix()

    figure_paths = generate_figures(experiment_dir, result)

    negotiation_summary_csv_path = None
    if result["route_entries"]:
        negotiation_summary_csv = tables_dir / "negotiations_summary.csv"
        rows = []
        for entry in result["route_entries"]:
            rows.append(
                [
                    entry["route_name"],
                    entry["mode"],
                    entry["category"],
                    entry.get("agent_count", 0),
                    entry.get("negotiations_count", 0),
                    entry.get("negotiation_avg_rounds", 0.0),
                    entry.get("negotiation_median_rounds", 0.0),
                    entry.get("negotiation_max_rounds", 0),
                    entry.get("negotiation_suggestions", 0),
                    entry.get("negotiation_speakers", 0),
                    entry.get("negotiation_avg_consensus", 0.0),
                    entry.get("negotiation_avg_safety", 0.0),
                    entry.get("negotiation_avg_efficiency", 0.0),
                    entry.get("negotiation_avg_total", 0.0),
                    entry.get("negotiation_avg_min_distance", 0.0),
                ]
            )
        write_csv_file(
            negotiation_summary_csv,
            [
                "Route",
                "Mode",
                "Category",
                "Agent Count",
                "Negotiations",
                "Avg Rounds",
                "Median Rounds",
                "Max Rounds",
                "Suggestions",
                "Unique Speakers",
                "Avg Consensus",
                "Avg Safety",
                "Avg Efficiency",
                "Avg Total Score",
                "Avg Min Distance",
            ],
            [
                [
                    row[0],
                    row[1],
                    row[2],
                    str(int(row[3])),
                    str(int(row[4])),
                    f"{row[5]:.2f}",
                    f"{row[6]:.2f}",
                    str(int(row[7])),
                    str(int(row[8])),
                    f"{row[9]:.2f}",
                    f"{row[10]:.2f}",
                    f"{row[11]:.2f}",
                    f"{row[12]:.2f}",
                    f"{row[13]:.2f}",
                    f"{row[14]:.2f}",
                ]
                for row in rows
            ],
        )
        negotiation_summary_csv_path = negotiation_summary_csv.relative_to(experiment_dir).as_posix()

    negotiation_detailed_csv_path = None
    if result.get("negotiation_records"):
        negotiation_detailed_csv = tables_dir / "negotiations_detailed.csv"
        write_csv_file(
            negotiation_detailed_csv,
            [
                "Route",
                "Mode",
                "Run",
                "Stage",
                "Event",
                "Rounds",
                "Unique Speakers",
                "Suggestion",
                "Consensus",
                "Safety",
                "Efficiency",
                "Total Score",
                "Min Distance",
            ],
            [
                [
                    record["route_id"],
                    "No NPC" if record["route_id"].startswith("Interdrive_no_npc_") else "NPC",
                    record["run_id"],
                    record["stage"],
                    record["event"],
                    record["rounds"],
                    record["unique_speakers"],
                    int(record["suggestion"]),
                    record["cons_score"] if record["cons_score"] is not None else "",
                    record["safety_score"] if record["safety_score"] is not None else "",
                    record["efficiency_score"] if record["efficiency_score"] is not None else "",
                    record["total_score"] if record["total_score"] is not None else "",
                    record["min_distance"] if record["min_distance"] is not None else "",
                ]
                for record in result["negotiation_records"]
            ],
        )
        negotiation_detailed_csv_path = negotiation_detailed_csv.relative_to(experiment_dir).as_posix()

    resources = {
        "per_route_csv": route_csv_path.relative_to(experiment_dir).as_posix(),
        "category_csvs": category_csv_paths,
        "infractions_csv": infractions_csv_path,
        "unmatched_csv": unmatched_csv_path,
        "figures": figure_paths,
        "negotiation_summary_csv": negotiation_summary_csv_path,
        "negotiation_detailed_csv": negotiation_detailed_csv_path,
    }

    markdown_content = create_markdown_report(label, result, resources)
    markdown_path = experiment_dir / "report.md"
    markdown_path.write_text(markdown_content)

    return {
        "experiment_dir": experiment_dir,
        "markdown_path": markdown_path,
        "markdown_content": markdown_content,
        "resources": resources,
    }


def resolve_results_dir(path: str):
    if not os.path.isdir(path):
        return None
    v2x_path = os.path.join(path, "v2x_final")
    if os.path.isdir(v2x_path):
        return v2x_path
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CARLA leaderboard results for NPC/no-NPC experiments."
    )
    parser.add_argument(
        "base_path",
        help="Base directory containing results or a direct path to a v2x_final folder.",
    )
    parser.add_argument(
        "suffixes",
        nargs="*",
        help="Optional experiment suffixes to append to base_path (each will be analyzed).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        help="Directory where detailed reports (markdown, CSV, figures) will be written.",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        dest="markdown_path",
        help="If provided, aggregate markdown for all experiments into this single file.",
    )

    args = parser.parse_args()

    suffixes = args.suffixes if args.suffixes else [None]
    expected_multiple = len(suffixes) > 1
    any_valid = False
    markdown_sections: Optional[List[str]] = [] if args.markdown_path else None

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        ensure_directory(output_dir)

    for suffix in suffixes:
        target_path = args.base_path if suffix is None else os.path.join(args.base_path, suffix)
        resolved = resolve_results_dir(target_path)
        if not resolved:
            print(f"Invalid path: {target_path}")
            continue

        label = (
            suffix
            if suffix is not None
            else os.path.basename(os.path.normpath(target_path))
        )
        result = analyze_results(resolved)
        if not result:
            print(f"No completed route results found in {resolved}")
            continue

        render_console_report(label, result)

        if output_dir:
            outputs = write_experiment_outputs(output_dir, label, result, nest=expected_multiple)
            if markdown_sections is not None:
                markdown_sections.append(outputs["markdown_content"])
        elif markdown_sections is not None:
            markdown_sections.append(create_markdown_report(label, result))

        any_valid = True

    if markdown_sections is not None and markdown_sections:
        with open(args.markdown_path, "w") as outfile:
            outfile.write("\n\n---\n\n".join(markdown_sections))
            outfile.write("\n")

    if not any_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
