import json
import os
import sys
from collections import defaultdict

from tabulate import tabulate


CATEGORY_RULES = [
    ("IC", lambda r: any(key in r for key in ("ins_ss", "ins_sl", "ins_oppo", "ins_chaos"))),
    ("LM", lambda r: any(key in r for key in ("ins_sr", "ins_c", "ins_rl", "hw_merge"))),
    ("LC", lambda r: any(key in r for key in ("crosschange", "hw_c"))),
]
CATEGORY_ORDER = ["Total", "IC", "LM", "LC"]


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
        round(average(metrics["score_composed"]), 2),  # DS
        round(average(metrics["score_route"]), 2),     # RC
        round(average(metrics["score_penalty"]), 3),   # IS
        round(average(metrics["success_rate"]), 3),
        round(average(metrics["game_time"]), 2),
    ]


def analyze_results(main_path: str):
    route_dirs = sorted(
        d for d in os.listdir(main_path) if d.startswith("Interdrive_no_npc_")
    )
    if not route_dirs:
        print(f"No route directories found in {main_path}")
        return

    route_summary = []
    infractions_summary = []
    unmatched_routes = []

    category_metrics = {cat: init_metrics() for cat in CATEGORY_ORDER}
    category_metrics["Other"] = init_metrics()

    for route_dir in route_dirs:
        route_path = os.path.join(main_path, route_dir)
        ego_dirs = [d for d in os.listdir(route_path) if d.startswith("ego_vehicle")]
        if not ego_dirs:
            continue

        result_path = os.path.join(route_path, ego_dirs[0], "results.json")
        data = safe_load_json(result_path)
        if not data:
            continue

        records = data.get("_checkpoint", {}).get("records", [])
        if not records:
            continue

        record = records[-1]
        route_name = route_dir.split("Interdrive_no_npc_")[-1]
        category = classify_route(route_name)
        if category == "Other":
            unmatched_routes.append(route_name)

        score_route = record["scores"].get("score_route", 0.0)
        score_penalty = record["scores"].get("score_penalty", 0.0)
        score_composed = record["scores"].get("score_composed", 0.0)
        duration_game = record["meta"].get("duration_game", 0.0)
        duration_system = record["meta"].get("duration_system", 0.0)
        success_flag = 1 if score_composed > 99.95 else 0

        route_summary.append(
            [
                route_name,
                category if category in ("IC", "LM", "LC") else "",
                round(score_composed, 3),  # DS
                round(score_route, 3),     # RC
                round(score_penalty, 3),   # IS
                success_flag,
                round(duration_game, 2),
                round(duration_system, 2),
            ]
        )

        infractions = record.get("infractions", {})
        if infractions:
            details = {k: len(v) for k, v in infractions.items() if len(v) > 0}
            if details:
                infractions_summary.append([route_name, details])

        update_metrics(
            category_metrics["Total"],
            score_route,
            score_penalty,
            score_composed,
            success_flag,
            duration_game,
        )
        update_metrics(
            category_metrics.get(category, category_metrics["Other"]),
            score_route,
            score_penalty,
            score_composed,
            success_flag,
            duration_game,
        )

    print("\n=== Per-Route Summary ===")
    print(
        tabulate(
            route_summary,
            headers=[
                "Route",
                "Category",
                "DS",
                "RC",
                "IS",
                "SR",
                "Game Time (s)",
                "System Time (s)",
            ],
            tablefmt="github",
        )
    )

    print("\n=== Routes with Infractions ===")
    if not infractions_summary:
        print("✅ No infractions detected.")
    else:
        for route_name, details in infractions_summary:
            print(f"❌ {route_name} -> {details}")

    print("\n=== Category Summary (IC/LM/LC) ===")
    category_rows = []
    for cat in CATEGORY_ORDER:
        metrics = category_metrics.get(cat, init_metrics())
        category_rows.append([cat] + format_row(metrics))
    print(
        tabulate(
            category_rows,
            headers=[
                "Category",
                "Routes",
                "DS",
                "RC",
                "IS",
                "SR",
                "Avg Game Time (s)",
            ],
            tablefmt="github",
        )
    )

    if unmatched_routes:
        print("\n⚠️  Routes not mapped to IC/LM/LC:")
        print(", ".join(sorted(unmatched_routes)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python results_analysis2.py <results_dir>")
        sys.exit(1)

    root_path = sys.argv[1]
    if os.path.isdir(root_path):
        v2x_path = os.path.join(root_path, "v2x_final")
        if os.path.isdir(v2x_path):
            analyze_results(v2x_path)
        else:
            analyze_results(root_path)
    else:
        print(f"Invalid path: {root_path}")
