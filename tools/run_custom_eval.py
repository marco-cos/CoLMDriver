#!/usr/bin/env python3
"""Convenience launcher for running custom CARLA leaderboard evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from setup_scenario_from_zip import prepare_routes_from_zip, parse_route_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=Path, help="ZIP produced by scenario_builder.")
    group.add_argument(
        "--routes-dir",
        type=Path,
        help="Existing routes directory prepared for the leaderboard.",
    )

    parser.add_argument(
        "--scenario-name",
        help="Name for the scenario (defaults to ZIP stem or routes-dir name).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation/leaderboard/data/CustomRoutes"),
        help="Where to place extracted routes when using --zip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing scenario directory if it already exists.",
    )
    parser.add_argument(
        "--results-tag",
        help="Folder tag under results/. Defaults to scenario name.",
    )
    parser.add_argument(
        "--ego-num",
        type=int,
        help="Override the number of ego vehicles (auto-detected otherwise).",
    )
    parser.add_argument("--port", type=int, default=2002, help="CARLA server port.")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        dest="tm_port",
        help="Traffic Manager port (defaults to port + 5).",
    )
    parser.add_argument(
        "--scenario-parameter",
        default="simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml",
        help="Scenario parameter YAML.",
    )
    parser.add_argument(
        "--scenarios",
        default="simulation/leaderboard/data/scenarios/no_scenarios.json",
        help="Scenario JSON to load alongside routes.",
    )
    parser.add_argument(
        "--agent",
        default="simulation/leaderboard/team_code/colmdriver_agent.py",
        help="Path to the evaluation agent Python file.",
    )
    parser.add_argument(
        "--agent-config",
        default="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
        help="Agent configuration file.",
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Route repetitions.")
    parser.add_argument("--track", default="SENSORS", help="Leaderboard track name.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Route timeout.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available.",
    )
    parser.add_argument(
        "--skip-existed",
        action="store_true",
        default=True,
        help="Skip routes that already have results (default: on).",
    )
    parser.add_argument(
        "--no-skip-existed",
        dest="skip_existed",
        action="store_false",
        help="Force rerun even if results exist.",
    )
    parser.add_argument(
        "--carla-seed", type=int, default=2000, help="Seed for CARLA world randomisation."
    )
    parser.add_argument(
        "--traffic-seed",
        type=int,
        default=2000,
        help="Seed for CARLA Traffic Manager.",
    )
    parser.add_argument(
        "--record",
        default="",
        help="Optional CARLA recording file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without running the evaluator.",
    )
    return parser.parse_args()


def append_pythonpath(env: Dict[str, str], path: Path) -> None:
    path_str = str(path)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path_str}:{current}" if current else path_str


def find_carla_egg(carla_root: Path) -> Path:
    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    eggs = sorted(dist_dir.glob("carla-*.egg"))
    if not eggs:
        raise FileNotFoundError(f"No CARLA egg found under {dist_dir}")
    py3_eggs = [egg for egg in eggs if "-py3" in egg.name]
    if py3_eggs:
        return py3_eggs[0]
    return eggs[0]


def detect_ego_routes(routes_dir: Path) -> int:
    count = 0
    for xml_path in routes_dir.rglob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            count += 1
    return count


def read_manifest(routes_dir: Path) -> tuple[Path | None, Dict[str, List[Dict[str, str]]] | None]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return manifest_path, None
    return manifest_path, data


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    scenario_summary = None
    routes_dir: Path
    scenario_name = args.scenario_name
    manifest_path: Path | None = None
    actors_manifest: Dict[str, List[Dict[str, str]]] | None = None

    if args.zip is not None:
        scenario_summary = prepare_routes_from_zip(
            zip_path=args.zip,
            scenario_name=args.scenario_name,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        routes_dir = scenario_summary["output_dir"]
        scenario_name = scenario_summary["scenario_name"]
        auto_ego_num = scenario_summary["ego_count"]
        manifest_path = scenario_summary.get("manifest_path")
        actors_manifest = scenario_summary.get("actors_manifest")
    else:
        routes_dir = args.routes_dir.expanduser().resolve()
        if not routes_dir.exists():
            raise FileNotFoundError(routes_dir)
        auto_ego_num = detect_ego_routes(routes_dir)
        if auto_ego_num == 0:
            raise RuntimeError(
                f"No ego routes located under {routes_dir}. "
                "Ensure your XML files include role=\"ego\"."
            )
        manifest_path, actors_manifest = read_manifest(routes_dir)

    ego_num = args.ego_num or auto_ego_num
    scenario_name = scenario_name or routes_dir.name
    results_tag = args.results_tag or scenario_name

    carla_root = repo_root / "carla"
    leaderboard_root = repo_root / "simulation" / "leaderboard"
    scenario_runner_root = repo_root / "simulation" / "scenario_runner"
    data_root = repo_root / "simulation" / "assets" / "v2xverse_debug"

    env = os.environ.copy()
    env["CARLA_ROOT"] = str(carla_root)
    env["LEADERBOARD_ROOT"] = str(leaderboard_root)
    env["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_root)
    env["DATA_ROOT"] = str(data_root)
    env["ROUTES"] = str(routes_dir)
    env["ROUTES_DIR"] = str(routes_dir)
    env.setdefault("COLMDRIVER_OFFLINE", "1")
    if manifest_path and manifest_path.exists():
        env["CUSTOM_ACTOR_MANIFEST"] = str(manifest_path)
        actors_root = manifest_path.parent / "actors"
        if actors_root.exists():
            env["CUSTOM_ACTOR_ROOT"] = str(actors_root)

    append_pythonpath(env, scenario_runner_root)
    append_pythonpath(env, leaderboard_root)
    append_pythonpath(env, leaderboard_root / "team_code")
    append_pythonpath(env, carla_root / "PythonAPI")
    append_pythonpath(env, carla_root / "PythonAPI" / "carla")
    append_pythonpath(env, find_carla_egg(carla_root))

    result_root = repo_root / "results" / "results_driving_custom" / results_tag
    save_path = result_root / "image"
    checkpoint_endpoint = result_root / "results.json"
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_endpoint.parent.mkdir(parents=True, exist_ok=True)

    env["RESULT_ROOT"] = str(result_root)
    env["SAVE_PATH"] = str(save_path)
    env["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint)

    tm_port = args.tm_port or (args.port + 5)

    evaluator = (
        leaderboard_root
        / "leaderboard"
        / "leaderboard_evaluator_parameter.py"
    )
    cmd: List[str] = [
        sys.executable,
        str(evaluator),
        "--routes_dir",
        str(routes_dir),
        "--ego-num",
        str(ego_num),
        "--scenario_parameter",
        str((repo_root / args.scenario_parameter).resolve()),
        "--agent",
        str((repo_root / args.agent).resolve()),
        "--agent-config",
        str((repo_root / args.agent_config).resolve()),
        "--port",
        str(args.port),
        "--trafficManagerPort",
        str(tm_port),
        "--scenarios",
        str((repo_root / args.scenarios).resolve()),
        "--repetitions",
        str(args.repetitions),
        "--track",
        args.track,
        "--checkpoint",
        str(checkpoint_endpoint),
        "--debug",
        "1" if args.debug else "0",
        "--record",
        args.record,
        "--resume",
        "1" if args.resume else "0",
        "--carlaProviderSeed",
        str(args.carla_seed),
        "--trafficManagerSeed",
        str(args.traffic_seed),
        "--skip_existed",
        "1" if args.skip_existed else "0",
        "--timeout",
        str(args.timeout),
    ]

    print("Scenario directory:", routes_dir)
    print("Ego vehicles:", ego_num)
    if scenario_summary is not None:
        print("Actors discovered:")
        for path, role, route_id, town in scenario_summary["routes"]:
            rel = path.relative_to(routes_dir)
            print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    elif actors_manifest:
        print("Actors discovered:")
        for role, entries in actors_manifest.items():
            for entry in entries:
                rel = entry.get("file", "?")
                route_id = entry.get("route_id", "?")
                town = entry.get("town", "?")
                print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    if actors_manifest:
        non_ego = {
            role: len(entries)
            for role, entries in actors_manifest.items()
            if role != "ego"
        }
        if non_ego:
            print(
                "Non-ego actor counts:",
                ", ".join(f"{role}={count}" for role, count in sorted(non_ego.items())),
            )
    if manifest_path:
        print("Actor manifest:", manifest_path)
    print("Results will be stored under:", result_root)
    print("\nCommand:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
