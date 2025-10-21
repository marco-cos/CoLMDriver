#!/usr/bin/env python3
"""Analyze negotiation logs and produce communication statistics/visualizations."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


BASE_TYPE_LABELS: Dict[str, str] = {
    "hw": "Highway",
    "ins": "Intersection",
}

SCENARIO_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
    ("ins", "ss"): "Cross: Straight-Straight",
    ("ins", "sl"): "Cross: Straight-Left",
    ("ins", "sr"): "Cross: Straight-Right",
    ("ins", "oppo"): "Cross: Opposite Lane",
    ("ins", "chaos"): "Cross: Chaos",
    ("ins", "crosschange"): "Change: Right-Straight",
    ("ins", "c"): "Change: Right-Straight",
    ("ins", "rl"): "Change: Right-Left",
    ("hw", "merge"): "Merge: Neighbor Lane",
    ("hw", "c"): "Change: Highway",
}


def scenario_type_label(base_type: str, sub_type: str) -> str:
    """Return a human-friendly description for a scenario type."""
    base_label = BASE_TYPE_LABELS.get(base_type, base_type.title())
    description = SCENARIO_DESCRIPTIONS.get((base_type, sub_type))
    if not description:
        description = sub_type.replace("_", " ").title() if sub_type else base_label
    if description.lower().startswith(base_label.lower()):
        return description
    return f"{base_label} - {description}"


def scenario_bar_label(route_id: str, agent_count: Optional[int], base_type: str, sub_type: str) -> str:
    """Compose a descriptive label for per-scenario visualizations."""
    parts = route_id.split("_")
    route_code = parts[0] if parts else route_id
    town = parts[1] if len(parts) > 1 else ""
    type_label = scenario_type_label(base_type, sub_type)
    town_segment = f"{town} - " if town else ""
    agents_text = f"{agent_count} agents" if agent_count is not None else "agents: n/a"
    return f"{route_code} ({agents_text})\n{town_segment}{type_label}"


# Source taken from eval_mode.sh full_route_list variable
_ROUTE_AGENT_SPEC = (
    '("r1_town05_ins_c:2" "r2_town05_ins_c:2" "r3_town05_ins_c:2" '
    '"r4_town06_ins_c:2" "r5_town06_ins_c:2" "r6_town07_ins_c:2" '
    '"r7_town05_ins_ss:2" "r8_town05_ins_ss:2" "r9_town06_ins_ss:2" '
    '"r10_town07_ins_ss:2" "r11_town05_ins_sl:2" "r12_town06_ins_sl:2" '
    '"r13_town05_ins_sl:2" "r14_town07_ins_sl:2" "r15_town07_ins_sl:2" '
    '"r16_town05_ins_sl:2" "r17_town05_ins_sr:2" "r18_town05_ins_sr:2" '
    '"r19_town05_ins_sr:2" "r20_town06_ins_sr:2" "r21_town07_ins_sr:2" '
    '"r22_town07_ins_sr:2" "r23_town05_ins_oppo:3" "r24_town05_ins_rl:3" '
    '"r25_town05_ins_crosschange:3" "r26_town05_ins_chaos:6" '
    '"r27_town06_hw_merge:3" "r28_town06_hw_c:6" "r29_town06_hw_merge:4" '
    '"r30_town06_hw_merge:4" "r31_town05_ins_oppo:4" "r32_town05_ins_oppo:4" '
    '"r33_town05_ins_rl:4" "r34_town05_ins_rl:4" "r35_town05_ins_crosschange:4" '
    '"r36_town05_ins_crosschange:4" "r37_town05_ins_chaos:8" '
    '"r38_town05_ins_chaos:8" "r39_town06_hw_c:8" "r40_town06_hw_c:8" '
    '"r41_town05_ins_oppo:4" "r42_town05_ins_rl:4" "r43_town05_ins_crosschange:4" '
    '"r44_town05_ins_chaos:8" "r45_town06_hw_merge:4" "r46_town06_hw_c:7")'
)


@dataclass
class ScenarioMeta:
    scenario_name: str
    route_id: str
    agent_count: Optional[int]
    base_type: str
    sub_type: str


def parse_route_agent_map(spec: str) -> Dict[str, int]:
    """Parse the route:agent mapping string into a dictionary."""
    pattern = re.compile(r'"([^":]+):(\d+)"')
    mapping = {route: int(count) for route, count in pattern.findall(spec)}
    if not mapping:
        raise ValueError("Failed to parse route agent mapping.")
    return mapping


ROUTE_AGENT_MAP = parse_route_agent_map(_ROUTE_AGENT_SPEC)


def derive_scenario_meta(scenario_name: str) -> ScenarioMeta:
    """Extract route metadata (route id, type, etc.) from the scenario filename."""
    try:
        idx = scenario_name.index("_r")
    except ValueError as exc:  # pragma: no cover - unexpected naming
        raise ValueError(f"Scenario name not in expected format: {scenario_name}") from exc
    route_id = scenario_name[idx + 1 :]
    parts = route_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Route id does not contain expected parts: {route_id}")
    base_type = parts[2]
    sub_type = "_".join(parts[3:]) if len(parts) > 3 else ""
    agent_count = ROUTE_AGENT_MAP.get(route_id)
    return ScenarioMeta(
        scenario_name=scenario_name,
        route_id=route_id,
        agent_count=agent_count,
        base_type=base_type,
        sub_type=sub_type,
    )


def iter_negotiations(json_path: Path) -> Iterable[Tuple[int, str, Dict]]:
    """Yield (timestamp, inner_key, payload) for each negotiation in the log."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key, value in data.items():
        if key == "action" or not isinstance(value, dict):
            continue
        try:
            timestamp = int(key)
        except ValueError:
            continue
        for inner_key, payload in value.items():
            if not isinstance(payload, dict):
                continue
            yield timestamp, inner_key, payload


def extract_message_features(payload: Dict) -> Dict[str, Optional[float]]:
    """Extract message-driven metrics from a negotiation payload."""
    content = payload.get("content") or []
    messages = [msg for msg in content if isinstance(msg, dict)]
    texts = [msg.get("message", "") for msg in messages if isinstance(msg.get("message"), str)]
    rounds = len(texts)
    if rounds:
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]
        avg_words = statistics.mean(word_counts)
        avg_chars = statistics.mean(char_counts)
    else:
        avg_words = 0.0
        avg_chars = 0.0
    unique_agents = {
        msg.get("id") for msg in messages if isinstance(msg.get("id"), (int, str))
    }
    return {
        "communication_rounds": rounds,
        "unique_speakers": len(unique_agents),
        "avg_words_per_message": float(avg_words),
        "avg_chars_per_message": float(avg_chars),
    }


def build_negotiation_dataframe(log_dir: Path) -> pd.DataFrame:
    """Create a DataFrame with one row per negotiation instance across all scenarios."""
    records: List[Dict] = []
    for json_file in sorted(log_dir.glob("*.json")):
        scenario_name = json_file.stem
        try:
            meta = derive_scenario_meta(scenario_name)
        except ValueError:
            continue
        for timestamp, inner_key, payload in iter_negotiations(json_file):
            features = extract_message_features(payload)
            record = {
                "scenario": scenario_name,
                "route_id": meta.route_id,
                "agent_count": meta.agent_count,
                "base_type": meta.base_type,
                "sub_type": meta.sub_type,
                "timestamp": timestamp,
                "pair_key": inner_key,
                "communication_rounds": features["communication_rounds"],
                "unique_speakers": features["unique_speakers"],
                "avg_words_per_message": features["avg_words_per_message"],
                "avg_chars_per_message": features["avg_chars_per_message"],
                "cons_score": payload.get("cons_score"),
                "safety_score": payload.get("safety_score"),
                "efficiency_score": payload.get("efficiency_score"),
                "total_score": payload.get("total_score"),
                "suggestion_present": payload.get("suggestion") is not None,
                "min_distance": payload.get("min_distance"),
            }
            records.append(record)
    if not records:
        return pd.DataFrame(
            columns=[
                "scenario",
                "route_id",
                "agent_count",
                "base_type",
                "sub_type",
                "timestamp",
                "pair_key",
                "communication_rounds",
                "unique_speakers",
                "avg_words_per_message",
                "avg_chars_per_message",
                "cons_score",
                "safety_score",
                "efficiency_score",
                "total_score",
                "suggestion_present",
                "min_distance",
            ]
        )
    df = pd.DataFrame.from_records(records)
    if df["agent_count"].isna().any():
        missing = df[df["agent_count"].isna()]["route_id"].unique()
        raise KeyError(f"Agent count missing for routes: {missing}")
    return df


def add_overall_statistics(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute overall descriptive statistics for communication rounds."""
    rounds = df["communication_rounds"]
    return {
        "total_negotiations": int(len(rounds)),
        "total_rounds": int(rounds.sum()),
        "mean_rounds": float(rounds.mean()),
        "median_rounds": float(rounds.median()),
        "std_rounds": float(rounds.std(ddof=1)) if len(rounds) > 1 else 0.0,
        "min_rounds": float(rounds.min()),
        "max_rounds": float(rounds.max()),
    }


def scenario_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate negotiation statistics per scenario."""
    summary = (
        df.groupby(
            ["scenario", "route_id", "agent_count", "base_type", "sub_type"], dropna=False
        )
        .agg(
            negotiations=("communication_rounds", "size"),
            total_rounds=("communication_rounds", "sum"),
            mean_rounds=("communication_rounds", "mean"),
            median_rounds=("communication_rounds", "median"),
            std_rounds=("communication_rounds", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            min_rounds=("communication_rounds", "min"),
            max_rounds=("communication_rounds", "max"),
        )
        .reset_index()
    )
    return summary


def grouped_summary(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    """Aggregate rounds by a grouping field (e.g., agent_count, base_type)."""
    summary = (
        df.groupby(group_field)
        .agg(
            negotiations=("communication_rounds", "size"),
            total_rounds=("communication_rounds", "sum"),
            mean_rounds=("communication_rounds", "mean"),
            median_rounds=("communication_rounds", "median"),
            std_rounds=("communication_rounds", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            min_rounds=("communication_rounds", "min"),
            max_rounds=("communication_rounds", "max"),
        )
        .reset_index()
        .sort_values(group_field)
    )
    return summary


def save_figures(df: pd.DataFrame, scenario_stats: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save visualizations highlighting communication statistics."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = df.copy()
    if "scenario_type_label" not in df.columns:
        df["scenario_type_label"] = df.apply(
            lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
        )

    scenario_stats = scenario_stats.copy()
    if "scenario_type_label" not in scenario_stats.columns:
        scenario_stats["scenario_type_label"] = scenario_stats.apply(
            lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
        )
    if "scenario_bar_label" not in scenario_stats.columns:
        scenario_stats["scenario_bar_label"] = scenario_stats.apply(
            lambda row: scenario_bar_label(
                row["route_id"], row["agent_count"], row["base_type"], row["sub_type"]
            ),
            axis=1,
        )

    # Figure 1: Average negotiation rounds per scenario with descriptive labels.
    ordered = scenario_stats.sort_values("mean_rounds", ascending=False)
    fig1_height = max(10, 0.35 * len(ordered))
    fig1, ax1 = plt.subplots(figsize=(14, fig1_height))
    sns.barplot(
        data=ordered,
        y="scenario_bar_label",
        x="mean_rounds",
        color="#1f77b4",
        ax=ax1,
    )
    ax1.set_title("Interdrive No NPC Scenarios - Average Rounds per Negotiation")
    ax1.set_xlabel("Average communication rounds per negotiation")
    ax1.set_ylabel("Scenario (route id, location, setting)")
    ax1.grid(axis="x", alpha=0.3)
    max_rounds = ordered["mean_rounds"].max() if not ordered.empty else 0
    ax1.set_xlim(0, max_rounds + 3)
    for patch, mean_value, negotiations in zip(
        ax1.patches, ordered["mean_rounds"], ordered["negotiations"]
    ):
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax1.text(
            width + 0.3,
            y,
            f"{mean_value:.1f} avg\n{int(negotiations)} negotiations",
            va="center",
            fontsize=9,
        )
    fig1.tight_layout(rect=(0, 0.02, 1, 1))
    fig1.savefig(figures_dir / "avg_rounds_per_scenario.png")
    plt.close(fig1)

    # Figure 2: Distribution of rounds by agent count.
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="agent_count", y="communication_rounds", ax=ax2)
    sns.stripplot(
        data=df,
        x="agent_count",
        y="communication_rounds",
        color="black",
        alpha=0.4,
        ax=ax2,
        dodge=True,
    )
    ax2.set_title("Rounds per Negotiation by Agent Count")
    ax2.set_xlabel("Agent count (number of agents in the scenario)")
    ax2.set_ylabel("Communication rounds per negotiation")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "rounds_by_agent_count.png")
    plt.close(fig2)

    # Figure 3: Distribution by scenario setting with descriptive labels.
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    order = (
        df.groupby("scenario_type_label")["communication_rounds"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df,
        x="scenario_type_label",
        y="communication_rounds",
        order=order,
        showfliers=False,
        ax=ax3,
    )
    ax3.set_title("Rounds per Negotiation by Scenario Setting")
    ax3.set_xlabel("Scenario setting (base - description)")
    ax3.set_ylabel("Communication rounds per negotiation")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    fig3.tight_layout()
    fig3.savefig(figures_dir / "rounds_by_scenario_setting.png")
    plt.close(fig3)

    # Figure 4: Scenario-level mean rounds vs agent count with negotiation counts.
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=scenario_stats,
        x="agent_count",
        y="mean_rounds",
        hue="scenario_type_label",
        size="negotiations",
        sizes=(60, 360),
        ax=ax4,
    )
    ax4.set_title("Average Rounds per Negotiation vs Agent Count")
    ax4.set_xlabel("Agent count (number of agents in the scenario)")
    ax4.set_ylabel("Average communication rounds per negotiation")
    legend = ax4.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Scenario type (color)\nMarker size = # negotiations",
        borderaxespad=0,
    )
    if legend:
        for text in legend.texts:
            label = text.get_text()
            if label.strip().isdigit():
                text.set_text(f"{label.strip()} negotiations")
    fig4.tight_layout()
    fig4.savefig(figures_dir / "avg_rounds_vs_agent_count.png")
    plt.close(fig4)


def store_tables(
    df: pd.DataFrame,
    scenario_stats: pd.DataFrame,
    agent_stats: pd.DataFrame,
    base_type_stats: pd.DataFrame,
    flavor_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist tabular summaries to CSV files for further inspection."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    detailed_df = df.copy()
    if "scenario_type_label" not in detailed_df.columns:
        detailed_df["scenario_type_label"] = detailed_df.apply(
            lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
        )
    detailed_df.to_csv(tables_dir / "negotiations_detailed.csv", index=False)

    scenario_summary = scenario_stats.copy()
    if "scenario_type_label" not in scenario_summary.columns:
        scenario_summary["scenario_type_label"] = scenario_summary.apply(
            lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
        )
    if "scenario_bar_label" not in scenario_summary.columns:
        scenario_summary["scenario_bar_label"] = scenario_summary.apply(
            lambda row: scenario_bar_label(
                row["route_id"], row["agent_count"], row["base_type"], row["sub_type"]
            ),
            axis=1,
        )
    scenario_summary.to_csv(tables_dir / "scenario_summary.csv", index=False)

    agent_stats.to_csv(tables_dir / "agent_count_summary.csv", index=False)

    base_summary = base_type_stats.copy()
    base_summary["base_label"] = base_summary["base_type"].map(
        lambda base: BASE_TYPE_LABELS.get(base, base.title())
    )
    base_summary.to_csv(tables_dir / "base_type_summary.csv", index=False)

    flavor_summary = flavor_stats.copy()
    if "scenario_type_label" not in flavor_summary.columns:
        flavor_summary["scenario_type_label"] = flavor_summary.apply(
            lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
        )
    flavor_summary.to_csv(tables_dir / "scenario_setting_summary.csv", index=False)


def render_report(
    overall: Dict[str, float],
    agent_stats: pd.DataFrame,
    base_type_stats: pd.DataFrame,
    flavor_stats: pd.DataFrame,
) -> str:
    """Create a textual report summarizing key statistics."""
    lines = []
    lines.append("=== Overall Communication Statistics ===")
    lines.append(
        f"Negotiations: {overall['total_negotiations']} | "
        f"Rounds: {overall['total_rounds']} | "
        f"Mean rounds: {overall['mean_rounds']:.2f} | "
        f"Median: {overall['median_rounds']:.2f} | "
        f"Std: {overall['std_rounds']:.2f} | "
        f"Range: [{overall['min_rounds']:.0f}, {overall['max_rounds']:.0f}]"
    )
    lines.append("")
    lines.append("Terminology:")
    lines.append("  Negotiations = distinct negotiation sessions between agents.")
    lines.append("  Communication rounds = number of agent messages exchanged within a negotiation.")
    lines.append("")
    lines.append("=== Breakdown by Agent Count ===")
    agent_display = agent_stats.copy()
    agent_columns = [
        "agent_count",
        "negotiations",
        "total_rounds",
        "mean_rounds",
        "median_rounds",
        "std_rounds",
        "min_rounds",
        "max_rounds",
    ]
    agent_display = agent_display[[col for col in agent_columns if col in agent_display.columns]]
    lines.append(agent_display.to_string(index=False))
    lines.append("")
    lines.append("=== Breakdown by Base Scenario Type ===")
    base_display = base_type_stats.copy()
    base_columns = [
        "base_type",
        "base_label",
        "negotiations",
        "total_rounds",
        "mean_rounds",
        "median_rounds",
        "std_rounds",
        "min_rounds",
        "max_rounds",
    ]
    base_display = base_display[[col for col in base_columns if col in base_display.columns]]
    lines.append(base_display.to_string(index=False))
    lines.append("")
    lines.append("=== Breakdown by Scenario Setting (base + subtype) ===")
    lines.append(flavors_to_string(flavor_stats))
    legend_entries: List[str] = []
    for (base, sub), desc in sorted(
        SCENARIO_DESCRIPTIONS.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        base_label = BASE_TYPE_LABELS.get(base, base.title())
        entry = f"  {base_label} - {desc}"
        if entry not in legend_entries:
            legend_entries.append(entry)
    if legend_entries:
        lines.append("")
        lines.append("Scenario Type Legend:")
        lines.extend(legend_entries)
    return "\n".join(lines)


def flavors_to_string(df: pd.DataFrame) -> str:
    """Format scenario setting stats for the textual report."""
    display_df = df.copy()
    columns = []
    if "scenario_type_label" in display_df.columns:
        columns.append("scenario_type_label")
    columns.extend(
        col for col in ["base_type", "sub_type"] if col in display_df.columns
    )
    stat_columns = [
        "negotiations",
        "total_rounds",
        "mean_rounds",
        "median_rounds",
        "std_rounds",
        "min_rounds",
        "max_rounds",
    ]
    columns.extend(col for col in stat_columns if col in display_df.columns)
    if columns:
        display_df = display_df[columns]
    return display_df.to_string(index=False)


def write_report(report_text: str, output_dir: Path) -> None:
    """Persist the text report to disk."""
    report_file = output_dir / "communication_statistics.txt"
    with report_file.open("w", encoding="utf-8") as handle:
        handle.write(report_text)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze negotiation logs and generate communication statistics."
    )
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="Directory containing scenario-level nego.json files (e.g. collected_nego_logs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for reports/figures (default: <logs_dir>/analysis).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log_dir = args.logs_dir.expanduser().resolve()
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {log_dir}")
    output_dir = args.output.expanduser().resolve() if args.output else (log_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_negotiation_dataframe(log_dir)
    if df.empty:
        raise RuntimeError(f"No negotiation records found in {log_dir}.")

    df["scenario_type_label"] = df.apply(
        lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
    )

    overall_stats = add_overall_statistics(df)
    scenario_stats = scenario_level_summary(df)
    scenario_stats["scenario_type_label"] = scenario_stats.apply(
        lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
    )
    scenario_stats["scenario_bar_label"] = scenario_stats.apply(
        lambda row: scenario_bar_label(
            row["route_id"], row["agent_count"], row["base_type"], row["sub_type"]
        ),
        axis=1,
    )
    agent_stats = grouped_summary(df, "agent_count")
    base_type_stats = grouped_summary(df, "base_type")
    base_type_stats["base_label"] = base_type_stats["base_type"].map(
        lambda base: BASE_TYPE_LABELS.get(base, base.title())
    )
    flavor_stats = (
        df.groupby(["base_type", "sub_type"])
        .agg(
            negotiations=("communication_rounds", "size"),
            total_rounds=("communication_rounds", "sum"),
            mean_rounds=("communication_rounds", "mean"),
            median_rounds=("communication_rounds", "median"),
            std_rounds=(
                "communication_rounds",
                lambda x: x.std(ddof=1) if len(x) > 1 else 0.0,
            ),
            min_rounds=("communication_rounds", "min"),
            max_rounds=("communication_rounds", "max"),
        )
        .reset_index()
        .sort_values(["base_type", "sub_type"])
    )
    flavor_stats["scenario_type_label"] = flavor_stats.apply(
        lambda row: scenario_type_label(row["base_type"], row["sub_type"]), axis=1
    )

    store_tables(df, scenario_stats, agent_stats, base_type_stats, flavor_stats, output_dir)
    save_figures(df, scenario_stats, output_dir)

    report = render_report(overall_stats, agent_stats, base_type_stats, flavor_stats)
    write_report(report, output_dir)
    print(report)
    print(f"\nDetailed outputs saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
