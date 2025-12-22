#!/usr/bin/env python3
"""
run_path_refiner.py

A post-processing step that runs AFTER run_path_picker (picked_paths_detailed.json).

Goals:
- Choose per-vehicle spawn (start) and end points INSIDE the crop region.
- Support a small, intentional set of "behavior refinements" that modify the picked path:
    * Relative spawns: "Vehicle B spawns in front of Vehicle A" (optionally other lane).
    * Lane-change macro: "Vehicle X changes lanes into the same lane as Vehicle Y (cuts it off)".
- Use simple kinematics (constant speed on polyline) to align arrival times at conflict points
  (approximate path intersections) whenever possible.

Design philosophy (to avoid UNSAT):
- LLM output is treated as *soft preferences* (weighted penalties), not hard constraints.
- Solver is a small discrete search over a bounded set of candidate start indices + speed options.
- If constraints cannot be satisfied, we fall back to a reasonable default (first/last point in crop).

Output:
- Writes a refined picked paths JSON that is drop-in compatible with run_object_placer.py
  (it still contains top-level "picked" with per-vehicle "signature" -> "segments_detailed").
- Adds metadata under "refinement" including the constraints and chosen solution.
- Optional visualization showing crop, refined paths, start/end, inserted nodes, and conflict points.

Typical usage:
  python run_path_refiner.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --picked-paths scenario_builder_api/picked_paths_detailed.json \
    --description "..." \
    --out scenario_builder_api/picked_paths_refined.json \
    --viz --viz-out scenario_builder_api/picked_paths_refined_viz.png
"""

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Optional visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None

# For loading road network nodes
try:
    from run_object_placer import (
        load_nodes,
        build_segments_from_nodes,
        resolve_nodes_path,
    )
except Exception:
    load_nodes = None
    build_segments_from_nodes = None
    resolve_nodes_path = None


# -------------------------
# Geometry utilities
# -------------------------

@dataclass(frozen=True)
class CropBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _polyline_cumdist(pts: Sequence[Tuple[float, float]]) -> List[float]:
    out = [0.0]
    for i in range(1, len(pts)):
        out.append(out[-1] + _dist(pts[i - 1], pts[i]))
    return out


def _heading_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # world frame heading: atan2(dy, dx) in degrees
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _unit(v: Tuple[float, float]) -> Tuple[float, float]:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (1.0, 0.0)
    return (v[0] / n, v[1] / n)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _closest_point_on_segment(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    # Returns (closest_point, t in [0,1])
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = (bx - ax), (by - ay)
    denom = vx * vx + vy * vy
    if denom < 1e-9:
        return a, 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t = max(0.0, min(1.0, t))
    cp = (ax + t * vx, ay + t * vy)
    return cp, t


def _segment_segment_closest(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
) -> Tuple[float, Tuple[float, float], Tuple[float, float], float, float]:
    """
    Approx closest points between segments [a0,a1] and [b0,b1].

    Returns:
      (min_dist, pa, pb, ta, tb)
      where pa = a0 + ta*(a1-a0), pb = b0 + tb*(b1-b0), ta,tb in [0,1]
    """
    # Small, robust approximation: check endpoints projected onto the other segment.
    candidates: List[Tuple[float, Tuple[float, float], Tuple[float, float], float, float]] = []

    pb, tb = _closest_point_on_segment(a0, b0, b1)
    candidates.append((_dist(a0, pb), a0, pb, 0.0, tb))

    pb, tb = _closest_point_on_segment(a1, b0, b1)
    candidates.append((_dist(a1, pb), a1, pb, 1.0, tb))

    pa, ta = _closest_point_on_segment(b0, a0, a1)
    candidates.append((_dist(pa, b0), pa, b0, ta, 0.0))

    pa, ta = _closest_point_on_segment(b1, a0, a1)
    candidates.append((_dist(pa, b1), pa, b1, ta, 1.0))

    # Pick best
    return min(candidates, key=lambda x: x[0])


def find_conflict_between_polylines(
    p1: Sequence[Tuple[float, float]],
    p2: Sequence[Tuple[float, float]],
    dist_thresh_m: float = 3.0,
    min_angle_deg: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """
    Returns an approximate conflict if within threshold.

    First-encounter behavior:
    - Among all near-approaches within dist_thresh_m, pick the earliest encounter along the
      polylines (minimize max(s1, s2)), ignoring trivially parallel segment pairs.
    """
    if len(p1) < 2 or len(p2) < 2:
        return None

    c1 = _polyline_cumdist(p1)
    c2 = _polyline_cumdist(p2)

    # Collect near-approach candidates that are not trivially parallel
    candidates: List[Tuple[float, float, float, int, int, Tuple[float, float], Tuple[float, float], float, float]] = []
    # (s1, s2, d, i, j, pa, pb, ta, tb)

    for i in range(len(p1) - 1):
        a0, a1 = p1[i], p1[i + 1]
        va = (a1[0] - a0[0], a1[1] - a0[1])
        na = math.hypot(va[0], va[1])
        for j in range(len(p2) - 1):
            b0, b1 = p2[j], p2[j + 1]
            vb = (b1[0] - b0[0], b1[1] - b0[1])
            nb = math.hypot(vb[0], vb[1])

            d, pa, pb, ta, tb = _segment_segment_closest(a0, a1, b0, b1)
            if d > dist_thresh_m:
                continue

            ok = True
            if na > 1e-6 and nb > 1e-6:
                cosang = max(-1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)))
                ang = math.degrees(math.acos(cosang))
                if ang < float(min_angle_deg):
                    ok = False
            if not ok:
                continue

            # Arc-length along each at the closest point
            s1 = c1[i] + ta * _dist(p1[i], p1[i + 1])
            s2 = c2[j] + tb * _dist(p2[j], p2[j + 1])
            candidates.append((s1, s2, d, i, j, pa, pb, ta, tb))

    if not candidates:
        return None

    # First encounter: minimize max(s1, s2); tie-break by smaller distance
    s1, s2, d, i, j, pa, pb, ta, tb = min(candidates, key=lambda t: (max(t[0], t[1]), t[2]))

    conflict = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
    return {
        "dist_m": float(d),
        "point": {"x": float(conflict[0]), "y": float(conflict[1])},
        "s_along": {"p1_m": float(s1), "p2_m": float(s2)},
        "seg_index": {"p1": int(i), "p2": int(j)},
        "param": {"p1": float(ta), "p2": float(tb)},
    }


# -------------------------
# Path representation helpers
# -------------------------

def _segments_to_polyline_with_map(segments_detailed: List[Dict[str, Any]]) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """
    Concatenate per-segment polyline_sample into one polyline.
    Returns:
      points: [(x,y), ...]
      mapping: [(seg_i, local_pt_i), ...] for each point in points
    """
    pts: List[Tuple[float, float]] = []
    mp: List[Tuple[int, int]] = []
    for si, seg in enumerate(segments_detailed):
        pl = seg.get("polyline_sample") or []
        local: List[Tuple[float, float]] = [(float(p["x"]), float(p["y"])) for p in pl if "x" in p and "y" in p]
        if not local:
            continue
        for li, pxy in enumerate(local):
            # Avoid duplicating the seam point
            if pts and _dist(pts[-1], pxy) < 1e-6:
                continue
            pts.append(pxy)
            mp.append((si, li))
    return pts, mp


def _first_last_idx_in_crop(pts: Sequence[Tuple[float, float]], crop: CropBox) -> Optional[Tuple[int, int]]:
    inside = [i for i, (x, y) in enumerate(pts) if crop.contains(x, y)]
    if not inside:
        return None
    return inside[0], inside[-1]


def _polyline_slice(pts: Sequence[Tuple[float, float]], start_i: int, end_i: int) -> List[Tuple[float, float]]:
    start_i = max(0, min(start_i, len(pts) - 1))
    end_i = max(0, min(end_i, len(pts) - 1))
    if end_i <= start_i:
        end_i = min(len(pts) - 1, start_i + 1)
    return [tuple(map(float, pts[i])) for i in range(start_i, end_i + 1)]


def _polyline_bbox(pts: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {"xmin": float(min(xs)), "xmax": float(max(xs)), "ymin": float(min(ys)), "ymax": float(max(ys))}


def _polyline_length(pts: Sequence[Tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    return _polyline_cumdist(pts)[-1]


def _build_segment_payload_from_polyline(
    pts: Sequence[Tuple[float, float]],
    seg_template: Optional[Dict[str, Any]] = None,
    seg_id: Optional[int] = None,
) -> Dict[str, Any]:
    if len(pts) < 2:
        raise ValueError("segment polyline must have >=2 points")

    hs = _heading_deg(pts[0], pts[1])
    he = _heading_deg(pts[-2], pts[-1])
    out: Dict[str, Any] = {}
    if seg_template:
        for k in ["road_id", "section_id", "lane_id"]:
            if k in seg_template:
                out[k] = seg_template[k]
    out["seg_id"] = int(seg_id if seg_id is not None else (seg_template.get("seg_id") if seg_template else 0))
    out["length_m"] = float(_polyline_length(pts))
    out["bbox"] = _polyline_bbox(pts)
    out["start"] = {"point": {"x": float(pts[0][0]), "y": float(pts[0][1])}, "heading_deg": float(hs)}
    out["end"] = {"point": {"x": float(pts[-1][0]), "y": float(pts[-1][1])}, "heading_deg": float(he)}
    out["polyline_sample"] = [{"x": float(x), "y": float(y)} for (x, y) in pts]
    return out


def _slice_segments_detailed(
    segments_detailed: List[Dict[str, Any]],
    start_idx_global: int,
    end_idx_global: int,
) -> List[Dict[str, Any]]:
    """
    Slice original segments_detailed based on global point indices from the concatenated polyline.
    Keeps as much original segment structure as possible.
    """
    pts, mp = _segments_to_polyline_with_map(segments_detailed)
    if not pts or not mp:
        return []

    start_idx_global = max(0, min(start_idx_global, len(pts) - 1))
    end_idx_global = max(0, min(end_idx_global, len(pts) - 1))
    if end_idx_global <= start_idx_global:
        end_idx_global = min(len(pts) - 1, start_idx_global + 1)

    # Determine which segment/local indices are included
    included = list(range(start_idx_global, end_idx_global + 1))

    seg_points: Dict[int, List[Tuple[float, float]]] = {}
    for gi in included:
        si, _li = mp[gi]
        seg_points.setdefault(si, []).append(pts[gi])

    out: List[Dict[str, Any]] = []
    for si in sorted(seg_points.keys()):
        pl = seg_points[si]
        if len(pl) < 2:
            continue
        templ = segments_detailed[si]
        out.append(_build_segment_payload_from_polyline(pl, seg_template=templ, seg_id=int(templ.get("seg_id", si))))
    return out


# -------------------------
# LLM parsing (intentional constraints)
# -------------------------

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract any top-level JSON object from arbitrary text using a balanced-brace scan.
    """
    # First try whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start_search = 0
    while True:
        start = text.find("{", start_search)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start : i + 1]
                        try:
                            obj = json.loads(snippet)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start_search = start + 1


def _chat_template(tokenizer: Any) -> bool:
    return callable(getattr(tokenizer, "apply_chat_template", None))


def _llm_generate_json(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
) -> Optional[Dict[str, Any]]:
    if _chat_template(tokenizer):
        messages = [
            {"role": "system", "content": "You are a careful constraint extractor. You only output JSON."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        attn = (input_ids != tokenizer.pad_token_id).long()
        gen_kwargs = {"input_ids": input_ids, "attention_mask": attn}
        input_len = int(input_ids.shape[-1])
    else:
        enc = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = enc
        input_len = int(enc["input_ids"].shape[-1])

    # Build generation kwargs; omit temperature/top_p when not sampling to avoid warnings
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = top_p

    print(f"[DEBUG] refiner LLM: prompt_tokens={input_len}, max_new={max_new_tokens}, do_sample={do_sample}", flush=True)
    with torch.no_grad():
        out = model.generate(**gen_kwargs, **gen_config)
    print(f"[DEBUG] refiner LLM: generation complete, output_tokens={out.shape[-1] - input_len}", flush=True)

    gen = out[0][input_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return _extract_first_json_object(text)


def extract_refinement_constraints(
    description: str,
    vehicles: List[str],
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "vehicle_speeds": [{"vehicle":"Vehicle 1", "speed_class":"slow|normal|fast"}],
        "spawn_relations": [
           {"type":"ahead_of|behind_of", "a":"Vehicle X", "b":"Vehicle Y", "distance_m":10, "tolerance_m":5, "allow_other_lane":true}
        ],
        "lane_changes": [
           {"vehicle":"Vehicle X", "type":"merge_into_lane_of", "target":"Vehicle Y", "style":"cut_off|polite", "timing":"near_conflict|asap"}
        ],
        "options": {"synchronize_conflicts": true}
      }
    """
    # Intentional, narrow schema with explicit allowed values.
    vehicles_list = ", ".join(vehicles)
    prompt = f"""
You will read a driving scene description.

Your job: extract ONLY the minimal, explicitly-stated refinement requests that affect EGO vehicle spawn points,
and/or require inserting a lane-change into an ego path.

EGO vehicles are ONLY: {vehicles_list}

IMPORTANT:
- Only output fields you are confident are explicitly described.
- If a constraint is not explicitly described, OMIT it (do NOT guess).
- Do NOT invent new constraint types. Use ONLY the allowed schema below.
- If there are no refinements, return empty lists.

ALLOWED OUTPUT JSON SCHEMA (return JSON only):
{{
  "vehicle_speeds": [
    {{ "vehicle": "Vehicle 1", "speed_class": "slow" | "normal" | "fast" }}
  ],
  "spawn_relations": [
    {{
      "type": "ahead_of" | "behind_of",
      "a": "Vehicle X",
      "b": "Vehicle Y",
      "distance_m": <number, optional>,
      "tolerance_m": <number, optional>,
      "allow_other_lane": <true|false, optional>
    }}
  ],
  "lane_changes": [
    {{
      "vehicle": "Vehicle X",
      "type": "merge_into_lane_of",
      "target": "Vehicle Y",
      "style": "cut_off" | "polite",
      "timing": "near_conflict" | "asap",
      "phase": "before_intersection" | "in_intersection" | "after_intersection" | "unknown"
    }}
  ],
  "options": {{
    "synchronize_conflicts": <true|false, optional>
  }}
}}

Mapping guidance:
- If description says "slow" or "at a slow speed" -> speed_class="slow".
- If description says "fast" or "accelerates" -> speed_class="fast".
- Otherwise omit speed (we'll default to normal).
- "spawns in front of" -> spawn_relations: type="ahead_of".
- "spawns behind" -> spawn_relations: type="behind_of".
- "changes lanes into the same lane as Vehicle Y" -> lane_changes: merge_into_lane_of (style likely "cut_off" if "cut off").
- "changes lanes after the intersection" -> lane_changes: phase="after_intersection".
- "changes lanes before the intersection" -> lane_changes: phase="before_intersection".
- If lane change timing relative to intersection is not specified, use phase="unknown".

Scene description:
{description}
""".strip()

    if model is None or tokenizer is None:
        # No model provided: default to no extra constraints
        return {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    desc_lc = description.lower()

    obj = _llm_generate_json(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if not isinstance(obj, dict):
        return {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    # Filter + sanitize: keep only allowed vehicles and allowed keys
    vset = set(vehicles)
    out = {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    # options
    # Default behavior is to synchronize conflicts. We only allow the model to DISABLE
    # synchronization when the scene text explicitly asks for staggered / non-simultaneous
    # arrivals. This avoids the model "helpfully" outputting false and accidentally
    # turning off the optimization.
    opt = obj.get("options")
    if isinstance(opt, dict) and "synchronize_conflicts" in opt:
        requested = bool(opt.get("synchronize_conflicts"))
        if requested:
            out["options"]["synchronize_conflicts"] = True
        else:
            disable_triggers = [
                "do not synchronize",
                "don't synchronize",
                "not at the same time",
                "arrive at different times",
                "different times",
                "stagger",
                "one after another",
                "sequential",
                "wait for",
                "yield",
            ]
            if any(t in desc_lc for t in disable_triggers):
                out["options"]["synchronize_conflicts"] = False

    # speeds
    speeds = obj.get("vehicle_speeds", [])
    if isinstance(speeds, list):
        for s in speeds:
            if not isinstance(s, dict):
                continue
            v = s.get("vehicle")
            sc = s.get("speed_class")
            if v in vset and sc in ("slow", "normal", "fast"):
                out["vehicle_speeds"].append({"vehicle": v, "speed_class": sc})

    # spawn relations
    rels = obj.get("spawn_relations", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            typ = r.get("type")
            a = r.get("a")
            b = r.get("b")
            if typ not in ("ahead_of", "behind_of"):
                continue
            if a not in vset or b not in vset or a == b:
                continue
            dist_m = r.get("distance_m", 10.0)
            tol_m = r.get("tolerance_m", 6.0)
            allow_ol = r.get("allow_other_lane", True)
            try:
                dist_m = float(dist_m)
                tol_m = float(tol_m)
            except Exception:
                dist_m, tol_m = 10.0, 6.0
            out["spawn_relations"].append(
                {
                    "type": typ,
                    "a": a,
                    "b": b,
                    "distance_m": max(0.0, dist_m),
                    "tolerance_m": max(0.0, tol_m),
                    "allow_other_lane": bool(allow_ol),
                }
            )

    # lane changes
    lcs = obj.get("lane_changes", [])
    if isinstance(lcs, list):
        for lc in lcs:
            if not isinstance(lc, dict):
                continue
            v = lc.get("vehicle")
            typ = lc.get("type")
            tgt = lc.get("target")
            style = lc.get("style", "polite")
            timing = lc.get("timing", "near_conflict")
            if typ != "merge_into_lane_of":
                continue
            if v not in vset or tgt not in vset or v == tgt:
                continue
            if style not in ("cut_off", "polite"):
                style = "polite"
            if timing not in ("near_conflict", "asap"):
                timing = "near_conflict"
            out["lane_changes"].append({"vehicle": v, "type": typ, "target": tgt, "style": style, "timing": timing})

    # Defensive: only keep lane-change macros if the description actually mentions a lane change / merge.
    lane_triggers = [
        "lane change",
        "change lane",
        "changes lane",
        "changing lane",
        "merge into",
        "merge onto",
        "merging",
        "cuts off",
        "cut off",
        "cutoff",
        "swerves into",
        "swerving into",
    ]
    if not any(k in desc_lc for k in lane_triggers):
        out["lane_changes"] = []

    return out


# -------------------------
# Lane-change macro (best-effort)
# -------------------------

def _resample_polyline(pts: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if n <= 2 or len(pts) <= 2:
        return list(pts)
    # sample by arc-length
    cum = _polyline_cumdist(pts)
    total = cum[-1]
    if total < 1e-6:
        return [pts[0]] * n
    targets = [total * (i / (n - 1)) for i in range(n)]
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            out.append(pts[-1])
        else:
            t0, t1 = cum[j], cum[j + 1]
            alpha = 0.0 if abs(t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)
            x = pts[j][0] + alpha * (pts[j + 1][0] - pts[j][0])
            y = pts[j][1] + alpha * (pts[j + 1][1] - pts[j][1])
            out.append((x, y))
    return out


def _find_closest_idx(pts: Sequence[Tuple[float, float]], p: Tuple[float, float]) -> int:
    best_i = 0
    best_d = float("inf")
    for i, q in enumerate(pts):
        d = _dist(q, p)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def apply_merge_into_lane_of(
    mover: Dict[str, Any],
    target: Dict[str, Any],
    crop: CropBox,
    style: str = "polite",
    timing: str = "near_conflict",
    synthetic_seg_id_base: int = 900000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (updated_mover_entry, debug_info).

    Best-effort macro:
      mover follows its picked path until a cut-start point, then transitions (synthetic segment)
      to a merge point on target's refined path, then follows target's path onwards.

    We choose merge point as:
      - conflict between mover/target if exists (preferred)
      - else nearest approach within crop
    """
    m_sig = (mover.get("signature") or {}) if isinstance(mover.get("signature"), dict) else {}
    t_sig = (target.get("signature") or {}) if isinstance(target.get("signature"), dict) else {}
    m_segs = m_sig.get("segments_detailed", []) if isinstance(m_sig.get("segments_detailed", []), list) else []
    t_segs = t_sig.get("segments_detailed", []) if isinstance(t_sig.get("segments_detailed", []), list) else []

    m_pts, _ = _segments_to_polyline_with_map(m_segs)
    t_pts, _ = _segments_to_polyline_with_map(t_segs)
    if len(m_pts) < 4 or len(t_pts) < 4:
        return mover, {"applied": False, "reason": "insufficient polyline points"}

    # pick merge candidate
    conflict = find_conflict_between_polylines(m_pts, t_pts, dist_thresh_m=4.0)
    if conflict is not None:
        merge_p = (float(conflict["point"]["x"]), float(conflict["point"]["y"]))
    else:
        # brute nearest pair of vertices, preferring inside crop
        best = None
        for i, p in enumerate(m_pts):
            for j, q in enumerate(t_pts):
                if not crop.contains(q[0], q[1]):
                    continue
                d = _dist(p, q)
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            return mover, {"applied": False, "reason": "no merge candidate in crop"}
        _d, _i, j = best
        merge_p = t_pts[j]

    merge_idx_t = _find_closest_idx(t_pts, merge_p)

    # gap (cut-off means tighter)
    gap_m = 5.0 if style == "cut_off" else 10.0
    # move forward along target by gap
    t_cum = _polyline_cumdist(t_pts)
    desired_s = min(t_cum[-1], t_cum[merge_idx_t] + gap_m)
    # find idx at desired_s
    j = merge_idx_t
    while j + 1 < len(t_cum) and t_cum[j] < desired_s:
        j += 1
    merge_idx_t2 = j
    merge_p2 = t_pts[merge_idx_t2]

    # choose cut start on mover some distance before reaching a point closest to merge
    m_idx_near = _find_closest_idx(m_pts, merge_p2)
    m_cum = _polyline_cumdist(m_pts)
    back_m = 12.0  # start lane change ~12m before merge vicinity
    desired_m_s = max(0.0, m_cum[m_idx_near] - back_m)
    i = m_idx_near
    while i - 1 >= 0 and m_cum[i] > desired_m_s:
        i -= 1
    cut_idx_m = i
    cut_p = m_pts[cut_idx_m]

    # build synthetic transition polyline (simple cubic-like control)
    fwd_m = _unit((m_pts[min(cut_idx_m + 1, len(m_pts) - 1)][0] - cut_p[0], m_pts[min(cut_idx_m + 1, len(m_pts) - 1)][1] - cut_p[1]))
    fwd_t = _unit((merge_p2[0] - t_pts[max(0, merge_idx_t2 - 1)][0], merge_p2[1] - t_pts[max(0, merge_idx_t2 - 1)][1]))
    d = _dist(cut_p, merge_p2)
    ctrl1 = (cut_p[0] + fwd_m[0] * d * 0.33, cut_p[1] + fwd_m[1] * d * 0.33)
    ctrl2 = (merge_p2[0] - fwd_t[0] * d * 0.33, merge_p2[1] - fwd_t[1] * d * 0.33)

    def bezier(t: float) -> Tuple[float, float]:
        # cubic Bezier: P0=cut, P1=ctrl1, P2=ctrl2, P3=merge
        x = (1 - t) ** 3 * cut_p[0] + 3 * (1 - t) ** 2 * t * ctrl1[0] + 3 * (1 - t) * t ** 2 * ctrl2[0] + t ** 3 * merge_p2[0]
        y = (1 - t) ** 3 * cut_p[1] + 3 * (1 - t) ** 2 * t * ctrl1[1] + 3 * (1 - t) * t ** 2 * ctrl2[1] + t ** 3 * merge_p2[1]
        return (x, y)

    trans = [bezier(k / 19.0) for k in range(20)]
    trans = _resample_polyline(trans, 12)

    # Build new per-segment structure: prefix from mover, synthetic, suffix from target
    prefix_pts = m_pts[: cut_idx_m + 1]
    suffix_pts = t_pts[merge_idx_t2:]


    # Build segments_detailed for new path: keep mover segments count low -> single segment each section
    new_segs: List[Dict[str, Any]] = []
    # prefix as one segment
    if len(prefix_pts) >= 2:
        new_segs.append(_build_segment_payload_from_polyline(_resample_polyline(prefix_pts, 10), seg_template=(m_segs[0] if m_segs else None), seg_id=synthetic_seg_id_base + 1))
    # transition segment
    new_segs.append(_build_segment_payload_from_polyline(trans, seg_template=(t_segs[0] if t_segs else None), seg_id=synthetic_seg_id_base + 2))
    # suffix as one segment
    if len(suffix_pts) >= 2:
        new_segs.append(_build_segment_payload_from_polyline(_resample_polyline(suffix_pts, 10), seg_template=(t_segs[-1] if t_segs else None), seg_id=synthetic_seg_id_base + 3))

    # Build updated signature
    new_sig = dict(m_sig)
    new_sig["segments_detailed"] = new_segs
    new_sig["segment_ids"] = [int(s.get("seg_id", 0)) for s in new_segs]
    new_sig["num_segments"] = int(len(new_segs))
    new_sig["length_m"] = float(sum(float(s.get("length_m", 0.0)) for s in new_segs))

    # entry/exit points
    if new_segs:
        new_sig["entry"] = dict(new_sig.get("entry", {}))
        new_sig["entry"]["point"] = {"x": float(new_segs[0]["start"]["point"]["x"]), "y": float(new_segs[0]["start"]["point"]["y"])}
        new_sig["exit"] = dict(new_sig.get("exit", {}))
        new_sig["exit"]["point"] = {"x": float(new_segs[-1]["end"]["point"]["x"]), "y": float(new_segs[-1]["end"]["point"]["y"])}

    mover2 = dict(mover)
    mover2["signature_original"] = mover.get("signature")
    mover2["signature"] = new_sig
    mover2["name_refined"] = f"{mover.get('name','')}__merge_into_{target.get('vehicle','')}"
    return mover2, {
        "applied": True,
        "style": style,
        "timing": timing,
        "cut_start_point": {"x": float(cut_p[0]), "y": float(cut_p[1])},
        "merge_point": {"x": float(merge_p2[0]), "y": float(merge_p2[1])},
    }


# -------------------------
# Solver (soft-CSP)
# -------------------------

def _speed_class_to_mps(speed_class: str) -> float:
    return {"slow": 5.0, "normal": 8.0, "fast": 12.0}.get(speed_class, 8.0)


def _candidate_speeds(base: float) -> List[float]:
    # Small discrete set around base
    opts = sorted(set([base, base - 2.0, base - 1.0, base + 1.0, base + 2.0]))
    out = [x for x in opts if 2.0 <= x <= 20.0]
    return out


def _default_start_end_in_crop(segments_detailed: List[Dict[str, Any]], crop: CropBox) -> Optional[Tuple[int, int, List[Tuple[float, float]]]]:
    pts, _ = _segments_to_polyline_with_map(segments_detailed)
    if not pts:
        return None
    se = _first_last_idx_in_crop(pts, crop)
    if se is None:
        return None
    s, e = se
    return s, e, pts


def _forward_axis_at(pts: Sequence[Tuple[float, float]], idx: int) -> Tuple[float, float]:
    j = min(len(pts) - 1, max(0, idx))
    k = min(len(pts) - 1, j + 1)
    if k == j and j - 1 >= 0:
        k = j
        j = j - 1
    return _unit((pts[k][0] - pts[j][0], pts[k][1] - pts[j][1]))


def _eval_spawn_relation(
    rel: Dict[str, Any],
    spawn_xy: Dict[str, Tuple[float, float]],
    forward_axis: Dict[str, Tuple[float, float]],
) -> float:
    """
    Penalty for violating a spawn relation.
    """
    a = rel["a"]
    b = rel["b"]
    typ = rel["type"]
    dist_m = float(rel.get("distance_m", 10.0))
    tol = float(rel.get("tolerance_m", 6.0))
    allow_ol = bool(rel.get("allow_other_lane", True))

    pa = spawn_xy[a]
    pb = spawn_xy[b]
    f = forward_axis[b]
    delta = (pa[0] - pb[0], pa[1] - pb[1])
    along = _dot(delta, f)
    lateral = abs(_cross2(delta, f))

    # If other lane not allowed, penalize lateral strongly beyond ~3.5m
    if not allow_ol and lateral > 3.5:
        return 1000.0 + (lateral - 3.5) ** 2

    desired = dist_m if typ == "ahead_of" else -dist_m
    lo = desired - tol
    hi = desired + tol
    if along < lo:
        return (lo - along) ** 2
    if along > hi:
        return (along - hi) ** 2
    return 0.0


def refine_spawn_and_speeds_soft_csp(
    per_vehicle: Dict[str, Dict[str, Any]],
    crop: CropBox,
    constraints: Dict[str, Any],
    conflict_dist_thresh_m: float = 3.0,
) -> Dict[str, Any]:
    """
    per_vehicle[veh] must include:
      - "segments_detailed": list
    Returns solution dict with chosen start/end indices and speeds.
    """
    vehicles = list(per_vehicle.keys())

    # Build base polylines and default start/end indices (in crop)
    base = {}
    for v in vehicles:
        segs = per_vehicle[v]["segments_detailed"]
        d = _default_start_end_in_crop(segs, crop)
        if d is None:
            # fallback: use entire polyline
            pts, _ = _segments_to_polyline_with_map(segs)
            if len(pts) < 2:
                raise SystemExit(f"[ERROR] Vehicle {v} has no polyline points.")
            base[v] = {"pts": pts, "start": 0, "end": len(pts) - 1}
        else:
            s, e, pts = d
            base[v] = {"pts": pts, "start": s, "end": e}

    # Speeds
    base_speed = {v: 8.0 for v in vehicles}
    for sp in constraints.get("vehicle_speeds", []):
        v = sp.get("vehicle")
        sc = sp.get("speed_class")
        if v in base_speed and sc in ("slow", "normal", "fast"):
            base_speed[v] = _speed_class_to_mps(sc)

    # Conflicts: compute for every pair
    conflicts = []
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            va, vb = vehicles[i], vehicles[j]
            ca = _polyline_slice(base[va]["pts"], base[va]["start"], base[va]["end"])
            cb = _polyline_slice(base[vb]["pts"], base[vb]["start"], base[vb]["end"])
            conf = find_conflict_between_polylines(ca, cb, dist_thresh_m=conflict_dist_thresh_m)
            if conf is not None:
                conflicts.append({"a": va, "b": vb, "conf": conf})

    # Candidate start indices (bounded around default)
    candidates_start = {}
    for v in vehicles:
        pts = base[v]["pts"]
        s0 = base[v]["start"]
        e0 = base[v]["end"]
        # allowable starts: inside crop, and keep at least 2 points before end
        inside = [i for i, (x, y) in enumerate(pts) if crop.contains(x, y)]
        if not inside:
            inside = list(range(len(pts)))
        # bound by window: within +/- 25m of default start in arc-length
        cum = _polyline_cumdist(pts)
        window_m = 25.0
        good = []
        for i in inside:
            if i >= e0:
                continue
            if abs(cum[i] - cum[s0]) <= window_m:
                good.append(i)
        if not good:
            good = [s0]
        # prune to <=12 candidates (spread)
        good = sorted(set(good))
        if len(good) > 12:
            step = max(1, len(good) // 12)
            good = good[::step][:12]
        candidates_start[v] = good

    # Candidate end indices (best effort).
    # We prefer to keep ends inside crop, but allow a small margin beyond the crop to avoid
    # truncating intended maneuvers right at the boundary (softly penalized in the objective).
    end_outside_margin_m = float(constraints.get("options", {}).get("end_outside_crop_margin_m", 0.0))
    candidates_end: Dict[str, List[int]] = {}

    def _point_outside_cost(x: float, y: float) -> float:
        # 0 inside crop; positive squared distance to the nearest crop edge otherwise.
        dx = 0.0
        if x < crop.xmin:
            dx = crop.xmin - x
        elif x > crop.xmax:
            dx = x - crop.xmax
        dy = 0.0
        if y < crop.ymin:
            dy = crop.ymin - y
        elif y > crop.ymax:
            dy = y - crop.ymax
        return dx * dx + dy * dy

    crop_end = CropBox(
        xmin=crop.xmin - end_outside_margin_m,
        xmax=crop.xmax + end_outside_margin_m,
        ymin=crop.ymin - end_outside_margin_m,
        ymax=crop.ymax + end_outside_margin_m,
    )

    for v in vehicles:
        pts = base[v]["pts"]
        s0 = base[v]["start"]
        e0 = base[v]["end"]
        cum = _polyline_cumdist(pts)

        inside_end = [i for i, (x, y) in enumerate(pts) if crop_end.contains(x, y)]
        if not inside_end:
            inside_end = list(range(len(pts)))

        window_m = 35.0
        good = []
        for i in inside_end:
            if i <= s0 + 1:
                continue
            if abs(cum[i] - cum[e0]) <= window_m and i >= e0:
                good.append(i)

        if not good:
            good = [e0]

        good = sorted(set(good))
        if len(good) > 8:
            step = max(1, len(good) // 8)
            good = good[::step][:8]
        candidates_end[v] = good

    speed_opts = {v: _candidate_speeds(base_speed[v]) for v in vehicles}

    # Adaptively reduce candidate counts if there are many vehicles to avoid combinatorial explosion
    # Target: keep total search space under ~500k
    n_vehicles = len(vehicles)
    if n_vehicles >= 4:
        # Reduce candidates for larger vehicle counts
        max_starts = 4 if n_vehicles >= 5 else 6
        max_ends = 3 if n_vehicles >= 5 else 4
        max_speeds = 2 if n_vehicles >= 5 else 3
        print(f"[DEBUG] refiner CSP: Reducing candidates due to {n_vehicles} vehicles (max_starts={max_starts}, max_ends={max_ends}, max_speeds={max_speeds})", flush=True)
        
        for v in vehicles:
            if len(candidates_start[v]) > max_starts:
                step = max(1, len(candidates_start[v]) // max_starts)
                candidates_start[v] = candidates_start[v][::step][:max_starts]
            if len(candidates_end[v]) > max_ends:
                step = max(1, len(candidates_end[v]) // max_ends)
                candidates_end[v] = candidates_end[v][::step][:max_ends]
            if len(speed_opts[v]) > max_speeds:
                # Keep base speed and closest alternatives
                base_spd = base_speed[v]
                sorted_by_dist = sorted(speed_opts[v], key=lambda s: abs(s - base_spd))
                speed_opts[v] = sorted_by_dist[:max_speeds]

    # Objective weights
    W_SYNC = 10.0
    W_SPAWN_REL = 4.0
    W_START_SHIFT = 0.2
    W_SPEED_SHIFT = 0.2
    W_SPAWN_SEP = 6.0  # soft penalty if spawns are too close
    MIN_SPAWN_SEP_M = 8.0  # Increased from 4.0 to account for vehicle length + CARLA safety margin

    spawn_rels = constraints.get("spawn_relations", []) if isinstance(constraints.get("spawn_relations", []), list) else []
    sync = bool(((constraints.get("options") or {}).get("synchronize_conflicts", True)))

    # Precompute distances to conflict points for each start choice (approx)
    # We'll compute t_to_conf = (s_conf - s_start)/speed, but s_conf from sliced polyline.
    # For consistency, we define conflicts on the default-sliced polylines and approximate with indices in those slices.
    # This is good enough for the heuristic.
    conflict_info = []
    if sync and conflicts:
        for c in conflicts:
            a, b = c["a"], c["b"]
            # conflict s along the sliced polylines (not full)
            sa = float(c["conf"]["s_along"]["p1_m"])
            sb = float(c["conf"]["s_along"]["p2_m"])
            conflict_info.append((a, b, sa, sb, c["conf"]["point"]))

    def score(assign_start: Dict[str, int], assign_speed: Dict[str, float], assign_end: Dict[str, int]) -> Tuple[float, Dict[str, Any]]:
        # spawn xy + forward axes
        spawn_xy = {}
        fwd = {}
        for v in vehicles:
            pts = base[v]["pts"]
            si = assign_start[v]
            spawn_xy[v] = pts[si]
            fwd[v] = _forward_axis_at(pts, si)

        total = 0.0

        # spawn relations
        for rel in spawn_rels:
            if rel.get("a") in spawn_xy and rel.get("b") in spawn_xy:
                total += W_SPAWN_REL * _eval_spawn_relation(rel, spawn_xy, fwd)

        # discourage overlapping spawns (softly)
        spawn_sep_debug = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                va, vb = vehicles[i], vehicles[j]
                pa, pb = spawn_xy[va], spawn_xy[vb]
                d = _dist(pa, pb)
                if d < MIN_SPAWN_SEP_M:
                    total += W_SPAWN_SEP * (MIN_SPAWN_SEP_M - d) ** 2
                    spawn_sep_debug.append({"a": va, "b": vb, "dist_m": d})

        # conflict sync
        conflict_points = []
        if sync and conflict_info:
            for (a, b, sa, sb, pconf) in conflict_info:
                pa = base[a]["pts"]
                pb = base[b]["pts"]
                ca = _polyline_cumdist(pa)
                cb = _polyline_cumdist(pb)
                # translate sconf by shift from default start in full cumdist
                shift_a = ca[assign_start[a]] - ca[base[a]["start"]]
                shift_b = cb[assign_start[b]] - cb[base[b]["start"]]
                da = max(0.0, sa - shift_a)
                db = max(0.0, sb - shift_b)
                ta = da / max(0.5, assign_speed[a])
                tb = db / max(0.5, assign_speed[b])
                total += W_SYNC * (ta - tb) ** 2
                conflict_points.append({"a": a, "b": b, "point": pconf, "t": {a: ta, b: tb}})

        # prefer small shifts
        for v in vehicles:
            pts = base[v]["pts"]
            cum = _polyline_cumdist(pts)
            ds = abs(cum[assign_start[v]] - cum[base[v]["start"]])
            total += W_START_SHIFT * ds
            total += W_SPEED_SHIFT * (assign_speed[v] - base_speed[v]) ** 2

        dbg = {
            "spawn_xy": {v: {"x": spawn_xy[v][0], "y": spawn_xy[v][1]} for v in vehicles},
            "conflicts": conflict_points,
        }
        if spawn_sep_debug:
            dbg["spawn_separation"] = spawn_sep_debug
        # Softly penalize choosing an end point outside the strict crop.
        W_END_OUTSIDE = 500.0
        W_END_SHIFT = 0.2  # keep end near default unless needed
        end_outside_debug = []
        for v in vehicles:
            ei = assign_end[v]
            x, y = base[v]["pts"][ei]
            c = _point_outside_cost(x, y)
            if c > 1e-9:
                total += W_END_OUTSIDE * c
                end_outside_debug.append({"vehicle": v, "end_idx": int(ei), "outside_cost": float(c)})
        if end_outside_debug:
            dbg["end_outside"] = end_outside_debug
        # Penalize moving end away from the default end to avoid degenerate early termination.
        for v in vehicles:
            ei = int(assign_end[v])
            base_e = int(base[v]["end"])
            if ei != base_e:
                total += W_END_SHIFT * float(abs(ei - base_e))
        return total, dbg

    # Brute-force search (vehicles small)
    # IMPORTANT: Cap iterations to prevent combinatorial explosion with many vehicles
    MAX_ITERATIONS = 500000  # Should complete in <10s on typical hardware
    iterations_count = 0
    
    best = None
    best_dbg = None
    best_assign_start: Optional[Dict[str, int]] = None
    best_assign_speed: Optional[Dict[str, float]] = None
    best_assign_end: Optional[Dict[str, int]] = None

    # recursive enumeration
    vehs = vehicles
    
    # Log search space size
    search_space = 1
    for v in vehs:
        n_starts = len(candidates_start[v])
        n_ends = len(candidates_end[v])
        n_speeds = len(speed_opts[v])
        combos = n_starts * n_ends * n_speeds
        search_space *= combos
        print(f"[DEBUG] refiner CSP: vehicle {v} has {n_starts} starts × {n_ends} ends × {n_speeds} speeds = {combos} combos", flush=True)
    print(f"[DEBUG] refiner CSP: total search space = {search_space:,} (capped at {MAX_ITERATIONS:,})", flush=True)

    def rec(i: int, cur_start: Dict[str, int], cur_speed: Dict[str, float], cur_end: Dict[str, int]) -> bool:
        nonlocal best, best_dbg, best_assign_start, best_assign_speed, best_assign_end, iterations_count
        if iterations_count >= MAX_ITERATIONS:
            return True  # Signal early stop
        if i == len(vehs):
            iterations_count += 1
            sc, dbg = score(cur_start, cur_speed, cur_end)
            if best is None or sc < best:
                best = sc
                best_dbg = dbg
                best_assign_start = dict(cur_start)
                best_assign_speed = dict(cur_speed)
                best_assign_end = dict(cur_end)
            return False
        v = vehs[i]
        for si in candidates_start[v]:
            cur_start[v] = si
            for ei in candidates_end[v]:
                if ei <= si + 1:
                    continue
                cur_end[v] = ei
                for spd in speed_opts[v]:
                    cur_speed[v] = spd
                    if rec(i + 1, cur_start, cur_speed, cur_end):
                        return True  # Early stop propagation
        return False

    t0_rec = time.time()
    rec(0, {}, {}, {})
    print(f"[DEBUG] refiner CSP: explored {iterations_count:,} leaf nodes in {time.time() - t0_rec:.2f}s", flush=True)

    # Build solution
    sol = {
        "score": float(best if best is not None else 0.0),
        "start_idx": {v: int((best_assign_start or {}).get(v, base[v]["start"])) for v in vehicles},
        "end_idx": {v: int((best_assign_end or {}).get(v, base[v]["end"])) for v in vehicles},
        "speed_mps": {v: float((best_assign_speed or {}).get(v, base_speed[v])) for v in vehicles},
        "debug": best_dbg or {},
        "conflict_count": int(len(conflicts)),
    }
    return sol


# -------------------------
# Visualization
# -------------------------

def visualize_refinement(
    out_png: str,
    crop: CropBox,
    picked_entries: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    lane_change_debug: Optional[List[Dict[str, Any]]] = None,
    conflict_times: Optional[List[Dict[str, Any]]] = None,
    seg_by_id: Optional[Dict[int, Any]] = None,
    show: bool = False,
):
    if plt is None:
        print("[WARNING] matplotlib not available; skipping viz.")
        return

    def _cluster_xy(conf_pts: List[Tuple[float, float, Any]], thresh_m: float = 2.5) -> List[Dict[str, Any]]:
        """
        Simple agglomerative clustering in world coords to merge near-duplicate conflict points.
        conf_pts: [(x,y,meta), ...]
        """
        clusters: List[Dict[str, Any]] = []
        for x, y, meta in conf_pts:
            placed = False
            for c in clusters:
                if math.hypot(x - c["x"], y - c["y"]) <= thresh_m:
                    c["members"].append(meta)
                    n = len(c["members"])
                    c["x"] = (c["x"] * (n - 1) + x) / n
                    c["y"] = (c["y"] * (n - 1) + y) / n
                    placed = True
                    break
            if not placed:
                clusters.append({"x": float(x), "y": float(y), "members": [meta]})
        return clusters

    def _place_label(
        ax,
        xy: Tuple[float, float],
        text: str,
        used_boxes: List[Tuple[float, float, float, float]],
        *,
        color: str = "black",
        fontsize: int = 8,
        weight: Optional[str] = None,
    ):
        """
        Greedy label placer in screen coords, but avoids overlaps using an approximate
        pixel-space bounding box for each label (much better than tracking only points).
        """
        offsets = (
            (20, 20), (20, -20), (-20, 20), (-20, -20),
            (30, 0), (-30, 0), (0, 30), (0, -30),
            (45, 15), (45, -15), (-45, 15), (-45, -15),
            (15, 45), (15, -45), (-15, 45), (-15, -45),
            (60, 0), (-60, 0), (0, 60), (0, -60),
        )
        x, y = xy
        x0, y0 = ax.transData.transform((x, y))

        # Cheap text box size estimate in pixels (good enough for decluttering)
        w = max(70.0, 7.0 * len(text) * (fontsize / 8.0))
        h = 18.0 * (fontsize / 8.0)

        def intersects(a, b) -> bool:
            return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

        best_dxdy = (20, 20)
        best_box = (x0 + 20, x0 + 20 + w, y0 + 20, y0 + 20 + h)
        best_score = -1e18

        for dx, dy in offsets:
            xp, yp = x0 + dx, y0 + dy
            box = (xp, xp + w, yp, yp + h)
            overlaps = sum(1 for ob in used_boxes if intersects(box, ob))

            # Prefer 0 overlaps; otherwise maximize distance to nearest label box center
            if overlaps == 0:
                score = 1e9
            else:
                cx, cy = xp + w / 2.0, yp + h / 2.0
                if used_boxes:
                    mind = min(
                        (cx - (ob[0] + ob[1]) / 2.0) ** 2 + (cy - (ob[2] + ob[3]) / 2.0) ** 2
                        for ob in used_boxes
                    )
                else:
                    mind = 0.0
                score = mind - 1e6 * overlaps

            if score > best_score:
                best_score = score
                best_dxdy = (dx, dy)
                best_box = box

            if overlaps == 0:
                break

        used_boxes.append(best_box)

        ax.annotate(
            text,
            xy=(x, y),
            xytext=best_dxdy,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.6),
            zorder=10,
        )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    used_boxes: List[Tuple[float, float, float, float]] = []

    # Set limits and invert X axis to match scene_objects viz
    ax.set_xlim(crop.xmin - 10, crop.xmax + 10)
    ax.set_ylim(crop.ymin - 10, crop.ymax + 10)
    ax.invert_xaxis()

    # Draw road network nodes in background (if seg_by_id provided)
    if seg_by_id:
        import numpy as _np
        all_pts = []
        for pts in seg_by_id.values():
            if pts is not None and len(pts):
                all_pts.append(pts)
        if all_pts:
            pts_concat = _np.vstack(all_pts)
            ax.scatter(pts_concat[:, 0], pts_concat[:, 1], s=6, color="lightgray", alpha=0.35, zorder=0)

    # Crop box (dashed rectangle)
    rect = plt.Rectangle(
        (crop.xmin, crop.ymin),
        crop.xmax - crop.xmin,
        crop.ymax - crop.ymin,
        fill=False,
        linestyle="--",
        linewidth=2,
        edgecolor="blue",
    )
    ax.add_patch(rect)

    cmap = plt.cm.get_cmap("tab10")

    # paths and markers
    for i, pe in enumerate(picked_entries):
        v = pe.get("vehicle", "?")
        sig = (pe.get("signature") or {}) if isinstance(pe.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        pts, _ = _segments_to_polyline_with_map(segs)
        if len(pts) < 2:
            continue
        color = cmap(i % 10)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=3.0, alpha=0.85, color=color, label=v)

        # start/end markers
        ax.scatter([pts[0][0]], [pts[0][1]], marker="o", s=90, facecolors=color, edgecolors="white", linewidths=1.5, zorder=6)
        ax.scatter([pts[-1][0]], [pts[-1][1]], marker="s", s=90, facecolors=color, edgecolors="white", linewidths=1.5, zorder=6)

        _place_label(ax, (pts[0][0], pts[0][1]), f"{v} start", used_boxes, fontsize=9)
        _place_label(ax, (pts[-1][0], pts[-1][1]), f"{v} end", used_boxes, fontsize=9)

    # conflict points (cluster near-duplicates)
    conf_pts: List[Tuple[float, float, Any]] = []
    for c in conflicts:
        p = c.get("point") or {}
        if "x" in p and "y" in p:
            conf_pts.append((float(p["x"]), float(p["y"]), c))

    for cl in _cluster_xy(conf_pts, thresh_m=2.5):
        x, y = float(cl["x"]), float(cl["y"])
        ax.scatter([x], [y], marker="x", s=90, color="red", zorder=8)
        label = "conflict" if len(cl["members"]) == 1 else f"conflict ×{len(cl['members'])}"
        _place_label(ax, (x, y), label, used_boxes, fontsize=9, color="red", weight="bold")

    # conflict timing (HUD box in axes coords, not stacked at intersection)
    if conflict_times:
        lines: List[str] = []
        for ct in conflict_times:
            tmap = ct.get("t") or {}
            keys = list(tmap.keys())
            if len(keys) == 2:
                a, b = keys[0], keys[1]
                ta, tb = float(tmap[a]), float(tmap[b])
                lines.append(f"{a} vs {b}: {ta:.1f}s / {tb:.1f}s (Δ={abs(ta - tb):.1f}s)")
            elif tmap:
                lines.append(", ".join(f"{k}:{float(v):.1f}s" for k, v in tmap.items()))
        if lines:
            ax.text(
                0.01, 0.99,
                "Conflict timing\n" + "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                zorder=20,
            )

    # lane-change points (cut-start / merge)
    if lane_change_debug:
        for item in lane_change_debug:
            dbg = item.get("debug") or {}
            if isinstance(dbg, dict) and dbg.get("applied"):
                cs = dbg.get("cut_start_point") or {}
                mp = dbg.get("merge_point") or {}
                if "x" in cs and "y" in cs:
                    ax.scatter([cs["x"]], [cs["y"]], marker="^", s=70, color="purple", zorder=7)
                    _place_label(ax, (float(cs["x"]), float(cs["y"])), "cut_start", used_boxes, fontsize=8, color="purple")
                if "x" in mp and "y" in mp:
                    ax.scatter([mp["x"]], [mp["y"]], marker="v", s=70, color="purple", zorder=7)
                    _place_label(ax, (float(mp["x"]), float(mp["y"])), "merge", used_boxes, fontsize=8, color="purple")

    # Build legend with Start/End marker, place outside
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="gray"))
    labels.append("Start (○) / End (□)")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


# -------------------------
# Main refinement driver
# -------------------------

def refine_picked_paths_with_model(
    picked_paths_json: str,
    description: str,
    out_json: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    viz: bool = False,
    viz_out: str = "picked_paths_refined_viz.png",
    viz_show: bool = False,
    prompt_out: Optional[str] = None,
    nodes_root: Optional[str] = None,
) -> Dict[str, Any]:
    t_refiner_start = time.time()
    t0 = time.time()
    with open(picked_paths_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    crop_region = data.get("crop_region") or {}
    crop = CropBox(
        xmin=float(crop_region.get("xmin", -1e9)),
        xmax=float(crop_region.get("xmax", 1e9)),
        ymin=float(crop_region.get("ymin", -1e9)),
        ymax=float(crop_region.get("ymax", 1e9)),
    )

    # Load nodes for road network visualization (if helpers available)
    seg_by_id: Optional[Dict[int, Any]] = None
    nodes_field = data.get("nodes")
    if viz and nodes_field and load_nodes and build_segments_from_nodes and resolve_nodes_path:
        try:
            resolved = resolve_nodes_path(picked_paths_json, str(nodes_field), nodes_root)
            if os.path.exists(resolved):
                nodes_data = load_nodes(resolved)
                all_segs = build_segments_from_nodes(nodes_data)
                seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segs}
        except Exception as e:
            print(f"[WARNING] Could not load nodes for viz: {e}")

    picked = data.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths JSON missing 'picked' list.")

    vehicles = [p.get("vehicle") for p in picked if isinstance(p, dict) and isinstance(p.get("vehicle"), str)]
    vehicles = [v for v in vehicles if v]
    if not vehicles:
        raise SystemExit("[ERROR] No vehicles found in picked paths.")
    print(f"[TIMING] refiner setup (load data, crop): {time.time() - t0:.2f}s", flush=True)

    t0 = time.time()
    constraints = extract_refinement_constraints(
        description=description,
        vehicles=vehicles,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    print(f"[TIMING] refiner LLM constraint extraction: {time.time() - t0:.2f}s", flush=True)

    if prompt_out:
        try:
            Path(prompt_out).write_text(
                json.dumps({"description": description, "vehicles": vehicles, "constraints": constraints}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Apply lane-change macros first (best effort)
    t0 = time.time()
    picked2 = [dict(p) for p in picked]
    lc_debug = []
    synthetic_id = 900000
    v_to_idx = {p.get("vehicle"): i for i, p in enumerate(picked2)}
    for lc in constraints.get("lane_changes", []):
        v = lc["vehicle"]
        tgt = lc["target"]
        if v not in v_to_idx or tgt not in v_to_idx:
            continue
        mover = picked2[v_to_idx[v]]
        target = picked2[v_to_idx[tgt]]
        updated, dbg = apply_merge_into_lane_of(
            mover=mover,
            target=target,
            crop=crop,
            style=lc.get("style", "polite"),
            timing=lc.get("timing", "near_conflict"),
            synthetic_seg_id_base=synthetic_id,
        )
        synthetic_id += 10
        picked2[v_to_idx[v]] = updated
        lc_debug.append({"lane_change": lc, "debug": dbg})
    print(f"[TIMING] refiner lane-change macros: {time.time() - t0:.2f}s", flush=True)

    # Build per_vehicle segments for solver
    t0 = time.time()
    per_vehicle = {}
    for p in picked2:
        v = p.get("vehicle")
        sig = (p.get("signature") or {}) if isinstance(p.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        per_vehicle[v] = {"segments_detailed": segs}

    solution = refine_spawn_and_speeds_soft_csp(per_vehicle, crop, constraints)
    print(f"[TIMING] refiner CSP solve: {time.time() - t0:.2f}s", flush=True)

    # Apply start/end slicing to segments_detailed
    t0 = time.time()
    refined_picked = []
    for p in picked2:
        v = p.get("vehicle")
        sig = (p.get("signature") or {}) if isinstance(p.get("signature"), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        pts, _ = _segments_to_polyline_with_map(segs)
        if not pts:
            refined_picked.append(p)
            continue

        s_idx = int(solution["start_idx"].get(v, 0))
        e_idx = int(solution["end_idx"].get(v, len(pts) - 1))
        new_segs = _slice_segments_detailed(segs, s_idx, e_idx)

        sig2 = dict(sig)
        sig2["segments_detailed"] = new_segs
        sig2["segment_ids"] = [int(s.get("seg_id", 0)) for s in new_segs]
        sig2["num_segments"] = int(len(new_segs))
        sig2["length_m"] = float(sum(float(s.get("length_m", 0.0)) for s in new_segs))
        if new_segs:
            sig2["entry"] = dict(sig2.get("entry", {}))
            sig2["entry"]["point"] = {"x": float(new_segs[0]["start"]["point"]["x"]), "y": float(new_segs[0]["start"]["point"]["y"])}
            sig2["exit"] = dict(sig2.get("exit", {}))
            sig2["exit"]["point"] = {"x": float(new_segs[-1]["end"]["point"]["x"]), "y": float(new_segs[-1]["end"]["point"]["y"])}

        p2 = dict(p)
        if "signature_original" not in p2:
            p2["signature_original"] = p.get("signature")
        p2["signature"] = sig2
        p2["refined"] = {
            "start_idx_global": s_idx,
            "end_idx_global": e_idx,
            "speed_mps": float(solution["speed_mps"].get(v, 8.0)),
        }
        refined_picked.append(p2)
    print(f"[TIMING] refiner apply slicing: {time.time() - t0:.2f}s", flush=True)

    out_payload = dict(data)
    out_payload["source_picked_paths"] = picked_paths_json
    out_payload["picked"] = refined_picked
    out_payload["refinement"] = {
        "constraints": constraints,
        "lane_change_debug": lc_debug,
        "solution": solution,
        "notes": {
            "semantics": "All LLM constraints are treated as soft preferences; defaults used if infeasible.",
            "kinematics": "Constant speed on polyline; time = distance/speed.",
            "conflict_detection": "Closest approach between polylines with a distance threshold.",
        },
    }

    t0 = time.time()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote refined picked paths to: {out_json}")
    print(f"[TIMING] refiner total (internal): {time.time() - t_refiner_start:.2f}s", flush=True)

    if viz:
        # For visualization, always show geometric conflict points (even if sync is disabled).
        # Also compute per-vehicle arrival times at each conflict using the chosen speeds.
        speed_mps = solution.get("speed_mps", {}) or {}
        conflicts_geom: List[Dict[str, Any]] = []
        conflict_times: List[Dict[str, Any]] = []

        # Build (vehicle -> polyline) from the final refined segments
        v_to_pts: Dict[str, List[Tuple[float, float]]] = {}
        for pe in refined_picked:
            v = pe.get("vehicle")
            sig = (pe.get("signature") or {}) if isinstance(pe.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
            pts, _ = _segments_to_polyline_with_map(segs)
            if isinstance(v, str) and pts:
                v_to_pts[v] = pts
        vehicles_for_viz = sorted(v_to_pts.keys())
        for i in range(len(vehicles_for_viz)):
            for j in range(i + 1, len(vehicles_for_viz)):
                va, vb = vehicles_for_viz[i], vehicles_for_viz[j]
                conf = find_conflict_between_polylines(v_to_pts[va], v_to_pts[vb], dist_thresh_m=3.0)
                if conf is None:
                    continue
                conflicts_geom.append({"a": va, "b": vb, "point": conf.get("point", {})})
                sa = float((conf.get("s_along") or {}).get("p1_m", 0.0))
                sb = float((conf.get("s_along") or {}).get("p2_m", 0.0))
                ta = sa / max(0.5, float(speed_mps.get(va, 8.0)))
                tb = sb / max(0.5, float(speed_mps.get(vb, 8.0)))
                conflict_times.append({"a": va, "b": vb, "point": conf.get("point", {}), "t": {va: ta, vb: tb}})

        visualize_refinement(
            viz_out,
            crop,
            refined_picked,
            conflicts=conflicts_geom,
            lane_change_debug=lc_debug,
            conflict_times=conflict_times,
            seg_by_id=seg_by_id,
            show=viz_show,
        )
        print(f"[INFO] Wrote refinement viz to: {viz_out}")

    return out_payload


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--description", required=True, help="Scene description")
    ap.add_argument("--out", default="picked_paths_refined.json", help="Output refined picked paths JSON")
    ap.add_argument("--prompt-out", default="", help="Write debug prompt/constraints JSON here (optional)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--viz", action="store_true", help="Write visualization PNG")
    ap.add_argument("--viz-out", default="picked_paths_refined_viz.png")
    ap.add_argument("--viz-show", action="store_true")
    ap.add_argument("--nodes-root", default=None, help="Directory to search for nodes JSON files")

    args = ap.parse_args()

    model, tokenizer = load_model(args.model)
    refine_picked_paths_with_model(
        picked_paths_json=args.picked_paths,
        description=args.description,
        out_json=args.out,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        viz=args.viz,
        viz_out=args.viz_out,
        viz_show=args.viz_show,
        prompt_out=(args.prompt_out if args.prompt_out else None),
        nodes_root=args.nodes_root,
    )


if __name__ == "__main__":
    main()