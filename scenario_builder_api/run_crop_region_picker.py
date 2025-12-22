#!/usr/bin/env python3
"""
run_crop_region_picker.py

LLM + CSP centered crop-region picker for CARLA towns.

What this is (aligned with your intent)
---------------------------------------
- The extractor is LLM-first: it reads unconstrained natural language and produces
  a *latent geometry specification* (topology + requirements) as JSON.
- We then solve a *constraint satisfaction / optimization* assignment:
  scenario_i -> (town, crop_box) from a discrete set of candidate crops.

Crucial detail: mirrored X axis
-------------------------------
We do not implement left/right ourselves. We reuse generate_legal_paths.build_path_signature(),
whose classify_turn_world() already accounts for your canonical mirrored-X world frame
(X+ is West, X- is East). That guarantees left/right/straight match your pipeline.

Inputs
------
- --town-nodes-dir : directory containing Town*.json (node dumps)
- scenarios loaded from either:
    * --scenarios-file : one scenario per non-empty line
    * --scenarios-dir  : loads .txt / .json / .jsonl
      - .txt: one scenario per non-empty line (IDs auto-assigned)
      - .json: list of {"id": "...", "text": "..."} or dict {"scenarios":[...]}
      - .jsonl: one JSON object per line; must contain "text" (optional "id")

Outputs
-------
- --out : mapping town -> crop -> list of scenario ids
- --out-detailed : includes per-scenario extracted geometry spec + chosen crop + crop features
- Optional: --viz-out-dir writes one PNG per scenario (crop, segments, junction centers, text)

Why this is "LLM + CSP centered"
--------------------------------
1) LLM always runs (unless --extractor=rules explicitly).
2) The LLM spec is expressive (not keyword flags).
3) Assignment is global with capacity + reuse penalty to avoid degenerate "everything in one crop"
   while still preferring minimal crops.

Notes / limitations
-------------------
- This script selects crops that *make the described maneuvers feasible* and provide run-up/run-out.
  It does not guarantee your full downstream pipeline (path picker + refiner + object placer)
  will succeed for every scenario. If you want that, add a verification loop that actually runs
  run_scenario_pipeline on the chosen crop and falls back if it fails.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional viz
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None
    mpatches = None


# ---------------------------------------------------------------------------
# Import generate_legal_paths.py from your codebase
# ---------------------------------------------------------------------------

def _import_generate_legal_paths():
    try:
        import generate_legal_paths as _glp
        return _glp
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.getcwd(),
        here,
        os.path.join(here, "scenario_builder_api"),
        os.path.join(here, "fullcodeandlog"),
        os.path.join(here, "fullcodeandlog", "fullcodeandlog"),
        "/mnt/data/fullcodeandlog",
        "/mnt/data/fullcodeandlog/fullcodeandlog",
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "generate_legal_paths.py")):
            sys.path.insert(0, c)
            try:
                import generate_legal_paths as _glp
                return _glp
            except Exception:
                continue

    raise ModuleNotFoundError(
        "Could not import generate_legal_paths.py. Put this script next to it or add its folder to PYTHONPATH."
    )

glp = _import_generate_legal_paths()


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM output
# ---------------------------------------------------------------------------

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Balanced-brace scan to extract a top-level JSON object from model output."""
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    snippet = text[start : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        start = None
                        continue
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropKey:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_str(self) -> str:
        return f"crop_{self.xmin:.1f}_{self.xmax:.1f}_{self.ymin:.1f}_{self.ymax:.1f}"


@dataclass
class CropFeatures:
    town: str
    crop: CropKey
    center_xy: Tuple[float, float]

    turns: List[str]
    entry_dirs: List[str]
    exit_dirs: List[str]
    dirs: List[str]
    has_oncoming_pair: bool
    is_t_junction: bool
    is_four_way: bool
    has_merge_onto_same_road: bool
    lane_count_est: int
    has_multi_lane: bool

    maneuver_stats: Dict[str, Dict[str, float]]  # man -> {"count":..., "max_entry_dist":..., "max_exit_dist":...}

    n_paths: int
    junction_count: int
    area: float

    _segments_full: Optional[List[Any]] = None
    _junction_centers: Optional[List[np.ndarray]] = None


@dataclass
class Scenario:
    sid: str
    text: str
    source: str = ""


@dataclass
class GeometrySpec:
    topology: str  # "intersection"|"t_junction"|"corridor"|"unknown"
    degree: int    # 3 or 4, 0 if unknown

    required_maneuvers: Dict[str, int]
    needs_oncoming: bool
    needs_merge_onto_same_road: bool
    needs_multi_lane: bool
    min_lane_count: int

    min_entry_runup_m: float
    min_exit_runout_m: float

    preferred_entry_cardinals: List[str]
    avoid_extra_intersections: bool

    confidence: float
    notes: str


# ---------------------------------------------------------------------------
# LLM extractor (mandatory by default)
# ---------------------------------------------------------------------------

class LLMGeometryExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._mdl = None

    def _load(self):
        if self._tok is not None and self._mdl is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._mdl = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self._mdl.to(self.device)
        self._mdl.eval()

    def extract(self, scenario_text: str) -> GeometrySpec:
        self._load()
        import torch
        import time

        schema = '''
Return JSON only, matching this schema exactly:
{
  "topology": "intersection" | "t_junction" | "corridor" | "unknown",
  "degree": 0 | 3 | 4,
  "required_maneuvers": {"straight": 0-3, "left": 0-3, "right": 0-3},
  "needs_oncoming": true|false,
  "needs_merge_onto_same_road": true|false,
  "needs_multi_lane": true|false,
  "min_lane_count": 1-3,
  "min_entry_runup_m": number,
  "min_exit_runout_m": number,
  "preferred_entry_cardinals": ["N","S","E","W"] or [],
  "avoid_extra_intersections": true|false,
  "confidence": 0.0-1.0,
  "notes": "short"
}
'''
        guidance = (
            "You are extracting ROAD-GEOMETRY requirements from a driving scenario.\n"
            "The goal is to pick a map crop where the described maneuvers are possible.\n"
            "Interpret maneuvers (left/right/straight) relative to each vehicle's travel direction.\n"
            "Do NOT encode yield order or priority rules.\n"
            "Be permissive: only require what is necessary to realize the described relations.\n"
            "- If the scenario mentions a turn (left/right) by any vehicle, topology is usually 'intersection'.\n"
            "- If it explicitly says 'T junction' / 'T-junction', topology='t_junction', degree=3.\n"
            "- If it's about narrow passages / oncoming negotiation without explicit intersection, topology='corridor'.\n"
            "- If lane changes or 'other lane' are mentioned, needs_multi_lane=true and min_lane_count>=2.\n"
            "- min_entry_runup_m: approach distance before the main conflict (~25-40 typical).\n"
            "- min_exit_runout_m: distance after conflict to place 'after turning/exiting' hazards (~20-35 if mentioned).\n"
            "- If no compass directions are explicitly mentioned, preferred_entry_cardinals=[].\n"
        )

        prompt = guidance + "\n" + schema + "\nScenario:\n" + scenario_text.strip() + "\n"

        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        print(f"[DEBUG] crop_picker LLM: prompt_tokens={input_len}, max_new=256", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = self._mdl.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.pad_token_id,
            )
        elapsed = time.time() - t0
        out_tokens = out.shape[-1] - input_len
        print(f"[DEBUG] crop_picker LLM: done in {elapsed:.1f}s, output_tokens={out_tokens}", flush=True)
        txt = self._tok.decode(out[0], skip_special_tokens=True)

        obj = _extract_first_json_object(txt)
        if obj is None:
            return self._fallback_spec(scenario_text, notes="parse_failed")

        try:
            return GeometrySpec(
                topology=str(obj.get("topology", "unknown")),
                degree=int(obj.get("degree", 0)),
                required_maneuvers=dict(obj.get("required_maneuvers", {"straight": 1, "left": 0, "right": 0})),
                needs_oncoming=bool(obj.get("needs_oncoming", False)),
                needs_merge_onto_same_road=bool(obj.get("needs_merge_onto_same_road", False)),
                needs_multi_lane=bool(obj.get("needs_multi_lane", False)),
                min_lane_count=int(obj.get("min_lane_count", 1)),
                min_entry_runup_m=float(obj.get("min_entry_runup_m", 28.0)),
                min_exit_runout_m=float(obj.get("min_exit_runout_m", 18.0)),
                preferred_entry_cardinals=list(obj.get("preferred_entry_cardinals", [])) or [],
                avoid_extra_intersections=bool(obj.get("avoid_extra_intersections", True)),
                confidence=float(obj.get("confidence", 0.5)),
                notes=str(obj.get("notes", "")),
            )
        except Exception:
            return self._fallback_spec(scenario_text, notes="bad_fields")

    def _fallback_spec(self, scenario_text: str, notes: str = "") -> GeometrySpec:
        d = scenario_text.lower()
        topology = "intersection" if ("turn" in d or "junction" in d or "intersection" in d) else "corridor"
        if "t junction" in d or "t-junction" in d:
            topology = "t_junction"
        needs_ml = ("change lanes" in d) or ("changes lanes" in d) or ("other lane" in d)
        needs_oncoming = ("oncoming" in d) or ("opposite direction" in d)

        req = {"straight": 1, "left": 0, "right": 0}
        if "turn left" in d or "turns left" in d:
            req["left"] = 1
        if "turn right" in d or "turns right" in d:
            req["right"] = 1

        return GeometrySpec(
            topology=topology,
            degree=3 if topology == "t_junction" else 0,
            required_maneuvers=req,
            needs_oncoming=needs_oncoming,
            needs_merge_onto_same_road=("onto the road vehicle" in d and "traveling on" in d),
            needs_multi_lane=needs_ml,
            min_lane_count=2 if needs_ml else 1,
            min_entry_runup_m=40.0 if ("spawns behind" in d or "chain" in d) else 28.0,
            min_exit_runout_m=28.0 if ("after exiting" in d or "after turning" in d) else 18.0,
            preferred_entry_cardinals=[],
            avoid_extra_intersections=True,
            confidence=0.15,
            notes=f"fallback:{notes}",
        )


# ---------------------------------------------------------------------------
# Town indexing: junction centers, candidate crops, crop features
# ---------------------------------------------------------------------------

def _opposite_dir(d: str) -> str:
    return {"E": "W", "W": "E", "N": "S", "S": "N"}.get(d, d)

def _cluster_points(points: np.ndarray, eps: float = 12.0) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    parents = list(range(len(points)))

    def find(a: int) -> int:
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(len(points)):
        nbrs = tree.query_ball_point(points[i], r=eps)
        for j in nbrs:
            if j != i:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(points)):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def detect_junction_centers(segments: List[Any], adj: List[List[int]]) -> List[np.ndarray]:
    n = len(segments)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in adj[i]:
            indeg[j] += 1

    pts = []
    for i, s in enumerate(segments):
        if len(adj[i]) >= 2 or indeg[i] >= 2:
            pts.append(s.points[-1])
        for j in adj[i]:
            if segments[j].road_id != s.road_id:
                pts.append(s.points[-1])
                break

    if not pts:
        return []
    pts = np.asarray(pts, dtype=float)
    clusters = _cluster_points(pts, eps=14.0)
    centers = [pts[idxs].mean(axis=0) for idxs in clusters if len(idxs) >= 3]
    return centers


def _crop_contains_point(c: CropKey, p: np.ndarray) -> bool:
    return (c.xmin <= float(p[0]) <= c.xmax) and (c.ymin <= float(p[1]) <= c.ymax)


def _point_xy(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _estimate_lane_count(segments_crop: List[Any]) -> int:
    by: Dict[Tuple[int, int], set] = {}
    for s in segments_crop:
        key = (int(s.road_id), int(s.section_id))
        by.setdefault(key, set()).add(int(s.lane_id))
    if not by:
        return 1
    return max(1, max(len(v) for v in by.values()))


def compute_crop_features(
    town_name: str,
    segments_full: List[Any],
    junction_centers: List[np.ndarray],
    center_xy: Tuple[float, float],
    crop: CropKey,
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> Optional[CropFeatures]:
    cb = glp.CropBox(crop.xmin, crop.xmax, crop.ymin, crop.ymax)
    segs_crop = glp.crop_segments(segments_full, cb)
    if len(segs_crop) < 6:
        return None

    adj_crop = glp.build_connectivity(segs_crop)
    paths = glp.generate_legal_paths(
        segs_crop, adj_crop, cb,
        min_path_length=min_path_len,
        max_paths=max_paths,
        max_depth=max_depth,
        allow_within_region_fallback=False
    )
    if len(paths) < 6:
        return None

    sigs = [glp.build_path_signature(p) for p in paths]
    entry_dirs = sorted(set(s["entry"]["cardinal4"] for s in sigs))
    exit_dirs = sorted(set(s["exit"]["cardinal4"] for s in sigs))
    dirs = sorted(set(entry_dirs) | set(exit_dirs))
    turns = sorted(set(s["entry_to_exit_turn"] for s in sigs))

    straights = [s for s in sigs if s["entry_to_exit_turn"] == "straight"]
    entry_set = set(s["entry"]["cardinal4"] for s in straights)
    has_oncoming = any((_opposite_dir(d) in entry_set) for d in entry_set)

    is_t = (len(dirs) == 3)
    is_four = (len(dirs) >= 4)

    by_exit: Dict[Tuple[int, int], set] = {}
    for s in sigs:
        key = (int(s["exit"]["road_id"]), int(s["exit"]["section_id"]))
        by_exit.setdefault(key, set()).add(s["entry_to_exit_turn"])
    has_merge = any(("straight" in v and ("left" in v or "right" in v)) for v in by_exit.values())

    lane_count = _estimate_lane_count(segs_crop)
    has_ml = lane_count >= 2

    jct_count = sum(1 for jc in junction_centers if _crop_contains_point(crop, jc))

    cx, cy = center_xy
    man_stats: Dict[str, Dict[str, float]] = {}
    for man in ["straight", "left", "right", "uturn"]:
        man_stats[man] = {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0}

    for s in sigs:
        man = s["entry_to_exit_turn"]
        ent = s["entry"]["point"]
        ex = s["exit"]["point"]
        ent_x, ent_y = _point_xy(ent)
        ex_x, ex_y = _point_xy(ex)
        ent_d = float(math.hypot(ent_x - cx, ent_y - cy))
        ex_d = float(math.hypot(ex_x - cx, ex_y - cy))
        st = man_stats.setdefault(man, {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0})
        st["count"] += 1.0
        st["max_entry_dist"] = max(st["max_entry_dist"], ent_d)
        st["max_exit_dist"] = max(st["max_exit_dist"], ex_d)

    area = (crop.xmax - crop.xmin) * (crop.ymax - crop.ymin)
    return CropFeatures(
        town=town_name,
        crop=crop,
        center_xy=center_xy,
        turns=turns,
        entry_dirs=entry_dirs,
        exit_dirs=exit_dirs,
        dirs=dirs,
        has_oncoming_pair=has_oncoming,
        is_t_junction=is_t,
        is_four_way=is_four,
        has_merge_onto_same_road=has_merge,
        lane_count_est=lane_count,
        has_multi_lane=has_ml,
        maneuver_stats=man_stats,
        n_paths=len(paths),
        junction_count=jct_count,
        area=float(area),
        _segments_full=segments_full,
        _junction_centers=junction_centers,
    )


def build_candidate_crops_for_town(
    town_name: str,
    town_json_path: str,
    radii: List[float],
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> List[CropFeatures]:
    data = glp.load_nodes(town_json_path)
    segments_full = glp.build_segments(data, min_points=6)
    adj_full = glp.build_connectivity(segments_full)
    jcenters = detect_junction_centers(segments_full, adj_full)

    feats: List[CropFeatures] = []
    for jc in jcenters:
        cx, cy = float(jc[0]), float(jc[1])
        for r in radii:
            ck = CropKey(cx - r, cx + r, cy - r, cy + r)
            f = compute_crop_features(
                town_name=town_name,
                segments_full=segments_full,
                junction_centers=jcenters,
                center_xy=(cx, cy),
                crop=ck,
                min_path_len=min_path_len,
                max_paths=max_paths,
                max_depth=max_depth,
            )
            if f is not None:
                feats.append(f)

    uniq: Dict[str, CropFeatures] = {}
    for f in feats:
        k = f.crop.to_str()
        if k not in uniq:
            uniq[k] = f
        else:
            a = (f.junction_count, f.area)
            b = (uniq[k].junction_count, uniq[k].area)
            if a < b:
                uniq[k] = f

    out = list(uniq.values())
    out.sort(key=lambda x: (x.junction_count, x.area))
    return out


# ---------------------------------------------------------------------------
# Crop feasibility against GeometrySpec
# ---------------------------------------------------------------------------

def _maneuver_needed_count(spec: GeometrySpec, man: str) -> int:
    v = spec.required_maneuvers.get(man, 0)
    try:
        return int(v)
    except Exception:
        return 0


def crop_satisfies_spec(spec: GeometrySpec, crop: CropFeatures) -> bool:
    if spec.topology == "t_junction":
        if not crop.is_t_junction:
            return False
        if spec.degree == 3 and len(crop.dirs) < 3:
            return False
    elif spec.topology == "intersection":
        if len(crop.dirs) < 3:
            return False
        if spec.degree == 4 and not crop.is_four_way:
            return False
        if spec.degree == 3 and not crop.is_t_junction:
            return False

    for man in ["straight", "left", "right"]:
        need = _maneuver_needed_count(spec, man)
        if need > 0:
            if crop.maneuver_stats.get(man, {}).get("count", 0.0) < 1.0:
                return False

    if spec.needs_oncoming and not crop.has_oncoming_pair:
        return False

    if spec.needs_merge_onto_same_road and not crop.has_merge_onto_same_road:
        return False

    if spec.needs_multi_lane:
        if crop.lane_count_est < max(2, spec.min_lane_count):
            return False

    if spec.preferred_entry_cardinals:
        if not any(d in crop.entry_dirs for d in spec.preferred_entry_cardinals):
            return False

    for man in ["straight", "left", "right"]:
        need = _maneuver_needed_count(spec, man)
        if need > 0:
            st = crop.maneuver_stats.get(man, {})
            if float(st.get("max_entry_dist", 0.0)) < float(spec.min_entry_runup_m):
                return False
            if float(st.get("max_exit_dist", 0.0)) < float(spec.min_exit_runout_m):
                return False

    return True


def crop_base_cost(spec: GeometrySpec, crop: CropFeatures, junction_penalty: float) -> float:
    cost = crop.area
    if spec.avoid_extra_intersections:
        cost += junction_penalty * max(0, crop.junction_count - 1)

    if spec.topology == "t_junction" and crop.is_t_junction:
        cost *= 0.97
    if spec.needs_multi_lane and crop.has_multi_lane:
        cost *= 0.98
    if spec.needs_merge_onto_same_road and crop.has_merge_onto_same_road:
        cost *= 0.98
    if spec.topology == "intersection" and spec.degree == 0 and crop.is_four_way:
        cost *= 0.98
    return float(cost)


# ---------------------------------------------------------------------------
# CSP / optimization assignment
# ---------------------------------------------------------------------------

@dataclass
class AssignmentResult:
    mapping: Dict[str, Dict[str, List[str]]]
    detailed: Dict[str, Any]


def solve_assignment(
    scenarios: List[Scenario],
    specs: Dict[str, GeometrySpec],
    crops: List[CropFeatures],
    domain_k: int,
    capacity_per_crop: int,
    reuse_weight: float,
    junction_penalty: float,
    log_every: int = 0,
    viz_out_dir: str = "",
    viz_invert_x: bool = False,
    viz_dpi: int = 150,
    viz_max: int = 0,
) -> AssignmentResult:

    domain: Dict[str, List[Tuple[CropFeatures, float]]] = {}
    total = len(scenarios)
    for i, sc in enumerate(scenarios, start=1):
        spec = specs[sc.sid]
        feas = [c for c in crops if crop_satisfies_spec(spec, c)]
        scored = [(c, crop_base_cost(spec, c, junction_penalty=junction_penalty)) for c in feas]
        scored.sort(key=lambda x: x[1])
        domain[sc.sid] = scored[: max(1, domain_k)]
        if log_every and (i % log_every == 0 or i == total):
            print(f"[INFO] domain {i}/{total}")

    order = sorted(scenarios, key=lambda sc: (len(domain[sc.sid]), -specs[sc.sid].confidence))
    scenarios_by_id = {sc.sid: sc for sc in scenarios}

    assigned: Dict[str, CropFeatures] = {}
    used: set = set()
    load: Dict[Tuple[str, str], int] = {}

    def key(c: CropFeatures) -> Tuple[str, str]:
        return (c.town, c.crop.to_str())

    def incremental_cost(cand: CropFeatures, base: float) -> float:
        inc = base
        if key(cand) not in used:
            inc += reuse_weight
        if capacity_per_crop > 0 and load.get(key(cand), 0) >= capacity_per_crop:
            inc += 1e12
        return inc

    total_order = len(order)
    viz_enabled = bool(viz_out_dir)
    viz_pending: List[Tuple[str, str, CropFeatures]] = []
    viz_count = 0
    viz_rendered: Dict[str, Tuple[str, str]] = {}

    def flush_viz(force: bool = False) -> None:
        nonlocal viz_enabled, viz_count
        if not viz_enabled:
            viz_pending.clear()
            return
        if not viz_pending and not force:
            return
        for sid, text, feat in list(viz_pending):
            if viz_max and viz_count >= viz_max:
                viz_enabled = False
                break
            out_png = os.path.join(
                viz_out_dir, f"{sid}__{feat.town}__{feat.crop.to_str()}.png"
            )
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=text,
                    crop=feat.crop,
                    crop_feat=feat,
                    invert_x=viz_invert_x,
                    dpi=viz_dpi,
                )
                viz_count += 1
                viz_rendered[sid] = (feat.town, feat.crop.to_str())
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")
        viz_pending.clear()

    for i, sc in enumerate(order, start=1):
        sid = sc.sid
        options = domain[sid]
        if not options:
            continue
        best = None
        best_val = float("inf")
        for cand, base in options:
            val = incremental_cost(cand, base)
            if val < best_val:
                best_val = val
                best = (cand, base)
        if best is None or best_val >= 1e11:
            cand, base = options[0]
        else:
            cand, base = best
        assigned[sid] = cand
        used.add(key(cand))
        load[key(cand)] = load.get(key(cand), 0) + 1
        if viz_enabled:
            viz_pending.append((sid, sc.text, cand))
        if log_every and (i % log_every == 0 or i == total_order):
            print(f"[INFO] assigned {i}/{total_order}")
            flush_viz()

    def total_objective() -> float:
        base_sum = 0.0
        used_now = set()
        load_now: Dict[Tuple[str, str], int] = {}
        for sid, c in assigned.items():
            spec = specs[sid]
            base_sum += crop_base_cost(spec, c, junction_penalty=junction_penalty)
            used_now.add(key(c))
            load_now[key(c)] = load_now.get(key(c), 0) + 1
        cap_pen = 0.0
        if capacity_per_crop > 0:
            for _, v in load_now.items():
                if v > capacity_per_crop:
                    cap_pen += 1e9 * (v - capacity_per_crop)
        return base_sum + reuse_weight * len(used_now) + cap_pen

    for _ in range(2):
        improved = False
        cur_obj = total_objective()
        for sc in order:
            sid = sc.sid
            if sid not in assigned:
                continue
            cur_crop = assigned[sid]
            cur_key = key(cur_crop)
            for cand, _ in domain[sid]:
                if key(cand) == cur_key:
                    continue
                if capacity_per_crop > 0:
                    cur_load = sum(1 for _, c2 in assigned.items() if key(c2) == key(cand))
                    if cur_load >= capacity_per_crop:
                        continue
                assigned[sid] = cand
                new_obj = total_objective()
                if new_obj + 1e-6 < cur_obj:
                    improved = True
                    cur_obj = new_obj
                    cur_crop = cand
                    cur_key = key(cand)
                else:
                    assigned[sid] = cur_crop
        if not improved:
            break

    flush_viz(force=True)
    if viz_rendered:
        for sid, prev_key in list(viz_rendered.items()):
            feat = assigned.get(sid)
            if feat is None:
                continue
            new_key = (feat.town, feat.crop.to_str())
            if new_key == prev_key:
                continue
            sc = scenarios_by_id.get(sid)
            if sc is None:
                continue
            out_png = os.path.join(
                viz_out_dir, f"{sid}__{feat.town}__{feat.crop.to_str()}.png"
            )
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=sc.text,
                    crop=feat.crop,
                    crop_feat=feat,
                    invert_x=viz_invert_x,
                    dpi=viz_dpi,
                )
                viz_rendered[sid] = new_key
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")

    mapping: Dict[str, Dict[str, List[str]]] = {}
    detailed: Dict[str, Any] = {"assignments": {}, "unassigned": []}

    for sc in scenarios:
        sid = sc.sid
        if sid not in assigned:
            detailed["unassigned"].append({"id": sid, "text": sc.text, "source": sc.source})
            continue
        c = assigned[sid]
        t = c.town
        ck = c.crop.to_str()
        mapping.setdefault(t, {}).setdefault(ck, []).append(sid)

        spec = specs[sid]
        detailed["assignments"][sid] = {
            "scenario": sc.text,
            "source": sc.source,
            "town": t,
            "crop": [c.crop.xmin, c.crop.xmax, c.crop.ymin, c.crop.ymax],
            "center_xy": list(c.center_xy),
            "geometry_spec": spec.__dict__,
            "crop_features": {
                "dirs": c.dirs,
                "turns": c.turns,
                "entry_dirs": c.entry_dirs,
                "exit_dirs": c.exit_dirs,
                "has_oncoming_pair": c.has_oncoming_pair,
                "is_t_junction": c.is_t_junction,
                "is_four_way": c.is_four_way,
                "has_merge_onto_same_road": c.has_merge_onto_same_road,
                "lane_count_est": c.lane_count_est,
                "junction_count": c.junction_count,
                "maneuver_stats": c.maneuver_stats,
                "n_paths": c.n_paths,
                "area": c.area,
            },
        }

    for t in list(mapping.keys()):
        mapping[t] = dict(sorted(mapping[t].items(), key=lambda kv: kv[0]))
        for k0 in mapping[t]:
            mapping[t][k0] = sorted(mapping[t][k0])

    return AssignmentResult(mapping=mapping, detailed=detailed)


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def _read_txt_scenarios(path: str, source: str) -> List[Scenario]:
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines()]
    items = [ln for ln in lines if ln]
    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    for i, txt in enumerate(items, start=1):
        sid = f"{base}_{i:03d}"
        out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def _read_json_scenarios(path: str, source: str) -> List[Scenario]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    
    # Check if this is the new categorized format (dict with category keys and scenario lists as values)
    if isinstance(obj, dict):
        # Check if it has "scenarios" key (old format wrapper)
        if "scenarios" in obj:
            obj = obj["scenarios"]
            if not isinstance(obj, list):
                raise ValueError(f"{path}: 'scenarios' key must contain a list")
        else:
            # New categorized format: keys are categories, values are lists of scenarios
            # Check if all values are lists (categorized format) vs dict with id/text (old item format)
            has_list_values = any(isinstance(v, list) for v in obj.values())
            has_text_key = "text" in obj
            
            if has_list_values and not has_text_key:
                # This is the new categorized format
                out: List[Scenario] = []
                global_index = 1
                
                # Process categories in sorted order for consistency
                for category in sorted(obj.keys()):
                    scenarios = obj[category]
                    if not isinstance(scenarios, list):
                        raise ValueError(f"{path}: category '{category}' must contain a list of scenarios")
                    
                    for scenario_text in scenarios:
                        if isinstance(scenario_text, str):
                            txt = scenario_text.strip()
                            if not txt:
                                continue
                            # Format: category_globalindex (e.g., intersection_multi_vehicle_001)
                            sid = f"{category}_{global_index:03d}"
                            out.append(Scenario(sid=sid, text=txt, source=source))
                            global_index += 1
                        elif isinstance(scenario_text, dict):
                            txt = str(scenario_text.get("text", "")).strip()
                            if not txt:
                                continue
                            # Allow custom ID or generate default
                            sid = str(scenario_text.get("id", "")).strip() or f"{category}_{global_index:03d}"
                            out.append(Scenario(sid=sid, text=txt, source=source))
                            global_index += 1
                
                return out
            else:
                # Old format: single dict item with id/text keys
                # Fall through to list processing below
                obj = [obj]
    
    # Old list format (or converted single dict)
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected list, categorized dict, or {{'scenarios':[...]}}")
    
    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    for i, item in enumerate(obj, start=1):
        if isinstance(item, str):
            out.append(Scenario(sid=f"{base}_{i:03d}", text=item, source=source))
        elif isinstance(item, dict):
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            sid = str(item.get("id", "")).strip() or f"{base}_{i:03d}"
            out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def _read_jsonl_scenarios(path: str, source: str) -> List[Scenario]:
    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        i = 0
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            i += 1
            item = json.loads(ln)
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            sid = str(item.get("id", "")).strip() or f"{base}_{i:03d}"
            out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def load_scenarios(scenarios_file: str, scenarios_dir: str) -> List[Scenario]:
    out: List[Scenario] = []
    if scenarios_file:
        out.extend(_read_txt_scenarios(scenarios_file, source=scenarios_file))
    if scenarios_dir:
        for fn in sorted(os.listdir(scenarios_dir)):
            p = os.path.join(scenarios_dir, fn)
            if not os.path.isfile(p):
                continue
            lfn = fn.lower()
            if lfn.endswith(".txt"):
                out.extend(_read_txt_scenarios(p, source=p))
            elif lfn.endswith(".jsonl"):
                out.extend(_read_jsonl_scenarios(p, source=p))
            elif lfn.endswith(".json"):
                out.extend(_read_json_scenarios(p, source=p))

    seen = set()
    uniq: List[Scenario] = []
    for s in out:
        if s.sid in seen:
            continue
        seen.add(s.sid)
        uniq.append(s)
    return uniq


# ---------------------------------------------------------------------------
# Visualization (one per scenario)
# ---------------------------------------------------------------------------

def _wrap_text(s: str, width: int = 120) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= width:
        return s
    import textwrap as _tw
    return "\n".join(_tw.wrap(s, width=width))


def save_viz(
    out_png: str,
    scenario_id: str,
    scenario_text: str,
    crop: CropKey,
    crop_feat: CropFeatures,
    invert_x: bool,
    dpi: int,
) -> None:
    if plt is None or mpatches is None:
        raise RuntimeError("matplotlib not available")

    seg_full = crop_feat._segments_full
    jcenters = crop_feat._junction_centers
    if seg_full is None or jcenters is None:
        raise RuntimeError("missing cached segments/junction centers")

    cb = glp.CropBox(crop.xmin, crop.xmax, crop.ymin, crop.ymax)
    segs_crop = glp.crop_segments(seg_full, cb)

    fig = plt.figure(figsize=(9.2, 7.6), dpi=dpi)
    ax = plt.gca()

    for seg in segs_crop:
        pts = np.asarray(seg.points, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0, alpha=0.55)
        ax.scatter([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], s=8, alpha=0.75)

    rect = mpatches.Rectangle((crop.xmin, crop.ymin), crop.xmax - crop.xmin, crop.ymax - crop.ymin,
                              fill=False, linewidth=2.0)
    ax.add_patch(rect)

    xs, ys = [], []
    for jc in jcenters:
        if _crop_contains_point(crop, jc):
            xs.append(float(jc[0]))
            ys.append(float(jc[1]))
    if xs:
        ax.scatter(xs, ys, s=38, marker="x")

    cx, cy = crop_feat.center_xy
    ax.scatter([cx], [cy], s=55, marker="o", alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = f"{scenario_id} | {crop_feat.town} | {crop.to_str()}"
    ax.set_title(title, fontsize=10)
    fig.suptitle(_wrap_text(scenario_text, width=110), fontsize=9, y=0.98)

    pad = 6.0
    ax.set_xlim(crop.xmin - pad, crop.xmax + pad)
    ax.set_ylim(crop.ymin - pad, crop.ymax + pad)

    if invert_x:
        ax.invert_xaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town-nodes-dir", required=True, help="Directory containing Town*.json")
    ap.add_argument("--scenarios-file", default="", help="One scenario per line")
    ap.add_argument("--scenarios-dir", default="", help="Directory of scenario files (.txt/.json/.jsonl)")

    ap.add_argument("--out", required=True, help="Output mapping JSON")
    ap.add_argument("--out-detailed", default="", help="Output detailed JSON")

    ap.add_argument("--extractor", choices=["llm", "rules"], default="llm",
                    help="Use LLM-centered extraction (default) or a conservative fallback")
    ap.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--radii", default="45,55,65",
                    help="Comma-separated crop half-width radii (meters). Include at least one >=55 for run-up.")
    ap.add_argument("--min-path-length", type=float, default=22.0)
    ap.add_argument("--max-paths", type=int, default=80)
    ap.add_argument("--max-depth", type=int, default=8)

    ap.add_argument("--domain-k", type=int, default=50)
    ap.add_argument("--capacity-per-crop", type=int, default=10)
    ap.add_argument("--reuse-weight", type=float, default=4000.0)
    ap.add_argument("--junction-penalty", type=float, default=25000.0)
    ap.add_argument("--log-every", type=int, default=25,
                    help="Print progress every N scenarios (0 = off). With --viz-out-dir, writes viz in batches.")

    ap.add_argument("--viz-out-dir", default="")
    ap.add_argument("--viz-invert-x", action="store_true")
    ap.add_argument("--viz-dpi", type=int, default=150)
    ap.add_argument("--viz-max", type=int, default=0)

    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios_file, args.scenarios_dir)
    if not scenarios:
        raise SystemExit("No scenarios loaded. Provide --scenarios-file or --scenarios-dir")

    town_files = []
    for fn in sorted(os.listdir(args.town_nodes_dir)):
        if fn.lower().endswith(".json") and fn.lower().startswith("town"):
            town_files.append(os.path.join(args.town_nodes_dir, fn))
    if not town_files:
        raise SystemExit(f"No Town*.json found in {args.town_nodes_dir}")

    radii = [float(x) for x in args.radii.split(",") if x.strip()]

    all_crops: List[CropFeatures] = []
    for p in town_files:
        town = os.path.splitext(os.path.basename(p))[0]
        print(f"[INFO] indexing {town} ...")
        crops = build_candidate_crops_for_town(
            town_name=town,
            town_json_path=p,
            radii=radii,
            min_path_len=args.min_path_length,
            max_paths=args.max_paths,
            max_depth=args.max_depth,
        )
        print(f"[INFO]  {town}: {len(crops)} candidate crops")
        all_crops.extend(crops)

    if not all_crops:
        raise SystemExit("No candidate crops built. Try larger radii or smaller --min-path-length.")

    specs: Dict[str, GeometrySpec] = {}
    extractor = LLMGeometryExtractor(model_name=args.llm_model, device=args.device)
    total = len(scenarios)
    viz_streaming = False
    viz_out_dir = args.viz_out_dir
    if viz_out_dir:
        if plt is None or mpatches is None:
            print("[WARN] matplotlib not available; skipping visualization")
            viz_out_dir = ""
        elif args.log_every and args.log_every > 0:
            viz_streaming = True

    if args.extractor == "llm":
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor.extract(sc.text)
            if args.log_every and (i % args.log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")
    else:
        for i, sc in enumerate(scenarios, start=1):
            specs[sc.sid] = extractor._fallback_spec(sc.text, notes="rules_mode")
            if args.log_every and (i % args.log_every == 0 or i == total):
                print(f"[INFO] extracted {i}/{total}")

    res = solve_assignment(
        scenarios=scenarios,
        specs=specs,
        crops=all_crops,
        domain_k=args.domain_k,
        capacity_per_crop=args.capacity_per_crop,
        reuse_weight=args.reuse_weight,
        junction_penalty=args.junction_penalty,
        log_every=args.log_every,
        viz_out_dir=viz_out_dir if viz_streaming else "",
        viz_invert_x=args.viz_invert_x,
        viz_dpi=args.viz_dpi,
        viz_max=args.viz_max,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res.mapping, f, indent=2, sort_keys=True)
    print(f"[OK] wrote mapping to {args.out}")

    if args.out_detailed:
        with open(args.out_detailed, "w", encoding="utf-8") as f:
            json.dump(res.detailed, f, indent=2, sort_keys=True)
        print(f"[OK] wrote detailed to {args.out_detailed}")

    if args.viz_out_dir and not viz_streaming and viz_out_dir:
        items = list(res.detailed.get("assignments", {}).items())
        if args.viz_max and args.viz_max > 0:
            items = items[: args.viz_max]

        scen_map = {s.sid: s.text for s in scenarios}

        crop_lookup: Dict[Tuple[str, str], CropFeatures] = {}
        for c in all_crops:
            crop_lookup[(c.town, c.crop.to_str())] = c

        for sid, info in items:
            town = info["town"]
            crop_vals = info["crop"]
            ck = CropKey(float(crop_vals[0]), float(crop_vals[1]), float(crop_vals[2]), float(crop_vals[3]))
            crop_str = ck.to_str()
            feat = crop_lookup.get((town, crop_str))
            if feat is None:
                continue
            out_png = os.path.join(args.viz_out_dir, f"{sid}__{town}__{ck.to_str()}.png")
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=scen_map.get(sid, info.get("scenario", "")),
                    crop=ck,
                    crop_feat=feat,
                    invert_x=args.viz_invert_x,
                    dpi=args.viz_dpi,
                )
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")
        print(f"[OK] wrote visualizations to {args.viz_out_dir}")
    elif viz_streaming:
        print(f"[OK] wrote visualizations to {args.viz_out_dir}")

    n_assigned = len(res.detailed.get("assignments", {}))
    n_total = len(scenarios)
    print(f"[SUMMARY] assigned {n_assigned}/{n_total} scenarios")
    if res.detailed.get("unassigned"):
        print(f"[WARN] {len(res.detailed['unassigned'])} scenarios unassigned (see --out-detailed)")

if __name__ == "__main__":
    main()
