#!/usr/bin/env python3
"""
run_object_placer.py

Given:
- picked_paths_detailed.json (output from your path picker; includes per-vehicle picked path + segments_detailed)
- carla_assets.json (list of spawnable CARLA assets)

This script uses a local HuggingFace instruction-tuned model to:
1) Extract non-ego actors (props / parked vehicles / pedestrians / cyclists / NPC vehicles) from a natural-language scene description
2) Resolve each actor to:
   - a specific ego vehicle (or none),
   - a specific segment index within that vehicle's picked path,
   - a longitudinal location along that segment (s_along in [0,1]),
   - a lateral relation relative to travel direction (center / half_right / right_edge / half_left / left_edge / offroad_*),
   - an asset_id chosen from carla_assets.json,
   - (optional) motion intent and speed profile.
3) Convert those anchors into concrete world-frame spawn transforms and waypoint trajectories (x,y,yaw,speed).

Outputs:
- scene_objects.json (IR + concrete world placements)
- scene_objects.png (viz of picked ego paths + all placed actors and motion arrows)

Notes:
- For accurate geometry we rebuild segment polylines from the nodes file referenced in picked_paths_detailed.json.
- If the nodes path in picked_paths_detailed.json is relative, we resolve it relative to the picked_paths_detailed.json location,
  or an optional --nodes-root.

Example:
  python run_object_placer.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --picked-paths scenario_builder_api/picked_paths_detailed.json \
    --carla-assets scenario_builder_api/carla_assets.json \
    --description "Vehicle 1 continues straight... parked truck... bicyclist half right ... pedestrian crosses..." \
    --out scene_objects.json \
    --viz-out scene_objects.png
"""

import argparse
import json
import difflib
import math
import os
import re
import random
import textwrap
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
from matplotlib.lines import Line2D
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy") from e

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ======================================================================================
# Small geometry helpers
# ======================================================================================

def wrap180(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0

def heading_deg_from_vec(v: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))

def cumulative_dist(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 2:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)], axis=0)

def point_and_tangent_at_s(points_xy: np.ndarray, s_along: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (point, unit_tangent) at fractional arc-length s_along ∈ [0,1].
    Uses piecewise-linear interpolation on the polyline.
    """
    pts = np.asarray(points_xy, dtype=float)
    s = float(min(1.0, max(0.0, s_along)))
    if len(pts) == 1:
        return pts[0].copy(), np.array([1.0, 0.0], dtype=float)

    cum = cumulative_dist(pts)
    total = float(cum[-1])
    if total < 1e-9:
        # Degenerate
        v = pts[-1] - pts[0]
        n = float(np.linalg.norm(v))
        t = v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return pts[0].copy(), t

    target = s * total
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))

    a = pts[idx]
    b = pts[idx + 1]
    seg_len = float(np.linalg.norm(b - a))
    if seg_len < 1e-9:
        # Find a non-degenerate neighbor for tangent
        j = idx
        while j + 1 < len(pts) and float(np.linalg.norm(pts[j + 1] - pts[j])) < 1e-9:
            j += 1
        if j + 1 < len(pts):
            v = pts[j + 1] - pts[j]
        else:
            v = pts[-1] - pts[0]
        n = float(np.linalg.norm(v))
        t = v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return a.copy(), t

    seg_start = float(cum[idx])
    alpha = (target - seg_start) / seg_len
    p = a + alpha * (b - a)
    t = (b - a) / seg_len
    return p, t

def _closest_point_s_m_on_polyline(points_xy: np.ndarray, point_xy: np.ndarray) -> Tuple[float, float]:
    """
    Return (min_dist_m, s_m) for the closest point on a polyline to point_xy.
    s_m is distance along the polyline from its start.
    """
    pts = np.asarray(points_xy, dtype=float)
    p = np.asarray(point_xy, dtype=float).reshape(2)
    if len(pts) < 2:
        d = float(np.linalg.norm(p - pts[0])) if len(pts) == 1 else 0.0
        return d, 0.0

    cum = cumulative_dist(pts)
    best_dist = float("inf")
    best_s_m = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-12:
            continue
        t = float(np.dot(p - a, ab) / ab_len2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        s_m = float(cum[i] + t * math.sqrt(ab_len2))
        if dist < best_dist:
            best_dist = dist
            best_s_m = s_m
    return best_dist, best_s_m

def _project_point_to_path_s_m(
    picked_entry: Dict[str, Any],
    seg_by_id: Dict[int, np.ndarray],
    point_xy: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Project a world point onto a vehicle's path and return (dist_m, path_s_m).
    path_s_m is distance along the full path from its start.
    """
    seg_ids = picked_entry.get("signature", {}).get("segment_ids", [])
    if not isinstance(seg_ids, list) or not seg_ids:
        return None

    total_offset = 0.0
    best_dist = None
    best_path_s = None
    for seg_id_raw in seg_ids:
        try:
            seg_id = int(seg_id_raw)
        except Exception:
            continue
        pts = seg_by_id.get(seg_id)
        if pts is None or len(pts) < 2:
            continue
        dist, s_m = _closest_point_s_m_on_polyline(pts, point_xy)
        path_s = total_offset + s_m
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_path_s = path_s
        total_offset += float(cumulative_dist(pts)[-1])
    if best_dist is None or best_path_s is None:
        return None
    return best_dist, best_path_s

def right_normal_world(tangent: np.ndarray) -> np.ndarray:
    """
    IMPORTANT: Your "WORLD_FRAME" is effectively left-handed due to mirrored X.
    In your turn classifier, +delta heading is "right".
    That corresponds to a +90° rotation being "right".
    So:
      right_normal = rot(+90°) = (-dy, dx)
      left_normal  = rot(-90°) = (dy, -dx)
    """
    dx, dy = float(tangent[0]), float(tangent[1])
    n = np.array([-dy, dx], dtype=float)
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 1e-9 else np.array([0.0, 1.0], dtype=float)

def left_normal_world(tangent: np.ndarray) -> np.ndarray:
    dx, dy = float(tangent[0]), float(tangent[1])
    n = np.array([dy, -dx], dtype=float)
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 1e-9 else np.array([0.0, -1.0], dtype=float)


# ======================================================================================
# Robust JSON extraction from LLM output (prefers the LAST valid object)
# ======================================================================================

def _extract_all_json_objects(text: str) -> List[Any]:
    """
    Extract JSON values (dict OR list) from arbitrary text.

    We collect multiple candidates because LLMs often wrap JSON in prose,
    code-fences, or output multiple JSON blobs. We later pick the most
    plausible one (usually the last one).
    """
    vals: List[Any] = []

    # 1) Direct parse (whole text is JSON)
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            vals.append(obj)
    except Exception:
        pass

    # 2) Code-fenced blocks ```json ... ``` or ``` ... ```
    for m in re.finditer(r"```(?:json)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
            if isinstance(obj, (dict, list)):
                vals.append(obj)
        except Exception:
            continue

    # Helper: balanced scan for a given opener/closer
    def _balanced_scan(opener: str, closer: str) -> None:
        n = len(text)
        start_search = 0
        while True:
            start = text.find(opener, start_search)
            if start < 0:
                break
            depth = 0
            in_str = False
            esc = False
            i = start
            while i < n:
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    i += 1
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        i += 1
                        continue
                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            snippet = text[start:i + 1]
                            try:
                                obj = json.loads(snippet)
                                if isinstance(obj, (dict, list)):
                                    vals.append(obj)
                            except Exception:
                                pass
                            break
                    i += 1
            start_search = start + 1

    # 3) Balanced dict scan {...}
    _balanced_scan("{", "}")
    # 4) Balanced list scan [...]
    _balanced_scan("[", "]")

    return vals

def _pick_last_matching(objs: List[Dict[str, Any]], required_top_keys: List[str]) -> Optional[Dict[str, Any]]:
    for obj in reversed(objs):
        ok = True
        for k in required_top_keys:
            if k not in obj:
                ok = False
                break
        if ok:
            return obj
    return None

def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    """Depth-first search for the first occurrence of a key in nested dict/list structures."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = _find_key_recursive(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _find_key_recursive(it, key)
            if got is not None:
                return got
    return None


def parse_llm_json(text: str, required_top_keys: List[str]) -> Dict[str, Any]:
    """
    Best-effort parse of LLM output into a JSON dict.

    Handles common failure modes:
    - top-level list (e.g., the model outputs just an array of actors)
    - required key nested (e.g., {"output": {"actors": [...]}})
    - JSON inside ```json fences
    - multiple JSON blobs; we usually want the LAST valid one
    """
    candidates = _extract_all_json_objects(text)

    # 1) Prefer exact top-level match (dict with required keys)
    for obj in reversed(candidates):
        if isinstance(obj, dict) and all(k in obj for k in required_top_keys):
            return obj

    # 2) If a single key is required and the model returned a top-level list, wrap it.
    if len(required_top_keys) == 1:
        k = required_top_keys[0]
        for obj in reversed(candidates):
            if isinstance(obj, list):
                return {k: obj}

    # 3) If key is nested somewhere, lift it to top-level.
    lifted: Dict[str, Any] = {}
    for k in required_top_keys:
        found = None
        for obj in reversed(candidates):
            found = _find_key_recursive(obj, k)
            if found is not None:
                break
        if found is None:
            lifted = {}
            break
        lifted[k] = found

    if lifted:
        return lifted

    # 4) Last resort: salvage by regex for the first required key (common case: actors)
    if len(required_top_keys) == 1:
        k = required_top_keys[0]
        # Try to grab an array following the key, even if surrounded by prose
        m = re.search(rf'"{re.escape(k)}"\s*:\s*(\[[\s\S]*?\])', text)
        if m:
            try:
                arr = json.loads(m.group(1))
                if isinstance(arr, list):
                    return {k: arr}
            except Exception:
                pass
        # Try to grab an object containing the key
        m2 = re.search(rf'\{{[\s\S]*?"{re.escape(k)}"\s*:\s*(\[[\s\S]*?\])[\s\S]*?\}}', text)
        if m2:
            snippet = m2.group(0)
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict) and k in obj:
                    return obj
            except Exception:
                pass

    # 5) Give up
    raise ValueError(
        f"Could not find JSON containing required top-level keys: {required_top_keys}.\n"
        f"Raw model output (truncated): {text[:800]!r}"
    )


# ======================================================================================
# Assets: lightweight filtering so we don't shove the whole list into the LLM
# ======================================================================================

@dataclass
class AssetBBox:
    """Bounding box dimensions for an asset (in meters)."""
    length: float  # extent in forward direction (x)
    width: float   # extent in lateral direction (y)
    height: float  # extent in vertical direction (z)

@dataclass
class Asset:
    category: str
    asset_id: str
    tags: List[str]
    attributes: List[Dict[str, Any]]
    bbox: Optional[AssetBBox] = None

# Global lookup for asset bounding boxes by asset_id
_ASSET_BBOX_CACHE: Dict[str, Optional[AssetBBox]] = {}

def load_assets(path: str) -> List[Asset]:
    global _ASSET_BBOX_CACHE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Asset] = []
    assets = data.get("assets", {})
    for cat, lst in assets.items():
        if not isinstance(lst, list):
            continue
        for a in lst:
            if not isinstance(a, dict):
                continue
            # Parse bounding box if present
            bbox_data = a.get("bbox")
            asset_bbox = None
            if isinstance(bbox_data, dict):
                try:
                    # Use length/width/height if available, else compute from extents
                    length = float(bbox_data.get("length", bbox_data.get("extent_x", 0) * 2))
                    width = float(bbox_data.get("width", bbox_data.get("extent_y", 0) * 2))
                    height = float(bbox_data.get("height", bbox_data.get("extent_z", 0) * 2))
                    if length > 0 and width > 0:
                        asset_bbox = AssetBBox(length=length, width=width, height=height)
                except (TypeError, ValueError):
                    pass
            
            asset_id = str(a.get("id", ""))
            out.append(Asset(
                category=str(cat),
                asset_id=asset_id,
                tags=[str(t).lower() for t in a.get("tags", []) if t is not None],
                attributes=a.get("attributes", []) if isinstance(a.get("attributes", []), list) else [],
                bbox=asset_bbox,
            ))
            # Cache the bbox for quick lookup
            _ASSET_BBOX_CACHE[asset_id] = asset_bbox
    return out

def get_asset_bbox(asset_id: str) -> Optional[AssetBBox]:
    """Get bounding box for an asset by its ID. Returns None if not available."""
    return _ASSET_BBOX_CACHE.get(asset_id)

def keyword_filter_assets(all_assets: List[Asset], keywords: List[str], categories: Optional[List[str]] = None, k: int = 12) -> List[Asset]:
    kws = [kw.lower().strip() for kw in keywords if kw and kw.strip()]
    cats = set([c.lower() for c in categories]) if categories else None

    scored: List[Tuple[float, Asset]] = []
    for a in all_assets:
        if cats and a.category.lower() not in cats:
            continue
        hay = " ".join([a.asset_id.lower()] + a.tags)
        score = 0.0
        for kw in kws:
            if kw in hay:
                score += 1.0
        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: (-x[0], x[1].asset_id))
    return [a for _, a in scored[:k]]


# ======================================================================================
# Nodes -> segments reconstruction (for accurate spawn points)
# (Matches your build_segments() grouping and seg_id assignment strategy.)
# ======================================================================================

def load_nodes(nodes_path: str) -> Dict[str, Any]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{nodes_path} missing top-level 'payload'")
    return data

def _unit_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    r = math.radians(wrap180(yaw_deg))
    return np.array([math.cos(r), math.sin(r)], dtype=float)

def _orient_polyline(points_xy: np.ndarray, yaws_deg: np.ndarray, orig_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    yaws = np.asarray(yaws_deg, dtype=float)
    idxs = np.asarray(orig_idx, dtype=int)
    if len(pts) < 2:
        return pts, yaws, idxs

    vecs = pts[1:] - pts[:-1]
    norms = (np.linalg.norm(vecs, axis=1) + 1e-9)
    dir_vecs = vecs / norms[:, None]
    yaw_vecs = np.vstack([_unit_from_yaw_deg(y) for y in yaws[:-1]])
    dots = np.sum(dir_vecs * yaw_vecs, axis=1)
    if float(np.nanmean(dots)) < 0.0:
        return pts[::-1].copy(), yaws[::-1].copy(), idxs[::-1].copy()
    return pts, yaws, idxs

def _split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws: np.ndarray, gap_m: float = 6.0):
    if len(pts) < 2:
        return [(idxs_sorted, pts, yaws)] if len(pts) > 0 else []
    jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cuts = [0]
    for i, d in enumerate(jumps):
        if float(d) > gap_m:
            cuts.append(i + 1)
    cuts.append(len(pts))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            out.append((idxs_sorted[a:b], pts[a:b], yaws[a:b]))
    return out

def build_segments_from_nodes(nodes: Dict[str, Any], min_points: int = 6) -> List[Dict[str, Any]]:
    payload = nodes["payload"]
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    from collections import defaultdict
    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    segments: List[Dict[str, Any]] = []
    seg_id_counter = 0

    # IMPORTANT: iteration order determines seg_id assignment (insertion order of grouped).
    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]

        for idxs_chunk, pts_chunk, yaws_chunk in _split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = _orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue
            segments.append({
                "seg_id": int(seg_id_counter),
                "road_id": int(rid),
                "lane_id": int(lid),
                "section_id": int(sid),
                "points": pts_o.astype(float),
                "yaws": np.asarray([wrap180(v) for v in yaws_o], dtype=float),
                "orig_idx": idxs_o.astype(int),
            })
            seg_id_counter += 1

    return segments


def _override_seg_points_with_picked(
    picked_list: List[Dict[str, Any]], seg_by_id: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Replace/augment seg_by_id geometry with any polyline_samples present in picked_list.
    This lets refined start/end slices or synthetic segments propagate into placement/viz.
    """
    out = dict(seg_by_id)
    for p in picked_list:
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        for s in segs:
            if not isinstance(s, dict):
                continue
            seg_id = s.get("seg_id", None)
            pl = s.get("polyline_sample") or []
            if seg_id is None or len(pl) < 2:
                continue
            try:
                sid = int(seg_id)
            except Exception:
                continue
            pts = np.array([[float(pt["x"]), float(pt["y"])] for pt in pl if "x" in pt and "y" in pt], dtype=float)
            if len(pts) >= 2:
                out[sid] = pts
    return out


# ======================================================================================
# Path Extension (extend paths through junctions when more length is needed)
# ======================================================================================

def _heading_at_end(pts: np.ndarray, k: int = 6) -> float:
    """Compute heading at end of polyline using last k+1 points."""
    k = min(k, len(pts) - 1)
    if k == 0:
        return 0.0
    v = pts[-1] - pts[-(k + 1)]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    return heading_deg_from_vec(v)


def _heading_at_start(pts: np.ndarray, k: int = 6) -> float:
    """Compute heading at start of polyline using first k+1 points."""
    k = min(k, len(pts) - 1)
    if k == 0:
        return 0.0
    v = pts[k] - pts[0]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    return heading_deg_from_vec(v)


def _ang_diff_deg(a: float, b: float) -> float:
    """Absolute wrapped difference in degrees."""
    return abs(wrap180(a - b))


def _segment_length(pts: np.ndarray) -> float:
    """Compute total arc length of a polyline."""
    if len(pts) < 2:
        return 0.0
    return float(cumulative_dist(pts)[-1])


def _find_best_successor_segment(
    end_pt: np.ndarray,
    end_heading: float,
    all_segments: List[Dict[str, Any]],
    excluded_seg_ids: set,
    connect_radius_m: float = 6.0,
    connect_yaw_tol_deg: float = 45.0,  # Stricter than normal for "straight-through"
) -> Optional[Dict[str, Any]]:
    """
    Find the best successor segment for path extension.
    
    Returns the segment that:
    1. Has start point within connect_radius_m of end_pt
    2. Has start heading within connect_yaw_tol_deg of end_heading
    3. Among valid candidates, picks the one with smallest heading difference (most aligned)
    
    Returns None if no valid successor found.
    """
    best_seg = None
    best_ang_diff = float('inf')
    
    for seg in all_segments:
        seg_id = seg.get("seg_id")
        if seg_id in excluded_seg_ids:
            continue
        
        pts = seg.get("points")
        if pts is None or len(pts) < 2:
            continue
        pts = np.asarray(pts, dtype=float)
        
        # Check distance from end_pt to segment start
        start_pt = pts[0]
        dist = np.linalg.norm(start_pt - end_pt)
        if dist > connect_radius_m:
            continue
        
        # Check heading alignment
        start_heading = _heading_at_start(pts)
        ang_diff = _ang_diff_deg(end_heading, start_heading)
        if ang_diff > connect_yaw_tol_deg:
            continue
        
        # Pick the most aligned one
        if ang_diff < best_ang_diff:
            best_ang_diff = ang_diff
            best_seg = seg
    
    return best_seg


def _polyline_sample_from_pts(pts: np.ndarray, max_points: int = 12) -> List[Dict[str, float]]:
    """Convert numpy points to polyline_sample format."""
    n = len(pts)
    if n == 0:
        return []
    if n <= max_points:
        return [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    idxs = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [{"x": float(pts[i, 0]), "y": float(pts[i, 1])} for i in idxs]


def _compute_path_length(picked_entry: Dict[str, Any], seg_by_id: Dict[int, np.ndarray]) -> float:
    """Compute total path length for a picked vehicle path."""
    sig = picked_entry.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    total = 0.0
    for sid in seg_ids:
        pts = seg_by_id.get(int(sid))
        if pts is not None and len(pts) >= 2:
            total += _segment_length(pts)
    return total


def _estimate_required_path_length(
    actor_specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Estimate how much path length each vehicle needs based on entity relations.
    
    This is a conservative estimate used for initial path extension.
    More precise extension happens during CSP solve.
    
    Returns a dict mapping vehicle_num -> required_length_m
    """
    # Distance mapping from qualitative to meters (same as Stage2 prompt)
    DISTANCE_TO_M = {
        "touching": 2.0,
        "close": 6.0,
        "medium": 12.0,
        "far": 23.0,  # For "Twenty meters later" type phrases
    }
    
    required: Dict[int, float] = {}
    
    # Compute current path lengths
    for p in picked_list:
        veh_num = _parse_vehicle_num(p.get("vehicle"))
        if veh_num is not None:
            current_len = _compute_path_length(p, seg_by_id)
            required[veh_num] = current_len
    
    # Build relation chains and compute cumulative distance needs
    # For chains like: entity_1 -> entity_2 (20m) -> entity_3 (20m), we need base_position + 40m
    
    entity_to_veh: Dict[str, int] = {}
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        veh_num = spec.get("vehicle_num")
        if veh_num is not None:
            entity_to_veh[eid] = int(veh_num)
    
    # Build dependency graph and compute chain lengths
    # ahead_of means: this entity is ahead_of other (so this entity is further along path)
    depends_on: Dict[str, Tuple[str, float]] = {}  # entity -> (depends_on_entity, distance_m)
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        for rel in spec.get("relations", []):
            if rel.get("type") in ("ahead_of",):
                other_id = str(rel.get("other_id", ""))
                distance = rel.get("distance", "medium")
                dist_m = DISTANCE_TO_M.get(distance, 12.0)
                depends_on[eid] = (other_id, dist_m)
    
    # For each chain root (entity with no dependencies), compute total chain length
    def compute_chain_length(eid: str, visited: set) -> float:
        """Compute cumulative distance from this entity to end of chain."""
        if eid in visited:
            return 0.0
        visited.add(eid)
        
        # Find entities that depend on this one
        chain_len = 0.0
        for other_eid, (dep_id, dist_m) in depends_on.items():
            if dep_id == eid:
                chain_len = max(chain_len, dist_m + compute_chain_length(other_eid, visited))
        return chain_len
    
    # Entities that have dependencies on them need extra path length
    for spec in actor_specs:
        eid = str(spec.get("id", ""))
        veh_num = spec.get("vehicle_num")
        if veh_num is None:
            continue
        
        chain_len = compute_chain_length(eid, set())
        if chain_len > 0:
            # This entity has things ahead of it - we need current + chain_len + margin
            current = required.get(veh_num, 0.0)
            # Assume base entity is placed around 60% of path (typical for after_exit)
            base_position_estimate = current * 0.6
            needed = base_position_estimate + chain_len + 10.0  # 10m margin
            required[veh_num] = max(current, needed)
    
    return required


def _find_parallel_vehicles(
    primary_veh_num: int,
    picked_list: List[Dict[str, Any]],
) -> List[Tuple[int, bool, str]]:
    """
    Find vehicles that share the same exit road as the primary vehicle,
    OR that enter on the primary's exit road (opposite direction).
    
    These vehicles travel in parallel, merge, or travel in the opposite
    direction on the same road - they should all be extended together when
    the primary vehicle is extended, so entities can be placed correctly.
    
    Returns list of (vehicle_number, is_same_direction, extend_mode) tuples.
    extend_mode is 'append' for same-direction, 'prepend' for opposite-direction.
    """
    # Get the primary vehicle's final road and heading
    primary_entry = None
    for p in picked_list:
        if _parse_vehicle_num(p.get("vehicle")) == primary_veh_num:
            primary_entry = p
            break
    
    if primary_entry is None:
        return []
    
    primary_roads = primary_entry.get("signature", {}).get("roads", [])
    if not primary_roads:
        return []
    
    primary_exit_road = primary_roads[-1]
    primary_exit_heading = primary_entry.get("signature", {}).get("exit", {}).get("heading_deg", 0)
    
    parallel_vehs = []
    for p in picked_list:
        veh_num = _parse_vehicle_num(p.get("vehicle"))
        if veh_num is None or veh_num == primary_veh_num:
            continue
        
        roads = p.get("signature", {}).get("roads", [])
        if not roads:
            continue
        
        exit_road = roads[-1]
        entry_road = roads[0]
        exit_heading = p.get("signature", {}).get("exit", {}).get("heading_deg", 0)
        entry_heading = p.get("signature", {}).get("entry", {}).get("heading_deg", 0)
        
        # Check if this vehicle shares the same EXIT road (same or opposite direction)
        if exit_road == primary_exit_road:
            # Determine if same direction (within 90 degrees) or opposite
            heading_diff = abs(exit_heading - primary_exit_heading)
            heading_diff = min(heading_diff, 360 - heading_diff)
            is_same_direction = heading_diff < 90
            parallel_vehs.append((veh_num, is_same_direction, 'append'))
        
        # Also check if this vehicle ENTERS on the primary's exit road (opposite direction)
        # This catches vehicles that start on the primary's exit road and turn off
        elif entry_road == primary_exit_road:
            # This vehicle enters where the primary exits - opposite direction
            # Check heading to confirm it's traveling opposite
            heading_diff = abs(entry_heading - primary_exit_heading)
            heading_diff = min(heading_diff, 360 - heading_diff)
            if heading_diff > 90:  # Opposite direction (more than 90 degrees different)
                parallel_vehs.append((veh_num, False, 'prepend'))
    
    return parallel_vehs


def _extend_parallel_paths(
    primary_veh_num: int,
    extended_pts: np.ndarray,
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    primary_end_pt_before_extension: np.ndarray,
) -> List[int]:
    """
    Extend parallel vehicles' paths using the same extension points.
    
    When we extend the primary vehicle's path, we want parallel vehicles
    (same exit road) to also extend so they can interact with entities 
    placed in the extended region.
    
    For same-direction vehicles: append offset extension points to the end
    For opposite-direction vehicles: prepend reversed extension points to the start
    
    Returns list of vehicle numbers that were extended.
    """
    parallel_vehs = _find_parallel_vehicles(primary_veh_num, picked_list)
    extended_vehs = []
    
    if len(extended_pts) == 0:
        return extended_vehs
    
    for veh_num, is_same_direction, extend_mode in parallel_vehs:
        picked_entry = None
        for p in picked_list:
            if _parse_vehicle_num(p.get("vehicle")) == veh_num:
                picked_entry = p
                break
        
        if picked_entry is None:
            continue
        
        sig = picked_entry.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        if not seg_ids:
            continue
        
        if extend_mode == 'append':
            # Same direction: extend from the END of the path
            last_seg_id = int(seg_ids[-1])
            last_pts = seg_by_id.get(last_seg_id)
            if last_pts is None or len(last_pts) < 2:
                continue
            
            last_pts = np.asarray(last_pts, dtype=float)
            end_pt = last_pts[-1]
            
            # Compute offset from primary vehicle's endpoint (before extension) to this vehicle's endpoint
            # This maintains the lane offset regardless of how far apart they are
            offset = end_pt - primary_end_pt_before_extension
            
            # Apply the offset to each extension point
            offset_pts = extended_pts + offset
            
            # Append to this vehicle's last segment
            new_pts = np.vstack([last_pts, offset_pts])
            seg_by_id[last_seg_id] = new_pts
            
            # Update segments_detailed
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == last_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the exit point
            new_end_pt = new_pts[-1]
            new_end_heading = _heading_at_end(new_pts)
            if "exit" in sig:
                sig["exit"]["point"] = {"x": float(new_end_pt[0]), "y": float(new_end_pt[1])}
                sig["exit"]["heading_deg"] = float(new_end_heading)
            
            extended_vehs.append(veh_num)
            print(f"[INFO] Extended parallel Vehicle {veh_num} (same dir) with {len(extended_pts)} waypoints")
            
        elif extend_mode == 'prepend':
            # Opposite direction relative to the primary: extend at this vehicle's ENTRY end.
            first_seg_id = int(seg_ids[0])  # Entry segment for this vehicle
            first_pts = seg_by_id.get(first_seg_id)
            if first_pts is None or len(first_pts) < 2:
                continue
            
            first_pts = np.asarray(first_pts, dtype=float)

            entry_pt = None
            if isinstance(sig.get("entry", {}), dict):
                ep = sig.get("entry", {}).get("point", {})
                if isinstance(ep, dict) and "x" in ep and "y" in ep:
                    entry_pt = np.array([float(ep["x"]), float(ep["y"])], dtype=float)

            start_pt = first_pts[0]
            end_pt = first_pts[-1]
            if entry_pt is None:
                entry_pt = start_pt

            d_start = float(np.linalg.norm(entry_pt - start_pt))
            d_end = float(np.linalg.norm(entry_pt - end_pt))
            entry_at_start = d_start <= d_end
            entry_end_pt = start_pt if entry_at_start else end_pt

            # Compute offset from primary's old end to this vehicle's entry endpoint
            offset = entry_end_pt - primary_end_pt_before_extension
            offset_pts = extended_pts + offset

            if len(offset_pts) >= 2:
                if entry_at_start:
                    # Ensure the last extension point lands near the existing entry.
                    if float(np.linalg.norm(offset_pts[-1] - entry_end_pt)) > float(np.linalg.norm(offset_pts[0] - entry_end_pt)):
                        offset_pts = offset_pts[::-1]
                    new_pts = np.vstack([offset_pts, first_pts])
                else:
                    # Entry is at the array end; extend after it and keep continuity.
                    if float(np.linalg.norm(offset_pts[0] - entry_end_pt)) > float(np.linalg.norm(offset_pts[-1] - entry_end_pt)):
                        offset_pts = offset_pts[::-1]
                    new_pts = np.vstack([first_pts, offset_pts])
            else:
                new_pts = np.vstack([offset_pts, first_pts]) if entry_at_start else np.vstack([first_pts, offset_pts])
            seg_by_id[first_seg_id] = new_pts
            
            # Update segments_detailed
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == first_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the entry point (where the opposite-direction vehicle actually starts)
            new_start_pt = new_pts[0] if entry_at_start else new_pts[-1]
            new_start_heading = _heading_at_start(new_pts) if entry_at_start else _heading_at_end(new_pts)
            if "entry" in sig:
                sig["entry"]["point"] = {"x": float(new_start_pt[0]), "y": float(new_start_pt[1])}
                if entry_at_start:
                    sig["entry"]["heading_deg"] = float(new_start_heading)
                else:
                    # Entry at array end implies travel opposite array order.
                    sig["entry"]["heading_deg"] = float((new_start_heading + 180) % 360)
            
            extended_vehs.append(veh_num)
            print(f"[INFO] Extended parallel Vehicle {veh_num} (opposite dir) with {len(extended_pts)} waypoints appended to entry")
    
    return extended_vehs


def extend_path_if_needed(
    picked_entry: Dict[str, Any],
    seg_by_id: Dict[int, np.ndarray],
    all_segments: List[Dict[str, Any]],
    target_length: float,
    max_extensions: int = 5,
    nodes: Optional[Dict[str, Any]] = None,
    picked_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, float, List[int]]:
    """
    Extend a vehicle's path geometry if it's shorter than target_length.
    
    Works purely with geometry - finds waypoints that continue the path
    and appends them directly to the last segment's polyline.
    
    Also extends parallel vehicles (same exit road) with the same extension points.
    
    Modifies seg_by_id in-place (extends the last segment's geometry).
    
    Returns (was_extended, new_total_length, list_of_parallel_vehicles_extended)
    """
    sig = picked_entry.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    
    # Get the primary vehicle number for extending parallel paths
    primary_veh_num = _parse_vehicle_num(picked_entry.get("vehicle"))
    
    if not seg_ids:
        return False, 0.0, []
    
    current_length = _compute_path_length(picked_entry, seg_by_id)
    if current_length >= target_length:
        return False, current_length, []
    
    # Get the last segment's geometry
    last_seg_id = int(seg_ids[-1])
    last_pts = seg_by_id.get(last_seg_id)
    if last_pts is None or len(last_pts) < 2:
        return False, current_length, []
    
    last_pts = np.asarray(last_pts, dtype=float)
    end_pt = last_pts[-1]
    end_heading = _heading_at_end(last_pts)
    
    # Save the endpoint BEFORE extension for computing offsets for parallel vehicles
    primary_end_pt_before_extension = end_pt.copy()
    
    extended_parallel_vehs = []
    
    # If we have raw nodes, use them to find continuation waypoints
    if nodes is not None and "payload" in nodes:
        extended_pts = _extend_path_from_nodes(
            end_pt, end_heading, nodes,
            target_extension=target_length - current_length + 10.0,  # Add margin
            connect_radius_m=8.0,
            connect_yaw_tol_deg=30.0,  # Strict for straight-through
        )
        
        if len(extended_pts) > 0:
            # Append extended points to the last segment
            new_pts = np.vstack([last_pts, extended_pts])
            seg_by_id[last_seg_id] = new_pts
            
            # Also update segments_detailed polyline_sample if present
            segments_detailed = sig.get("segments_detailed", [])
            for sd in segments_detailed:
                if sd.get("seg_id") == last_seg_id:
                    sd["polyline_sample"] = _polyline_sample_from_pts(new_pts, max_points=20)
                    sd["length_m"] = float(_segment_length(new_pts))
            
            # Update the exit point in the signature to reflect the new end
            new_end_pt = new_pts[-1]
            new_end_heading = _heading_at_end(new_pts)
            if "exit" in sig:
                sig["exit"]["point"] = {"x": float(new_end_pt[0]), "y": float(new_end_pt[1])}
                sig["exit"]["heading_deg"] = float(new_end_heading)
            
            new_length = _compute_path_length(picked_entry, seg_by_id)
            print(f"[INFO] Path extension: extended from {current_length:.1f}m to {new_length:.1f}m "
                  f"(added {len(extended_pts)} waypoints)")
            
            # Extend parallel vehicles with the same extension points
            if picked_list is not None and primary_veh_num is not None:
                extended_parallel_vehs = _extend_parallel_paths(
                    primary_veh_num, extended_pts, picked_list, seg_by_id,
                    primary_end_pt_before_extension
                )
            
            return True, new_length, extended_parallel_vehs
    
    # Fallback: try segment-based extension (original approach)
    excluded = set(int(s) for s in seg_ids)
    extensions = 0
    fallback_extended_pts = []
    
    while current_length < target_length and extensions < max_extensions:
        end_pt = last_pts[-1]
        end_heading = _heading_at_end(last_pts)
        
        next_seg = _find_best_successor_segment(
            end_pt, end_heading, all_segments, excluded,
            connect_radius_m=10.0,
            connect_yaw_tol_deg=45.0,
        )
        
        if next_seg is None:
            break
        
        next_seg_id = int(next_seg["seg_id"])
        next_pts = np.asarray(next_seg["points"], dtype=float)
        
        # Append to last segment geometry
        new_pts = np.vstack([last_pts, next_pts])
        seg_by_id[last_seg_id] = new_pts
        last_pts = new_pts
        fallback_extended_pts = np.vstack([fallback_extended_pts, next_pts]) if len(fallback_extended_pts) > 0 else next_pts
        
        excluded.add(next_seg_id)
        current_length = _segment_length(new_pts)
        extensions += 1
        
        print(f"[INFO] Path extension: added segment {next_seg_id}, path now {current_length:.1f}m")
    
    new_length = _compute_path_length(picked_entry, seg_by_id)
    
    # Extend parallel vehicles with the same extension points (fallback path)
    if extensions > 0 and picked_list is not None and primary_veh_num is not None and len(fallback_extended_pts) > 0:
        extended_parallel_vehs = _extend_parallel_paths(
            primary_veh_num, np.asarray(fallback_extended_pts), picked_list, seg_by_id,
            primary_end_pt_before_extension
        )
    
    return extensions > 0, new_length, extended_parallel_vehs


def _extend_path_from_nodes(
    start_pt: np.ndarray,
    start_heading: float,
    nodes: Dict[str, Any],
    target_extension: float,
    connect_radius_m: float = 8.0,
    connect_yaw_tol_deg: float = 30.0,
) -> np.ndarray:
    """
    Find waypoints from raw nodes that continue the path from start_pt/start_heading.
    
    Returns array of (N, 2) points to append to the path.
    """
    payload = nodes.get("payload", {})
    x = np.asarray(payload.get("x", []), dtype=float)
    y = np.asarray(payload.get("y", []), dtype=float)
    yaw = np.asarray(payload.get("yaw", []), dtype=float)
    
    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    
    all_pts = np.vstack([x, y]).T
    
    # Find starting candidates: close to start_pt and heading-aligned
    dists = np.linalg.norm(all_pts - start_pt, axis=1)
    close_mask = dists < connect_radius_m
    
    if not np.any(close_mask):
        return np.empty((0, 2), dtype=float)
    
    # Check heading alignment
    heading_diffs = np.abs(np.mod(yaw[close_mask] - start_heading + 180, 360) - 180)
    aligned_mask = heading_diffs < connect_yaw_tol_deg
    
    close_indices = np.where(close_mask)[0]
    aligned_indices = close_indices[aligned_mask]
    
    if len(aligned_indices) == 0:
        return np.empty((0, 2), dtype=float)
    
    # Pick the closest aligned point as the seed
    seed_idx = aligned_indices[np.argmin(dists[aligned_indices])]
    
    # Greedy walk: follow waypoints that continue in roughly the same direction
    extended_points = []
    current_pt = all_pts[seed_idx]
    current_heading = float(yaw[seed_idx])
    visited = {seed_idx}
    total_dist = 0.0
    
    while total_dist < target_extension:
        # Find next waypoint: ahead of current, aligned heading
        candidates = []
        for i in range(len(x)):
            if i in visited:
                continue
            pt = all_pts[i]
            
            # Must be ahead (in direction of travel)
            vec_to_pt = pt - current_pt
            dist_to_pt = np.linalg.norm(vec_to_pt)
            if dist_to_pt < 0.5 or dist_to_pt > 15.0:  # Skip too close or too far
                continue
            
            # Check if it's roughly ahead (within 60° of current heading)
            angle_to_pt = np.degrees(np.arctan2(vec_to_pt[1], vec_to_pt[0]))
            angle_diff = abs(wrap180(angle_to_pt - current_heading))
            if angle_diff > 60:
                continue
            
            # Check heading alignment of the waypoint itself
            heading_diff = abs(wrap180(float(yaw[i]) - current_heading))
            if heading_diff > connect_yaw_tol_deg:
                continue
            
            candidates.append((i, dist_to_pt, angle_diff))
        
        if not candidates:
            break
        
        # Pick the best candidate (closest and most aligned)
        candidates.sort(key=lambda c: (c[2], c[1]))  # Sort by angle_diff, then distance
        best_idx = candidates[0][0]
        
        next_pt = all_pts[best_idx]
        step_dist = np.linalg.norm(next_pt - current_pt)
        
        extended_points.append(next_pt)
        visited.add(best_idx)
        total_dist += step_dist
        current_pt = next_pt
        current_heading = float(yaw[best_idx])
    
    if not extended_points:
        return np.empty((0, 2), dtype=float)
    
    return np.array(extended_points, dtype=float)


def extend_paths_for_entity_spacing(
    actor_specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    all_segments: List[Dict[str, Any]],
) -> Dict[int, List[int]]:
    """
    Extend vehicle paths as needed to accommodate entity spacing requirements.
    
    Analyzes actor_specs for ahead_of/behind_of relations and extends paths
    that are too short.
    
    Returns dict mapping vehicle_num -> list of added segment IDs
    """
    # Estimate required lengths
    required_lengths = _estimate_required_path_length(actor_specs, picked_list, seg_by_id)
    
    extensions_made: Dict[int, List[int]] = {}
    
    for picked in picked_list:
        veh_num = _parse_vehicle_num(picked.get("vehicle"))
        if veh_num is None:
            continue
        
        current_len = _compute_path_length(picked, seg_by_id)
        target_len = required_lengths.get(veh_num, current_len)
        
        if target_len > current_len:
            print(f"[INFO] Vehicle {veh_num}: current path {current_len:.1f}m, "
                  f"need ~{target_len:.1f}m for entity spacing")
            
            was_extended, added_ids = extend_path_if_needed(
                picked, seg_by_id, all_segments, target_len
            )
            
            if was_extended:
                extensions_made[veh_num] = added_ids
                new_len = _compute_path_length(picked, seg_by_id)
                print(f"[INFO] Vehicle {veh_num}: extended to {new_len:.1f}m")
    
    return extensions_made


# ======================================================================================
# Prompt building
# ======================================================================================

LATERAL_RELATIONS = [
    "center",
    "half_right",
    "right_edge",
    "offroad_right",
    "half_left",
    "left_edge",
    "offroad_left",
]

# Lane width heuristic (meters)
LANE_WIDTH_M = 3.5

# Map qualitative to meters (right positive using right_normal_world)
LATERAL_TO_M = {
    "center": 0.0,
    "half_right": +0.25 * LANE_WIDTH_M,
    "right_edge": +0.45 * LANE_WIDTH_M,
    "offroad_right": +1.10 * LANE_WIDTH_M,
    "half_left": -0.25 * LANE_WIDTH_M,
    "left_edge": -0.45 * LANE_WIDTH_M,
    "offroad_left": -1.10 * LANE_WIDTH_M,
}

def build_vehicle_segment_summaries(picked: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a small, LLM-friendly summary of each ego vehicle's segments,
    preserving segment order (segment_index is 1-based).
    """
    out: Dict[str, Any] = {}
    for p in picked:
        veh = str(p.get("vehicle", "Vehicle"))
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        mans = sig.get("maneuvers_between", []) if isinstance(sig.get("maneuvers_between", []), list) else []
        summary = []
        for i, s in enumerate(segs):
            if not isinstance(s, dict):
                continue
            entry_man = None
            exit_man = None
            if i > 0 and i - 1 < len(mans):
                entry_man = mans[i - 1]
            if i < len(mans):
                exit_man = mans[i]
            summary.append({
                "segment_index": i + 1,
                "seg_id": s.get("seg_id"),
                "road_id": s.get("road_id"),
                "lane_id": s.get("lane_id"),
                "length_m": s.get("length_m"),
                "start_cardinal4": (s.get("start", {}) or {}).get("cardinal4"),
                "end_cardinal4": (s.get("end", {}) or {}).get("cardinal4"),
                "maneuver_into_segment": entry_man,   # None for first segment
                "maneuver_out_of_segment": exit_man,  # None for last segment
            })
        out[veh] = {
            "path_name": p.get("name"),
            "num_segments": sig.get("num_segments"),
            "segments": summary,
        }
    return out

def fewshot_examples() -> str:
    # Removed: few-shot examples were causing LLM to hallucinate actors not in Stage 1.
    # The schema in the prompt payload is sufficient.
    return ""

def build_stage1_prompt(description: str) -> str:
    """
    Stage 1: entity extraction, without forcing exact segment indices.
    Each entity gets a unique entity_id that Stage 2 MUST use.
    """
    return (
        "You will read a short driving scene description.\n"
        "\n"
        "Extract ALL non-ego actors that should be spawned as obstacles or interacting entities.\n"
        "Assign each entity a UNIQUE entity_id starting from 'entity_1', 'entity_2', etc.\n"
        "For each extracted entity, include an 'evidence' field: an EXACT quote (<=20 words) from the description\n"
        "that clearly mentions the actor (not just the location).\n"
        "\n"
        "\n"
        "CRITICAL RULES - READ CAREFULLY:\n"
        "\n"
        "1. 'Vehicle 1', 'Vehicle 2', 'Vehicle 3', etc. are ALWAYS ego vehicles.\n"
        "   They are NEVER extracted. They already exist in the simulation.\n"
        "\n"
        "2. Descriptions of ego vehicle maneuvers (straight, left, right, turns, continues)\n"
        "   are NOT actors to extract. These just describe what the ego vehicles do.\n"
        "\n"
        "3. Only extract ADDITIONAL actors that are:\n"
        "   - Static props (traffic cones, barriers, debris, boxes)\n"
        "   - Parked/stopped vehicles (parked car, delivery truck blocking lane)\n"
        "   - Pedestrians/walkers (person crossing, pedestrian on sidewalk)\n"
        "   - Cyclists/bicycles\n"
        "   - NPC vehicles that interact (vehicle that yields, cuts in, merges)\n"
        "\n"
        "4. If the description ONLY talks about ego vehicles and their routes,\n"
        "   with no obstacles, pedestrians, or interacting vehicles mentioned,\n"
        "   return: {\"entities\": []}\n"
        "\n"
        "EXAMPLES OF WHAT TO EXTRACT:\n"
        "- '5 traffic cones in the right side of its lane' -> extract as static_prop, quantity=5\n"
        "- 'multiple traffic cones' -> extract as static_prop, quantity=5 (default for 'multiple')\n"
        "- 'a bicyclist in the middle of its lane' -> extract as cyclist, quantity=1\n"
        "- 'a parked truck blocking the lane' -> extract as parked_vehicle, quantity=1\n"
        "- 'a pedestrian crosses the road' -> extract as walker, quantity=1\n"
        "\n"
        "EXAMPLES OF WHAT NOT TO EXTRACT:\n"
        "- 'Vehicle 1 continues straight' -> ego vehicle, do NOT extract\n"
        "- 'Another vehicle turns left' -> ego vehicle (Vehicle 2/3), do NOT extract\n"
        "- 'Vehicle 2 is coming from the same direction' -> ego vehicle, do NOT extract\n"
        "\n"
        "MOTION_HINT DEFINITIONS (use carefully):\n"
        "- 'static': does not move (parked vehicles, cones, barriers)\n"
        "- 'crossing': moves ACROSS the road, perpendicular to traffic (pedestrian crossing street)\n"
        "- 'follow_lane': moves ALONG the road in the lane direction (walks/rides in direction of road)\n"
        "- 'unknown': motion not specified\n"
        "\n"
        "CROSSING_DIRECTION (for crossing motion ONLY):\n"
        "- 'left': crosses FROM right TO left (direction of movement is leftward)\n"
        "- 'right': crosses FROM left TO right (direction of movement is rightward)\n"
        "- null: not specified or not applicable\n"
        "\n"
        "IMPORTANT: 'walks in the direction of the road' = follow_lane (ALONG the road)\n"
        "           'crosses the road' = crossing (ACROSS the road)\n"
        "\n"
        "QUANTITY EXTRACTION:\n"
        "- If a number is given (e.g., '5 cones'), use that number.\n"
        "- 'multiple', 'several', 'some' -> use quantity=5 as default.\n"
        "- 'a few' -> use quantity=3.\n"
        "- 'a', 'an', 'one', or no quantifier -> use quantity=1.\n"
        "\n"
        "GROUP_PATTERN (for quantity > 1):\n"
        "- 'across_lane': objects arranged side-by-side across the lane width\n"
        "- 'along_lane': objects arranged in a line along the direction of travel\n"
        "- 'diagonal': objects arranged diagonally (e.g., 'starting right, ending left')\n"
        "- 'unknown': arrangement not specified\n"
        "\n"
        "SPEED_HINT:\n"
        "- 'stopped': not moving (parked, standing still)\n"
        "- 'slow': moving slowly\n"
        "- 'normal': moving at normal speed\n"
        "- 'fast': moving fast\n"
        "- 'erratic': driving erratically, unpredictably, or aggressively\n"
        "- 'unknown': speed/behavior not specified\n"
        "\n"
        "DIRECTION_RELATIVE_TO (for NPC vehicles ONLY):\n"
        "- 'same': NPC travels in the same direction as the reference vehicle\n"
        "- 'opposite': NPC travels in the opposite direction (oncoming traffic)\n"
        "- null: not applicable or not specified\n"
        "\n"
        "Return JSON ONLY with this schema:\n"
        "IMPORTANT: The \'mention\' field MUST be an EXACT substring copied from the DESCRIPTION (no paraphrase).\n"
        "- Example: if DESCRIPTION says \'cyclist\', do NOT output \'bicyclist\'; copy \'cyclist\'.\n"
        "{\n"
        "  \"entities\": [\n"
        "    {\n"
        "      \"entity_id\": \"entity_1\",\n"
        "      \"mention\": \"...\",\n"
        "      \"evidence\": \"EXACT quote (<=20 words) from description that mentions this actor\",\n"
        "      \"actor_kind\": \"static_prop\" | \"parked_vehicle\" | \"walker\" | \"cyclist\" | \"npc_vehicle\",\n"
        "      \"quantity\": 1,\n"
        "      \"group_pattern\": \"across_lane\" | \"along_lane\" | \"diagonal\" | \"unknown\",\n"
        "      \"start_lateral\": \"right_edge\" | \"half_right\" | \"center\" | \"half_left\" | \"left_edge\" | null,\n"
        "      \"end_lateral\": \"right_edge\" | \"half_right\" | \"center\" | \"half_left\" | \"left_edge\" | null,\n"
        "      \"affects_vehicle\": \"Vehicle 1\" | \"Vehicle 2\" | \"Vehicle 3\" | null,\n"
        "      \"when\": \"on_approach\" | \"after_turn\" | \"in_intersection\" | \"after_exit\" | \"after_merge\" | \"unknown\",\n"
        "      \"lateral_relation\": \"center\" | \"half_right\" | \"right_edge\" | \"half_left\" | \"left_edge\" | \"unknown\",\n"
        "      \"motion_hint\": \"static\" | \"crossing\" | \"follow_lane\" | \"unknown\",\n"
        "      \"crossing_direction\": \"left\" | \"right\" | null,\n"
        "      \"speed_hint\": \"stopped\" | \"slow\" | \"normal\" | \"fast\" | \"erratic\" | \"unknown\",\n"
        "      \"direction_relative_to\": {\"vehicle\": \"Vehicle 1\", \"direction\": \"same\" | \"opposite\"} | null\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        f"DESCRIPTION:\n{description}\n"
    )

def build_stage2_prompt(
    description: str,
    vehicle_segments: Dict[str, Any],
    entities: List[Dict[str, Any]],
    per_entity_asset_options: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    Stage 2: resolve entities to concrete segment anchors + pick asset ids.
    Entities come from Stage 1 and each has a unique entity_id.
    """
    # Build list of valid entity_ids for the prompt
    valid_entity_ids = [e.get("entity_id", f"entity_{i+1}") for i, e in enumerate(entities)]
    
    payload = {
        "vehicle_segments": vehicle_segments,
        "entities": entities,
        "valid_entity_ids": valid_entity_ids,
        "lateral_relations_allowed": LATERAL_RELATIONS,
        "asset_options_per_entity": per_entity_asset_options,
        "notes": [
            "CRITICAL: Output AT MOST one actor per entity_id. If an entity seems invalid/unspawnable (e.g., map geometry), omit it.",
            "CRITICAL: Each actor's 'entity_id' field MUST match one of the valid_entity_ids.",
            "CRITICAL: Do NOT invent new actors. Only place the entities provided.",
            "If 'entities' is empty, return {\"actors\": []}.",
            "segment_index is 1-based and must be valid for the target_vehicle.",
            "s_along is a fraction in [0,1] along that segment.",
            "If something is 'after_turn' and the target vehicle has a non-straight maneuver (left/right/u-turn), place it on the post-turn (exit) segment. In the common 3-segment representation (approach, turn-connector, exit), this is typically segment_index=3 (exit), not segment_index=2 (turn connector).",
            "If something is 'after the merge region' or 'after merge', place it on the EXIT segment (segment_index=3 for a 3-segment path) since merges happen on or after the intersection exit. Use s_along > 0.5 for objects farther down the exit road.",
            "If unsure, still choose the best guess and use lower confidence.",
            "asset_id MUST be copied EXACTLY from the provided options for that entity.",
            "Include a confidence value in [0,1] for each actor; do not omit it.",
        ],
        "required_output_schema": {
            "actors": [
                {
                    "entity_id": "entity_1",
                    "semantic": "<from entity mention>",
                    "category": "vehicle|walker|static",
                    "asset_id": "<from asset_options_per_entity>",
                    "placement": {
                        "frame": "segment",
                        "target_vehicle": "Vehicle X",
                        "segment_index": 1,
                        "s_along": 0.5,
                        "lateral_relation": "center"
                    },
                    "motion": {
                        "type": "static|cross_perpendicular|follow_lane|straight_line",
                        "speed_profile": "stopped|slow|normal|fast|erratic"
                    },
                    "confidence": 0.9
                }
            ]
        }
    }

    return (
        "You are placing non-ego actors into a road network described by picked ego paths.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1) You MUST output EXACTLY one actor for each entity in the 'entities' list.\n"
        "2) Each actor MUST have an 'entity_id' that matches one from valid_entity_ids.\n"
        "3) Do NOT create any actors that are not in the entities list.\n"
        "4) If the entities list is empty, return {\"actors\": []}.\n"
        "\n"
        "For each entity, you must:\n"
        "1) Copy its entity_id to the actor.\n"
        "2) Choose a concrete segment anchor (target_vehicle + segment_index + s_along + lateral_relation).\n"
        "3) Choose an asset_id EXACTLY from the provided options for that entity.\n"
        "4) Provide motion:\n"
        "   - static: for parked/stationary objects.\n"
        "   - cross_perpendicular: for actors crossing the ego lane.\n"
        "   - follow_lane: for actors moving along the lane direction.\n"
        "   - straight_line: for actors moving between two anchors.\n"
        "\n"
        "Return JSON ONLY with top-level key 'actors'.\n"
        "\n"
        "INPUT (JSON):\n"
        + json.dumps(payload, indent=2)
        + "\n\nSCENE DESCRIPTION:\n"
        + description
        + "\n"
    )


# ======================================================================================
# LLM runner
# ======================================================================================

def generate_with_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_len = int(input_ids.shape[-1])
    else:
        enc = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = enc
        input_len = int(enc["input_ids"].shape[-1])

    # Log token budget so we can see headroom for generation.
    model_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    headroom = None if model_ctx is None else max(0, int(model_ctx) - input_len)
    allowed_new = None if headroom is None else max(0, headroom)
    print(
        f"[DEBUG] token budget: prompt={input_len}"
        + (f", model_ctx={model_ctx}" if model_ctx is not None else ", model_ctx=unknown")
        + f", requested_new={max_new_tokens}"
        + (f", max_new_before_ctx={allowed_new}" if allowed_new is not None else ""),
        flush=True,
    )

    if not do_sample:
        temperature = None
        top_p = None

    with torch.no_grad():
        out = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # IMPORTANT: decode only the newly generated tokens, not the echoed prompt.
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ======================================================================================
# Validation + repair prompting
# ======================================================================================



# --------------------------
# Post-processing guardrails
# --------------------------

def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0

def _heading_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))

def _infer_heading_change_deg(points: np.ndarray) -> float:
    """Approximate how much a segment turns (deg) from its start tangent to end tangent."""
    if points is None or len(points) < 3:
        return 0.0
    v0 = points[1] - points[0]
    v1 = points[-1] - points[-2]
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    h0 = _heading_deg(v0 / n0)
    h1 = _heading_deg(v1 / n1)
    return abs(_wrap180(h1 - h0))

def _segment_length_m(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))

def _infer_vehicle_turn_exit_indices(
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    turn_deg_threshold: float = 35.0,
) -> Dict[int, Dict[str, Any]]:
    """Infer which segments are 'turn' vs 'exit' per vehicle using geometry."""
    out: Dict[int, Dict[str, Any]] = {}
    for p in picked_list:
        veh_str = p.get("vehicle", "")
        m = re.search(r"(\d+)", veh_str)
        if not m:
            continue
        veh_num = int(m.group(1))
        seg_ids = [int(x) for x in p.get("signature", {}).get("segment_ids", [])]
        seg_lens: List[float] = []
        heading_changes: List[float] = []
        for sid in seg_ids:
            pts = seg_by_id.get(int(sid))
            seg_lens.append(_segment_length_m(pts))
            heading_changes.append(_infer_heading_change_deg(pts))

        turn_indices = {i + 1 for i, d in enumerate(heading_changes) if d >= turn_deg_threshold}

        # Fallback: if the path is non-straight but geometry didn't cross the threshold (rare),
        # assume segment 2 is the turn connector.
        entry_to_exit_turn = p.get("signature", {}).get("entry_to_exit_turn", "straight")
        if not turn_indices and entry_to_exit_turn in ("left", "right", "u_turn") and len(seg_ids) >= 2:
            turn_indices = {2}

        exit_index = None
        if turn_indices:
            last_turn = max(turn_indices)
            if last_turn < len(seg_ids):
                exit_index = last_turn + 1
            else:
                exit_index = last_turn

        out[veh_num] = {
            "turn_indices": turn_indices,
            "exit_index": exit_index,
            "seg_ids": seg_ids,
            "seg_lens": seg_lens,
        }
    return out

def _best_entity_match(actor_semantic: str, entities: List[Dict[str, Any]], target_vehicle: int) -> Optional[Dict[str, Any]]:
    """Match a Stage2 actor back to a Stage1 entity (for 'when' semantics)."""
    if not actor_semantic or not entities:
        return None
    a = actor_semantic.strip().lower()
    best = None
    best_score = 0.0
    for e in entities:
        if not isinstance(e, dict):
            continue
        ev = e.get("affects_vehicle", None)
        if ev is not None and ev != target_vehicle:
            continue
        mention = str(e.get("mention", "")).strip().lower()
        if not mention:
            continue
        a_tokens = set(re.findall(r"[a-z0-9]+", a))
        m_tokens = set(re.findall(r"[a-z0-9]+", mention))
        jacc = (len(a_tokens & m_tokens) / max(1, len(a_tokens | m_tokens)))
        seq = difflib.SequenceMatcher(None, a, mention).ratio()
        score = 0.6 * seq + 0.4 * jacc
        if score > best_score:
            best_score = score
            best = e
    if best_score < 0.30:
        return None
    return best

def apply_after_turn_segment_corrections(
    actors: List[Dict[str, Any]],
    stage1_entities: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> None:
    """In-place correction for a common failure mode:
    If an entity says 'after_turn' or 'after_merge' but Stage2 placed the actor on a turning segment,
    shift it to the inferred post-turn (exit) segment, preserving distance-from-segment-start in meters.
    """
    veh_info = _infer_vehicle_turn_exit_indices(picked_list, seg_by_id)

    for a in actors:
        placement = a.get("placement", {})
        try:
            tv = int(placement.get("target_vehicle", 0))
            seg_idx = int(placement.get("segment_index", 0))
            s_along = float(placement.get("s_along", 0.0))
        except Exception:
            continue

        if tv <= 0 or seg_idx <= 0:
            continue

        e = _best_entity_match(str(a.get("semantic", "")), stage1_entities, tv)
        if not e:
            continue

        when_phase = e.get("when", None)
        if when_phase not in ("after_turn", "after_merge"):
            continue

        info = veh_info.get(tv)
        if not info:
            continue

        turn_indices = info.get("turn_indices", set())
        exit_idx = info.get("exit_index", None)
        if not exit_idx:
            continue

        if seg_idx not in turn_indices or exit_idx == seg_idx:
            continue

        seg_lens = info.get("seg_lens", [])
        if seg_idx - 1 >= len(seg_lens) or exit_idx - 1 >= len(seg_lens):
            continue
        old_len = float(seg_lens[seg_idx - 1])
        new_len = float(seg_lens[exit_idx - 1])
        if old_len < 1e-6 or new_len < 1e-6:
            continue

        offset_m = float(np.clip(s_along * old_len, 0.0, old_len))
        new_s = float(np.clip(offset_m / new_len, 0.02, 0.98))

        placement["segment_index"] = int(exit_idx)
        placement["s_along"] = new_s

        motion = a.get("motion", {})
        if isinstance(motion, dict) and "anchor_s_along" in motion:
            motion["anchor_s_along"] = new_s
def validate_stage2_output(actors: List[Dict[str, Any]], vehicle_segments: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for i, a in enumerate(actors):
        if not isinstance(a, dict):
            errors.append(f"actors[{i}] is not an object")
            continue

        for k in ("id", "semantic", "category", "asset_id", "placement", "motion", "confidence"):
            if k not in a:
                errors.append(f"actors[{i}] missing key '{k}'")

        placement = a.get("placement", {})
        if not isinstance(placement, dict):
            errors.append(f"actors[{i}].placement must be an object")
            continue

        if placement.get("frame") != "segment":
            errors.append(f"actors[{i}].placement.frame must be 'segment' (for now)")
            continue

        tv = placement.get("target_vehicle")
        if tv not in vehicle_segments:
            errors.append(f"actors[{i}] target_vehicle '{tv}' not found in vehicle_segments")
            continue

        seg_idx = placement.get("segment_index")
        if not isinstance(seg_idx, int):
            errors.append(f"actors[{i}] segment_index must be int")
            continue

        num = vehicle_segments[tv].get("num_segments")
        if isinstance(num, int) and (seg_idx < 1 or seg_idx > num):
            errors.append(f"actors[{i}] segment_index {seg_idx} out of range [1,{num}] for {tv}")

        s_along = placement.get("s_along")
        if not isinstance(s_along, (int, float)) or not (0.0 <= float(s_along) <= 1.0):
            errors.append(f"actors[{i}] s_along must be in [0,1]")

        lat = placement.get("lateral_relation")
        if lat not in LATERAL_RELATIONS:
            errors.append(f"actors[{i}] lateral_relation '{lat}' invalid")

        conf = a.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"actors[{i}] confidence must be in [0,1]")

        motion = a.get("motion", {})
        if not isinstance(motion, dict):
            errors.append(f"actors[{i}].motion must be an object")
            continue
        mtype = motion.get("type")
        # Accept common alias and normalize.
        if mtype == "following_lane":
            motion["type"] = "follow_lane"
            mtype = "follow_lane"
        if mtype not in ("static", "cross_perpendicular", "follow_lane", "straight_line"):
            errors.append(f"actors[{i}].motion.type invalid")

    return errors

def build_repair_prompt(bad_json_text: str, errors: List[str]) -> str:
    return (
        "You returned JSON but it failed validation.\n"
        "Fix ONLY the JSON to satisfy the constraints. Return JSON ONLY.\n"
        "\n"
        "VALIDATION ERRORS:\n"
        + "\n".join([f"- {e}" for e in errors])
        + "\n\nBAD JSON:\n"
        + bad_json_text
        + "\n"
    )


# ======================================================================================
# Placement -> concrete world transforms and motion waypoints
# ======================================================================================

def infer_speed_mps(category: str, speed_profile: str) -> float:
    # Baselines: walkers ~1.7, cyclists ~4.0, vehicles ~8.0
    speed_profile = (speed_profile or "normal").lower()
    base = 1.7 if category == "walker" else (4.0 if category == "cyclist" else 8.0)
    if speed_profile in ("stopped", "stop", "static"):
        return 0.0
    if speed_profile == "slow":
        return 0.6 * base
    if speed_profile == "fast":
        return 1.6 * base
    if speed_profile == "erratic":
        # Erratic uses a higher base speed (actual speed varies in waypoint generation)
        return 1.3 * base
    return base

def resolve_nodes_path(picked_path_json_path: str, nodes_field: str, nodes_root: Optional[str]) -> str:
    # If nodes_field is absolute, use it. Otherwise resolve relative to picked_paths_detailed.json dir or --nodes-root.
    if os.path.isabs(nodes_field) and os.path.exists(nodes_field):
        return nodes_field
    # Try nodes_root first (if provided)
    if nodes_root:
        cand = os.path.join(nodes_root, nodes_field)
        if os.path.exists(cand):
            return cand
    # Resolve relative to the picked json file location
    base = os.path.dirname(os.path.abspath(picked_path_json_path))
    cand = os.path.join(base, nodes_field)
    if os.path.exists(cand):
        return cand
    # Final fallback: keep as-is (caller may want to handle)
    return nodes_field

def compute_spawn_from_anchor(
    seg_points: np.ndarray,
    s_along: float,
    lateral_relation: str,
    lateral_offset_m: Optional[float] = None,
) -> Dict[str, float]:
    p, t = point_and_tangent_at_s(seg_points, s_along)
    yaw = wrap180(heading_deg_from_vec(t))
    lat_m = float(lateral_offset_m) if lateral_offset_m is not None else float(LATERAL_TO_M.get(lateral_relation, 0.0))

    if abs(lat_m) > 1e-9:
        if lat_m > 0:
            n = right_normal_world(t)
        else:
            n = left_normal_world(t)
        p = p + abs(lat_m) * n

    return {"x": float(p[0]), "y": float(p[1]), "yaw_deg": float(yaw)}

def build_motion_waypoints(
    motion: Dict[str, Any],
    category: str,
    anchor_spawn: Dict[str, float],
    seg_points: np.ndarray,
) -> List[Dict[str, Any]]:
    mtype = motion.get("type", "static")
    speed_profile = motion.get("speed_profile", "normal")
    is_erratic = speed_profile == "erratic"
    speed = infer_speed_mps(category, speed_profile)

    if mtype == "static":
        return [{
            "x": anchor_spawn["x"],
            "y": anchor_spawn["y"],
            "yaw_deg": anchor_spawn["yaw_deg"],
            "speed_mps": 0.0,
        }]

    if mtype == "follow_lane":
        # Create a short forward trajectory along the segment
        s0 = float(motion.get("start_s_along", None) or 0.0)
        if s0 <= 0.0:
            s0 = None
        # start at anchor, then go forward by delta_s
        delta_m = float(motion.get("travel_distance_m", 18.0))
        # Convert delta_m to delta_s by arc length
        cum = cumulative_dist(seg_points)
        total = float(cum[-1]) if len(cum) else 0.0
        if total < 1e-6:
            return [{
                "x": anchor_spawn["x"],
                "y": anchor_spawn["y"],
                "yaw_deg": anchor_spawn["yaw_deg"],
                "speed_mps": speed,
            }]

        # anchor at s_anchor, then advance by delta_m
        # Determine current arc distance for anchor
        p_anchor, _ = point_and_tangent_at_s(seg_points, float(motion.get("anchor_s_along", 0.5)))
        # Find closest point index for approximate s distance
        dists = np.linalg.norm(seg_points - np.array([anchor_spawn["x"], anchor_spawn["y"]])[None, :], axis=1)
        idx = int(np.argmin(dists))
        s_anchor_dist = float(cum[idx])
        s_end_dist = min(total, s_anchor_dist + delta_m)

        # Sample along [s_anchor_dist, s_end_dist]
        num = int(motion.get("num_waypoints", 8))
        num = max(2, min(40, num))
        waypoints = []
        
        if is_erratic:
            # Erratic driving: variable speed, lane weaving, unpredictable heading
            import random
            random.seed(hash(str(anchor_spawn)))  # Deterministic but varied
            
            for k in range(num):
                target = s_anchor_dist + (s_end_dist - s_anchor_dist) * (k / (num - 1))
                s_frac = target / total if total > 1e-6 else 0.0
                spawn = compute_spawn_from_anchor(seg_points, s_frac, "center")
                
                # Erratic speed: varies between 50% and 150% of base speed
                erratic_speed = speed * (0.5 + random.random())
                
                # Erratic lateral offset: weave side to side (up to 1.5m)
                lat_offset = (random.random() - 0.5) * 3.0  # -1.5m to +1.5m
                _, t = point_and_tangent_at_s(seg_points, s_frac)
                n = right_normal_world(t) if lat_offset >= 0 else left_normal_world(t)
                spawn["x"] += abs(lat_offset) * n[0]
                spawn["y"] += abs(lat_offset) * n[1]
                
                # Erratic heading: slight random deviation (up to ±10 degrees)
                spawn["yaw_deg"] = wrap180(spawn["yaw_deg"] + (random.random() - 0.5) * 20)
                
                waypoints.append({**spawn, "speed_mps": erratic_speed, "erratic": True})
        else:
            for k in range(num):
                target = s_anchor_dist + (s_end_dist - s_anchor_dist) * (k / (num - 1))
                s_frac = target / total if total > 1e-6 else 0.0
                spawn = compute_spawn_from_anchor(seg_points, s_frac, "center")
                waypoints.append({**spawn, "speed_mps": speed})
        return waypoints

    if mtype == "cross_perpendicular":
        # For crossing motion, the pedestrian should:
        # 1. Start from the sidewalk (off-road), not just lane edge
        # 2. Cross the entire road width to the opposite sidewalk
        
        # Determine normal from segment tangent at anchor
        _, t = point_and_tangent_at_s(seg_points, float(motion.get("anchor_s_along", 0.5)))
        
        # Cross direction: use explicit direction, or infer from start lateral
        side = str(motion.get("cross_direction", "unknown")).lower()
        if side not in ("left", "right"):
            start_lat = str(motion.get("start_lateral", "")).lower()
            if "right" in start_lat:
                side = "left"  # starting on right side, cross to left
            elif "left" in start_lat:
                side = "right"  # starting on left side, cross to right
            else:
                side = "left"  # default to crossing left
        
        # Calculate road crossing geometry
        # Assume a typical 2-lane road per direction = 4 lanes total ≈ 14m road width
        # Plus sidewalk offset on each side ≈ 2m each = 18m total crossing
        # For a simpler 2-lane road, use ~10m crossing
        # We'll use a default that spans from sidewalk to sidewalk across a typical road
        default_road_crossing_m = 12.0  # Enough to cross a 2-lane road from curb to curb
        dist = float(motion.get("cross_distance_m", default_road_crossing_m))
        
        # IMPORTANT: Start from OFF-ROAD position, not from the lane
        # The anchor_spawn is at the lane position; we need to offset to the sidewalk
        # Move the start point outward by offroad offset (about 1 lane width from lane center)
        offroad_offset_m = 1.1 * LANE_WIDTH_M  # ~3.85m from lane center to sidewalk
        
        # Get base point from lane center
        center_spawn = compute_spawn_from_anchor(seg_points, float(motion.get("anchor_s_along", 0.5)), "center")
        p_center = np.array([center_spawn["x"], center_spawn["y"]], dtype=float)
        
        # Determine start side based on cross direction
        # If crossing "left" (rightward to leftward), start from right side (offroad_right)
        # If crossing "right" (leftward to rightward), start from left side (offroad_left)
        if side == "left":
            # Start from right sidewalk, cross left
            start_normal = right_normal_world(t)
            cross_normal = left_normal_world(t)
        else:
            # Start from left sidewalk, cross right
            start_normal = left_normal_world(t)
            cross_normal = right_normal_world(t)
        
        # Calculate start point on the sidewalk (off-road)
        p0 = p_center + offroad_offset_m * start_normal
        
        # Calculate end point on the opposite sidewalk
        # Total crossing = 2 * offroad_offset + road width
        p1 = p0 + dist * cross_normal
        
        yaw = wrap180(heading_deg_from_vec(cross_normal))
        return [
            {"x": float(p0[0]), "y": float(p0[1]), "yaw_deg": float(yaw), "speed_mps": speed},
            {"x": float(p1[0]), "y": float(p1[1]), "yaw_deg": float(yaw), "speed_mps": speed},
        ]

    if mtype == "straight_line":
        end_anchor = motion.get("end_anchor", {})
        wp = end_anchor.get("world_point")
        if not isinstance(wp, dict) or "x" not in wp or "y" not in wp:
            # If no end point, just hold
            return [{"x": anchor_spawn["x"], "y": anchor_spawn["y"], "yaw_deg": anchor_spawn["yaw_deg"], "speed_mps": speed}]
        p0 = np.array([anchor_spawn["x"], anchor_spawn["y"]], dtype=float)
        p1 = np.array([float(wp["x"]), float(wp["y"])], dtype=float)
        v = p1 - p0
        n = float(np.linalg.norm(v))
        yaw = wrap180(heading_deg_from_vec(v / n)) if n > 1e-9 else anchor_spawn["yaw_deg"]
        return [
            {"x": float(p0[0]), "y": float(p0[1]), "yaw_deg": float(yaw), "speed_mps": speed},
            {"x": float(p1[0]), "y": float(p1[1]), "yaw_deg": float(yaw), "speed_mps": speed},
        ]

    # fallback
    return [{"x": anchor_spawn["x"], "y": anchor_spawn["y"], "yaw_deg": anchor_spawn["yaw_deg"], "speed_mps": speed}]


# ======================================================================================
# Visualization
# ======================================================================================

def visualize(
    picked: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    actors_world: List[Dict[str, Any]],
    crop_region: Optional[Dict[str, Any]],
    out_path: str,
    description: Optional[str] = None,
    show: bool = False,
) -> None:
    if plt is None:
        print("[WARNING] matplotlib not installed; skipping visualization")
        return

    import matplotlib.patches as mpatches
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # -------------------------
    # Crop region / axes config
    # -------------------------
    if crop_region and all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
        xmin, xmax, ymin, ymax = crop_region["xmin"], crop_region["xmax"], crop_region["ymin"], crop_region["ymax"]
        max_range = max(float(xmax - xmin), float(ymax - ymin))
        margin_m = min(max(12.0, 0.12 * max_range), 60.0)
        ax.set_xlim(xmin - margin_m, xmax + margin_m)
        ax.set_ylim(ymin - margin_m, ymax + margin_m)
        ax.invert_xaxis()
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linestyle="--", linewidth=2)
        ax.add_patch(rect)

    cmap = plt.cm.get_cmap("tab10")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _offset_polyline(poly: np.ndarray, offset_m: float) -> np.ndarray:
        """Laterally offset a polyline for visual separation."""
        if abs(offset_m) < 1e-6 or poly is None or len(poly) < 2:
            return poly
        pts = np.asarray(poly, dtype=float)
        out = []
        n = len(pts)
        for k in range(n):
            if k == 0:
                t = pts[1] - pts[0]
            elif k == n - 1:
                t = pts[k] - pts[k - 1]
            else:
                t = pts[k + 1] - pts[k - 1]
            if offset_m >= 0:
                nvec = right_normal_world(t)
            else:
                nvec = left_normal_world(t)
            out.append(pts[k] + abs(offset_m) * nvec)
        return np.vstack(out)

    def _bbox_intersects(b1: Bbox, b2: Bbox) -> bool:
        return not (b1.x1 < b2.x0 or b1.x0 > b2.x1 or b1.y1 < b2.y0 or b1.y0 > b2.y1)

    def _inflate_bbox(bb: Bbox, px: float = 2.0) -> Bbox:
        return Bbox.from_extents(bb.x0 - px, bb.y0 - px, bb.x1 + px, bb.y1 + px)

    def _actor_marker(cat: str) -> str:
        cat = (cat or "").lower()
        if cat == "walker":
            return "P"
        if cat == "cyclist":
            return "D"
        if cat == "vehicle":
            return "s"
        return "x"

    def _get_asset_short(asset_id: str, fallback: str) -> str:
        asset_id = str(asset_id or "")
        if not asset_id:
            return fallback
        parts = asset_id.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else asset_id

    def _actor_bbox_dims_from_actor_or_cache(a: Dict[str, Any]) -> Tuple[float, float]:
        """
        Returns (length,width) in meters, or (0,0) if unknown.
        Prefers a['bbox'], else falls back to get_asset_bbox(asset_id).
        """
        asset_id = str(a.get("asset_id", ""))
        actor_bbox = a.get("bbox")
        if isinstance(actor_bbox, dict):
            try:
                length = float(actor_bbox.get("length", 0.0))
                width = float(actor_bbox.get("width", 0.0))
                if length > 0.05 and width > 0.05:
                    return length, width
            except Exception:
                pass

        asset_bbox = get_asset_bbox(asset_id)
        if asset_bbox:
            return float(asset_bbox.length), float(asset_bbox.width)
        return 0.0, 0.0

    def _draw_filled_oriented_bbox(x: float, y: float, yaw_deg: float, length: float, width: float,
                                   facecolor: Any, edgecolor: Any, alpha: float, zorder: int):
        # local corners centered at origin
        half_l, half_w = length / 2.0, width / 2.0
        corners_local = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w],
        ], dtype=float)
        yaw_rad = math.radians(float(yaw_deg))
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
        corners_world = corners_local @ rot.T + np.array([x, y], dtype=float)

        poly = mpatches.Polygon(
            corners_world,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.3,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(poly)
        return poly

    def _collect_forbidden_bboxes(fig, ax, artists, pad_px: float = 3.0) -> List[Bbox]:
        """
        Collect screen-space bboxes for "things labels should not overlap":
        - ego path lines
        - motion lines
        - bbox patches
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        forb: List[Bbox] = []
        for art in artists:
            try:
                bb = art.get_window_extent(renderer=renderer)
                if bb is not None:
                    forb.append(_inflate_bbox(bb, px=pad_px))
            except Exception:
                continue
        return forb

    def _place_labels_repel(ax, fig, items, forbidden_bboxes, fontsize=8, crop_region: Optional[Dict[str, Any]] = None):
        """
        Greedy label placement that avoids:
          - other labels
          - forbidden_bboxes (paths, motion polylines, bbox patches)
        Adds leader lines.
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        placed_bboxes: List[Bbox] = []
        used_annotations = []

        # Offset candidates: try lots, progressively farther away.
        offsets = []
        for r in (10, 16, 24, 34, 46, 60, 78, 98):
            offsets += [( r, 0), (-r, 0), (0, r), (0, -r)]
            offsets += [( r, r), ( r, -r), (-r, r), (-r, -r)]
            offsets += [( int(r*0.9), int(r*0.4)), ( int(r*0.9), -int(r*0.4)),
                        (-int(r*0.9), int(r*0.4)), (-int(r*0.9), -int(r*0.4))]
            offsets += [( int(r*0.4), int(r*0.9)), (-int(r*0.4), int(r*0.9)),
                        ( int(r*0.4), -int(r*0.9)), (-int(r*0.4), -int(r*0.9))]

        def ok_bbox(bb: Bbox) -> bool:
            for prev in placed_bboxes:
                if _bbox_intersects(bb, prev):
                    return False
            for fb in forbidden_bboxes:
                if _bbox_intersects(bb, fb):
                    return False
            return True

        def _whitespace_candidates(x: float, y: float) -> List[Tuple[float, float]]:
            if not crop_region or not all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
                return []

            xmin, xmax = float(crop_region["xmin"]), float(crop_region["xmax"])
            ymin, ymax = float(crop_region["ymin"]), float(crop_region["ymax"])
            ax_xmin, ax_xmax = sorted(ax.get_xlim())
            ax_ymin, ax_ymax = sorted(ax.get_ylim())

            max_range = max(xmax - xmin, ymax - ymin)
            pad_m = max(1.5, 0.02 * max_range)

            margins = {
                "left": xmin - ax_xmin,
                "right": ax_xmax - xmax,
                "bottom": ymin - ax_ymin,
                "top": ax_ymax - ymax,
            }
            side_order = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)
            rail_offsets = [0.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0]

            positions: List[Tuple[float, float]] = []
            for side, margin in side_order:
                if margin <= pad_m * 1.2:
                    continue
                if side in ("left", "right"):
                    base_x = xmin - pad_m if side == "left" else xmax + pad_m
                    base_x = min(max(base_x, ax_xmin + pad_m), ax_xmax - pad_m)
                    for dy in rail_offsets:
                        y_text = float(np.clip(y + dy, ax_ymin + pad_m, ax_ymax - pad_m))
                        positions.append((base_x, y_text))
                else:
                    base_y = ymin - pad_m if side == "bottom" else ymax + pad_m
                    base_y = min(max(base_y, ax_ymin + pad_m), ax_ymax - pad_m)
                    for dx in rail_offsets:
                        x_text = float(np.clip(x + dx, ax_xmin + pad_m, ax_xmax - pad_m))
                        positions.append((x_text, base_y))

            return positions

        for it in items:
            x, y = float(it["x"]), float(it["y"])
            label = str(it["label"])
            color = it.get("color", "black")
            zorder = int(it.get("zorder", 9))

            placed = False
            whitespace_positions = _whitespace_candidates(x, y)
            for (tx, ty) in whitespace_positions:
                ann = ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(tx, ty),
                    textcoords="data",
                    fontsize=fontsize,
                    color=color,
                    zorder=zorder,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                    annotation_clip=False,
                )
                bb = ann.get_window_extent(renderer=renderer)
                bb = _inflate_bbox(bb, px=2.0)
                if ok_bbox(bb):
                    placed_bboxes.append(bb)
                    used_annotations.append(ann)
                    placed = True
                    break
                ann.remove()

            if not placed:
                for (dx, dy) in offsets:
                    ann = ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=fontsize,
                        color=color,
                        zorder=zorder,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                    )
                    bb = ann.get_window_extent(renderer=renderer)
                    bb = _inflate_bbox(bb, px=2.0)
                    if ok_bbox(bb):
                        placed_bboxes.append(bb)
                        used_annotations.append(ann)
                        placed = True
                        break
                    ann.remove()

            if not placed:
                # Absolute last resort: still draw it (but at least boxed)
                ann = ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(offsets[-1][0], offsets[-1][1]),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color=color,
                    zorder=zorder,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.55),
                )
                used_annotations.append(ann)

        return used_annotations

    # ------------------------------------------------------------
    # Background lane geometry (de-emphasized)
    # ------------------------------------------------------------
    all_pts = []
    for pts in seg_by_id.values():
        if pts is not None and len(pts):
            all_pts.append(np.asarray(pts, dtype=float))
    if all_pts:
        pts_concat = np.vstack(all_pts)
        ax.scatter(
            pts_concat[:, 0],
            pts_concat[:, 1],
            s=4,
            color="lightgray",
            alpha=0.25,
            zorder=0,
            label=None
        )

    # ------------------------------------------------------------
    # Ego paths (offset a bit to avoid perfect overlap)
    # ------------------------------------------------------------
    ego_line_artists = []
    marker_artists = []
    n_paths = max(1, len(picked))
    offset_step = 0.6  # meters
    center = (n_paths - 1) / 2.0

    for i, p in enumerate(picked):
        veh = p.get("vehicle", f"Vehicle {i+1}")
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        color = cmap(i % 10)
        offset_m = (i - center) * offset_step

        all_segment_pts = []
        segments_detailed = sig.get("segments_detailed", [])
        if isinstance(segments_detailed, list) and segments_detailed:
            for seg in segments_detailed:
                if not isinstance(seg, dict):
                    continue
                poly = seg.get("polyline_sample", [])
                if isinstance(poly, list) and poly:
                    for pt in poly:
                        if isinstance(pt, dict) and "x" in pt and "y" in pt:
                            all_segment_pts.append(np.array([float(pt["x"]), float(pt["y"])], dtype=float))

        if not all_segment_pts:
            seg_ids = sig.get("segment_ids", [])
            if isinstance(seg_ids, list):
                for sid in seg_ids:
                    try:
                        sid_i = int(sid)
                    except Exception:
                        continue
                    pts = seg_by_id.get(sid_i)
                    if pts is not None and len(pts) > 0:
                        pts = np.asarray(pts, dtype=float)
                        for pt in pts:
                            all_segment_pts.append(np.asarray(pt, dtype=float))

        if not all_segment_pts:
            continue

        pts = np.vstack(all_segment_pts)
        pts_off = _offset_polyline(pts, offset_m)

        ln, = ax.plot(
            pts_off[:, 0],
            pts_off[:, 1],
            linewidth=3.0,
            alpha=0.82,
            color=color,
            label=veh,
            zorder=2,
        )
        ego_line_artists.append(ln)

        # Direction arrow
        if len(pts_off) >= 2:
            arr = ax.annotate(
                "",
                xy=(pts_off[-1, 0], pts_off[-1, 1]),
                xytext=(pts_off[-2, 0], pts_off[-2, 1]),
                arrowprops=dict(arrowstyle="->", lw=3, color=color, alpha=0.95, mutation_scale=18),
                zorder=3,
            )

        # Start/end markers (first/last within crop if crop exists)
        if crop_region and all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
            xmin, xmax = crop_region["xmin"], crop_region["xmax"]
            ymin, ymax = crop_region["ymin"], crop_region["ymax"]
            in_crop = [(idx, pt) for idx, pt in enumerate(pts) if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax]
            if in_crop:
                first_idx = in_crop[0][0]
                last_idx = in_crop[-1][0]
                if 0 <= first_idx < len(pts_off) and 0 <= last_idx < len(pts_off):
                    first_pt = pts_off[first_idx]
                    last_pt = pts_off[last_idx]
                    sc = ax.scatter(
                        [first_pt[0], last_pt[0]],
                        [first_pt[1], last_pt[1]],
                        s=90,
                        facecolors=color,
                        edgecolors="white",
                        linewidths=1.5,
                        alpha=0.95,
                        zorder=6,
                    )
                    marker_artists.append(sc)

    legend = None
    if len(picked) > 0:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, color="gray"))
        labels.append("Start/End")
        legend = ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------
    # Actors: draw motion first, then bboxes/markers, then labels last
    # ------------------------------------------------------------
    motion_line_artists = []
    bbox_patch_artists = []
    # marker_artists already contains start/end markers

    # Cluster labels by approximate position + asset_short to avoid stacked cone labels.
    BUCKET_M = 1.2
    clusters: Dict[Tuple[int, int, str], List[int]] = {}

    actor_pts = []
    for j, a in enumerate(actors_world):
        spawn = a.get("spawn", {})
        x, y = spawn.get("x"), spawn.get("y")
        if x is None or y is None:
            continue

        asset_short = _get_asset_short(str(a.get("asset_id", "")), str(a.get("category", "object")))
        bx = int(round(float(x) / BUCKET_M))
        by = int(round(float(y) / BUCKET_M))
        key = (bx, by, asset_short)
        clusters.setdefault(key, []).append(j)
        actor_pts.append((float(x), float(y), a))

    # 1) Draw motion polylines (so labels avoid them)
    for (x, y, a) in actor_pts:
        wps = a.get("world_waypoints", [])
        if isinstance(wps, list) and len(wps) >= 2:
            xs = [w["x"] for w in wps if "x" in w and "y" in w]
            ys = [w["y"] for w in wps if "x" in w and "y" in w]
            if len(xs) >= 2:
                is_erratic = (a.get("motion", {}) or {}).get("speed_profile") == "erratic"
                if is_erratic:
                    ln, = ax.plot(xs, ys, linestyle="-", linewidth=2.5, alpha=0.80, zorder=4)
                    motion_line_artists.append(ln)
                    sc = ax.scatter(xs, ys, s=18, marker="o", alpha=0.75, zorder=5)
                    marker_artists.append(sc)
                else:
                    ln, = ax.plot(xs, ys, linestyle=":", linewidth=1.6, alpha=0.75, zorder=4)
                    motion_line_artists.append(ln)

                arr = ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="->", lw=1.4, alpha=0.8),
                    zorder=5,
                )

    # 2) Draw bboxes if available; otherwise draw a marker
    for (x, y, a) in actor_pts:
        spawn = a.get("spawn", {}) or {}
        yaw_deg = float(spawn.get("yaw_deg", 0.0))

        cat = str(a.get("category", "static")).lower()
        length, width = _actor_bbox_dims_from_actor_or_cache(a)

        # If bbox exists: draw ONLY bbox (filled), no "x"/"square" marker
        if length > 0.10 and width > 0.10:
            # Use category-based color, but keep it fairly subtle.
            if cat == "vehicle":
                face = "tab:blue"
                edge = "tab:blue"
            elif cat == "walker":
                face = "tab:green"
                edge = "tab:green"
            elif cat == "cyclist":
                face = "tab:orange"
                edge = "tab:orange"
            else:
                face = "tab:gray"
                edge = "tab:gray"

            poly = _draw_filled_oriented_bbox(
                x=x, y=y, yaw_deg=yaw_deg,
                length=length, width=width,
                facecolor=face, edgecolor=edge,
                alpha=0.25, zorder=6
            )
            bbox_patch_artists.append(poly)
        else:
            m = _actor_marker(cat)
            sc = ax.scatter([x], [y], s=55, marker=m, zorder=7)
            marker_artists.append(sc)

    # ------------------------------------------------------------
    # Labels: one label per cluster, repel away from paths/motion/bboxes
    # ------------------------------------------------------------
    # Compute forbidden regions for labels (screen-space bboxes)
    forbidden_artists = ego_line_artists + motion_line_artists + bbox_patch_artists + marker_artists
    if legend is not None:
        forbidden_artists.append(legend)
    forbidden = _collect_forbidden_bboxes(fig, ax, forbidden_artists, pad_px=5.0)

    label_items = []
    for (bx, by, asset_short), idxs in clusters.items():
        # centroid anchor (better than "first actor")
        xs = []
        ys = []
        for j in idxs:
            sp = actors_world[j].get("spawn", {}) or {}
            if "x" in sp and "y" in sp:
                xs.append(float(sp["x"]))
                ys.append(float(sp["y"]))
        if not xs:
            continue
        x0 = float(sum(xs) / len(xs))
        y0 = float(sum(ys) / len(ys))

        count = len(idxs)
        a0 = actors_world[idxs[0]]
        if count > 1:
            label = f"{asset_short} ×{count}"
        else:
            actor_id = a0.get("id", "obj")
            label = f"{actor_id}: {asset_short}"

        label_items.append({"x": x0, "y": y0, "label": label, "zorder": 10})

    _place_labels_repel(ax, fig, label_items, forbidden_bboxes=forbidden, fontsize=8, crop_region=crop_region)

    # ------------------------------------------------------------
    # Title
    # ------------------------------------------------------------
    lines = []
    if description:
        desc_clean = " ".join(str(description).split())
        scene_text = textwrap.fill(desc_clean, width=90)
        lines.append(r"$\bf{Scene:}$ " + scene_text)
    lines.append(rf"$\bf{{Placed\ actors\ (n={len(actors_world)})}}$")
    ax.set_title("\n".join(lines), fontsize=12, loc="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Visualization saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ======================================================================================
# Main pipeline
# ======================================================================================

# ======================================================================================
# CSP-style placement (LLM emits symbolic preferences + relations; solver picks anchors)
# ======================================================================================

DIST_BUCKET_TO_M = {
    "touching": 1.0,
    "close": 4.0,
    "medium": 10.0,
    "far": 20.0,
    "unknown": None,
}
ALLOWED_PHASES = {"on_approach", "in_intersection", "after_turn", "after_exit", "after_merge", "any", "unknown"}
ALLOWED_REL_TYPES = {"ahead_of", "behind_of", "left_of", "right_of", "near"}
ALLOWED_DIST_BUCKETS = set(DIST_BUCKET_TO_M.keys())
ALLOWED_GROUP_PATTERNS = {"across_lane", "along_lane", "scatter", "unknown"}

AFTER_MERGE_CLEARANCE_M = 2.0
MERGE_POINT_MAX_DIST_M = 6.0

def _parse_vehicle_num(v: Any) -> Optional[int]:
    if v is None:
        return None
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None

def _compute_merge_min_s_by_vehicle(
    picked_payload: Dict[str, Any],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    max_dist_m: float = MERGE_POINT_MAX_DIST_M,
) -> Dict[int, float]:
    """
    Map target vehicle -> path_s_m of merge point (from path refiner), when available.
    Uses the closest merge point along the target's path and ignores far/off-path points.
    """
    refinement = picked_payload.get("refinement", {})
    if not isinstance(refinement, dict):
        return {}
    lane_change_debug = refinement.get("lane_change_debug", [])
    if not isinstance(lane_change_debug, list):
        return {}

    points_by_target: Dict[int, List[Tuple[float, float]]] = {}
    for item in lane_change_debug:
        if not isinstance(item, dict):
            continue
        dbg = item.get("debug") or {}
        if not isinstance(dbg, dict) or not dbg.get("applied"):
            continue
        mp = dbg.get("merge_point") or {}
        if not isinstance(mp, dict):
            continue
        try:
            x = float(mp.get("x"))
            y = float(mp.get("y"))
        except Exception:
            continue
        lane_change = item.get("lane_change") or {}
        target = lane_change.get("target") or lane_change.get("target_vehicle")
        target_num = _parse_vehicle_num(target)
        if target_num is None:
            continue
        points_by_target.setdefault(target_num, []).append((x, y))

    if not points_by_target:
        return {}

    picked_by_vehicle = {}
    for p in picked_list:
        vnum = _parse_vehicle_num(p.get("vehicle"))
        if vnum is not None:
            picked_by_vehicle[vnum] = p

    out: Dict[int, float] = {}
    for veh_num, pts in points_by_target.items():
        picked_entry = picked_by_vehicle.get(veh_num)
        if not picked_entry:
            continue
        best_s = None
        for x, y in pts:
            proj = _project_point_to_path_s_m(picked_entry, seg_by_id, np.array([x, y], dtype=float))
            if not proj:
                continue
            dist, path_s = proj
            if dist > max_dist_m:
                continue
            if best_s is None or path_s < best_s:
                best_s = path_s
        if best_s is not None:
            out[veh_num] = best_s
    return out

def _inside_crop_xy(x: float, y: float, crop_region: Any, margin_m: float = 0.0) -> bool:
    if not isinstance(crop_region, dict):
        return True
    try:
        xmin = float(crop_region.get("xmin"))
        xmax = float(crop_region.get("xmax"))
        ymin = float(crop_region.get("ymin"))
        ymax = float(crop_region.get("ymax"))
    except Exception:
        return True
    return (xmin + margin_m <= x <= xmax - margin_m) and (ymin + margin_m <= y <= ymax - margin_m)

def _actor_radius_m(category: str, actor_kind: str, asset_id: Optional[str] = None) -> float:
    """
    Return the collision radius for an actor. This is used to prevent
    actors from spawning on top of each other.
    
    If asset_id is provided and has a bounding box, use that for accurate sizing.
    Otherwise fall back to category-based defaults.
    """
    # Try to use bbox if available
    if asset_id:
        bbox = get_asset_bbox(asset_id)
        if bbox:
            # Use the larger of length/width divided by 2 as radius
            # (conservative: use max dimension for circular collision)
            return max(bbox.length, bbox.width) / 2.0
    
    # Fallback to category-based defaults
    c = str(category).lower()
    k = str(actor_kind).lower()
    if c == "vehicle" or k in ("parked_vehicle", "npc_vehicle"):
        return 2.5  # Vehicles need more space
    if c in ("walker", "cyclist") or k in ("walker", "cyclist"):
        return 0.8  # Walkers/cyclists
    # Static props like cones, barriers - still need some separation
    return 0.6  # Increased from 0.5 to avoid overlaps

def get_actor_bbox_dims(asset_id: Optional[str], category: str, actor_kind: str) -> Tuple[float, float]:
    """
    Get length and width for an actor (in meters).
    Returns (length, width) tuple. Uses bbox if available, else defaults.
    """
    if asset_id:
        bbox = get_asset_bbox(asset_id)
        if bbox:
            return (bbox.length, bbox.width)
    
    # Fallback to category-based defaults
    c = str(category).lower()
    k = str(actor_kind).lower()
    if c == "vehicle" or k in ("parked_vehicle", "npc_vehicle"):
        return (4.5, 1.8)  # Typical car
    if c == "walker" or k == "walker":
        return (0.6, 0.6)  # Pedestrian
    if c == "cyclist" or k == "cyclist":
        return (1.8, 0.6)  # Bicycle
    return (0.6, 0.6)  # Small static prop

# Minimum separation distance between any two actors (safety margin)
MIN_ACTOR_SEPARATION_M = 0.5

def _find_opposite_lane_segments(
    ref_vehicle_num: int,
    picked_list: List[Dict[str, Any]],
    all_segments: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    heading_diff_threshold: float = 135.0,
) -> List[Dict[str, Any]]:
    """
    Find segments that go in the opposite direction to a reference vehicle's path.
    Returns list of segment dicts with seg_id, points, and heading.
    """
    # Get reference vehicle's heading from first segment
    ref_entry = next((p for p in picked_list if _parse_vehicle_num(p.get("vehicle")) == ref_vehicle_num), None)
    if not ref_entry:
        return []
    
    sig = ref_entry.get("signature", {})
    entry = sig.get("entry", {})
    ref_heading = entry.get("heading_deg")
    if ref_heading is None:
        return []
    ref_heading = float(ref_heading)
    
    # Find segments with approximately opposite heading
    opposite_segs = []
    for seg in all_segments:
        seg_id = seg.get("seg_id")
        pts = seg_by_id.get(seg_id)
        if pts is None or len(pts) < 2:
            continue
        
        # Compute segment heading from first/last points
        seg_heading = heading_deg_from_vec(pts[-1] - pts[0])
        heading_diff = abs(wrap180(seg_heading - ref_heading))
        
        if heading_diff >= heading_diff_threshold:
            # Check if segment is in crop region
            mid_pt = pts[len(pts)//2]
            if _inside_crop_xy(float(mid_pt[0]), float(mid_pt[1]), crop_region, margin_m=2.0):
                opposite_segs.append({
                    "seg_id": seg_id,
                    "points": pts,
                    "heading_deg": seg_heading,
                    "length_m": float(cumulative_dist(pts)[-1]),
                })
    
    return opposite_segs

@dataclass
class CandidatePlacement:
    vehicle_num: int
    segment_index: int   # 1-based within vehicle path
    seg_id: int
    s_along: float
    lateral_relation: str
    x: float
    y: float
    yaw_deg: float
    path_s_m: float      # cumulative distance along vehicle path
    base_score: float

def build_stage2_constraints_prompt(
    description: str,
    vehicle_segments: Dict[str, Any],
    stage1_entities: List[Dict[str, Any]],
    per_entity_options: Dict[str, List[Dict[str, Any]]],
) -> str:
    payload = {
        "task": "Emit actor_specs with symbolic preferences/relations only. Solver picks concrete anchors later.",
        "description": description,
        "ego_paths": vehicle_segments,
        "stage1_entities": stage1_entities,
        "asset_candidates_by_entity_id": per_entity_options,
        "allowed_enums": {
            "phase": sorted(list(ALLOWED_PHASES)),
            "lateral_relation": sorted(list(LATERAL_TO_M.keys())),
            "relation_type": sorted(list(ALLOWED_REL_TYPES)),
            "distance_bucket": sorted(list(ALLOWED_DIST_BUCKETS)),
            "group_pattern": sorted(list(ALLOWED_GROUP_PATTERNS)),
            "category": ["static", "vehicle", "walker", "cyclist"],
        },
        "motion_type_definitions": {
            "static": "Object does not move (parked vehicles, cones, barriers, debris).",
            "follow_lane": "Moves ALONG the road in the lane direction (same direction as traffic). Use when description says 'walks in the direction of the road', 'walks along the road', 'travels in the lane direction', etc.",
            "cross_perpendicular": "Moves ACROSS the road (perpendicular to traffic). Use when description says 'crosses the road', 'crosses the street', 'jaywalking', etc.",
            "straight_line": "Moves in a straight line (not necessarily along or across road).",
            "zigzag_lane": "Weaves between lanes.",
            "unknown": "Motion not specified or unclear."
        },
        "motion_hint_to_type_mapping": {
            "static": "static",
            "crossing": "cross_perpendicular",
            "follow_lane": "follow_lane",
            "unknown": "unknown"
        },
        "output_schema": {
            "actor_specs": [
                {
                    "id": "entity_1",
                    "asset_id": "MUST be one of the asset_id strings provided for that entity_id",
                    "category": "static|vehicle|walker|cyclist",
                    "actor_kind": "copy from Stage1 actor_kind",
                    "quantity": "IGNORED (quantity is taken from Stage1). You may omit or set to 1.",
                    "anchor": {
                        "target_vehicle": "Vehicle N | none | unknown",
                        "phase": "on_approach|in_intersection|after_turn|after_exit|after_merge|any|unknown",
                        "lateral_preference": "one of lateral_relation enums or unknown"
                    },
                    "relations": [
                        {
                            "type": "ahead_of|behind_of|left_of|right_of|near",
                            "other_id": "entity_*",
                            "distance": "touching|close|medium|far|unknown",
                            "evidence": "EXACT quote (<=20 words) from description"
                        }
                    ],
                    "group_pattern": {
                        "pattern": "across_lane|along_lane|scatter|unknown",
                        "start_lateral": "optional lateral enum",
                        "end_lateral": "optional lateral enum",
                        "spacing_bucket": "tight|normal|sparse|auto"
                    },
                    "motion": {
                        "type": "DERIVE from stage1 motion_hint using motion_hint_to_type_mapping. 'follow_lane' means along road, 'crossing' means cross_perpendicular.",
                        "speed_profile": "slow|normal|fast|erratic|unknown",
                        "cross_direction": "left|right|unknown (ONLY for cross_perpendicular: direction the actor moves. 'from right to left' = 'left', 'from left to right' = 'right')"
                    },
                    "confidence": "float in [0,1]"
                }
            ]
        },
        "critical_rules": [
            "Return JSON ONLY. No prose, no markdown.",
            "CLOSED WORLD: use ONLY enums provided.",
            "Do NOT output segment_index, seg_id, s_along, x/y, yaw.",
            "Do NOT invent constraints. If unsure, use 'unknown' or omit.",
            "If a Stage1 entity seems like map geometry (e.g., intersection/road/lane/sidewalk) or lacks a clear actor noun, OMIT it from actor_specs (do not force a placement).",
            
            "SEQUENTIAL/DISTANCE RELATIONS ARE CRITICAL:",
            "  - If description says 'X meters later', 'X meters after', 'X meters down', use relation type='ahead_of' with appropriate distance bucket.",
            "  - Distance mapping: 1-3m='touching', 4-8m='close', 9-15m='medium', 16-30m='far'.",
            "  - If description says 'farther down', 'further along', 'past the X', use type='ahead_of' with distance='far'.",
            "  - The entity mentioned FIRST in the description is typically BEHIND the one mentioned later (so later entity is 'ahead_of' the earlier one).",
            
            "Every relation MUST have non-empty evidence quote. Otherwise omit.",
            "Keep constraints minimal (0-2 relations per actor), but DO extract explicitly stated distance/sequence relationships.",
            "MOTION TYPE CRITICAL: Use motion_hint from stage1_entities to set motion.type:",
            "  - If motion_hint='follow_lane' -> motion.type='follow_lane' (moves ALONG the road)",
            "  - If motion_hint='crossing' -> motion.type='cross_perpendicular' (moves ACROSS the road)",
            "  - If motion_hint='static' -> motion.type='static'",
            "  - 'walks in the direction of the road' means ALONG the road = follow_lane, NOT crossing!",
            "SPEED PROFILE: Use speed_hint from stage1_entities. If speed_hint='erratic', use speed_profile='erratic'."
        ]
    }
    return (
        "You are a constraint extractor for a driving-scene CSP.\n"
        "Your output is used by a solver; invented/over-specific constraints hurt feasibility.\n"
        "Follow payload rules EXACTLY.\n\n"
        "PAYLOAD (JSON):\n" + json.dumps(payload, indent=2)
    )

def _normalize_actor_spec_id(spec: Dict[str, Any]) -> Optional[str]:
    if isinstance(spec, dict):
        if spec.get("id"):
            return str(spec.get("id"))
        if spec.get("entity_id"):
            return str(spec.get("entity_id"))
    return None

def _get_vehicle_lane_id(picked: List[Dict[str, Any]], vehicle_name: str, phase: str) -> Optional[int]:
    """
    Get the lane_id for a vehicle at the relevant phase.
    For 'after_exit', 'after_merge', 'after_turn' -> use last segment.
    For 'on_approach' -> use first segment.
    For others -> use last segment (where conflicts typically happen).
    """
    for p in picked:
        if str(p.get("vehicle", "")) == vehicle_name:
            sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
            segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed"), list) else []
            if not segs:
                return None
            # For approach phases, use first segment; otherwise use last
            if phase in ("on_approach",):
                return segs[0].get("lane_id")
            else:
                return segs[-1].get("lane_id")
    return None

def validate_actor_specs(
    actor_specs: Any,
    stage1_entities: List[Dict[str, Any]],
    per_entity_options: Dict[str, List[Dict[str, Any]]],
    picked: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(actor_specs, list):
        return [], ["actor_specs must be a list"]
    if picked is None:
        picked = []
    stage1_by_id = {str(e.get("entity_id")): e for e in stage1_entities if e.get("entity_id")}
    clean: List[Dict[str, Any]] = []
    for raw in actor_specs:
        if not isinstance(raw, dict):
            continue
        sid = _normalize_actor_spec_id(raw)
        if not sid or sid not in stage1_by_id:
            warnings.append(f"Dropped actor_spec with unknown id: {sid}")
            continue
        e = stage1_by_id[sid]
        kind = str(raw.get("actor_kind") or e.get("actor_kind") or "static_prop")
        cat = str(raw.get("category") or "").lower()
        if cat not in ("static", "vehicle", "walker", "cyclist"):
            if kind in ("walker",):
                cat = "walker"
            elif kind in ("cyclist",):
                cat = "cyclist"
            elif kind in ("parked_vehicle", "npc_vehicle"):
                cat = "vehicle"
            else:
                cat = "static"

        # asset_id strict: must be from candidate list
        asset_id = str(raw.get("asset_id") or "")
        allowed = [str(o.get("asset_id")) for o in per_entity_options.get(sid, []) if isinstance(o, dict) and o.get("asset_id")]
        if asset_id not in set(allowed):
            if allowed:
                asset_id = allowed[0]
                warnings.append(f"{sid}: asset_id not in candidates; fell back to {asset_id}")
            else:
                warnings.append(f"{sid}: no asset candidates; dropped spec")
                continue

        qty = e.get("quantity", e.get("count", 1))  # Fix B: quantity is authoritative from Stage 1
        try:
            qty = int(qty)
        except Exception:
            qty = 1
        qty = max(1, qty)

        anchor = raw.get("anchor", {}) if isinstance(raw.get("anchor", {}), dict) else {}
        tv = anchor.get("target_vehicle", e.get("affects_vehicle", "unknown"))
        phase = str(anchor.get("phase", e.get("when", "unknown")))
        # For lateral preference, try: anchor.lateral_preference -> e.lateral_relation -> e.start_lateral
        lat = str(anchor.get("lateral_preference", "unknown"))
        if lat == "unknown":
            lat = str(e.get("lateral_relation", "unknown"))
        if lat == "unknown":
            # Fall back to start_lateral (e.g., "pedestrian appears to the left" -> start_lateral=left_edge)
            lat = str(e.get("start_lateral") or "unknown")

        tvs = str(tv)
        if tvs not in ("none", "unknown") and _parse_vehicle_num(tvs) is None:
            tvs = "unknown"

        if phase not in ALLOWED_PHASES:
            phase = "unknown"
        if lat not in LATERAL_TO_M and lat != "unknown":
            lat = "unknown"

        # relations: drop invalid/missing evidence
        rels: List[Dict[str, Any]] = []
        rin = raw.get("relations", [])
        if isinstance(rin, list):
            for r in rin:
                if not isinstance(r, dict):
                    continue
                rtype = str(r.get("type", ""))
                oid = str(r.get("other_id", ""))
                dist = str(r.get("distance", "unknown"))
                ev = str(r.get("evidence", "")).strip()
                if rtype not in ALLOWED_REL_TYPES:
                    continue
                if oid not in stage1_by_id:
                    continue
                if dist not in ALLOWED_DIST_BUCKETS:
                    dist = "unknown"
                if not ev:
                    continue
                rels.append({"type": rtype, "other_id": oid, "distance": dist, "evidence": ev})

        # group pattern optional (only meaningful for qty>1)
        gp = raw.get("group_pattern")
        group_pattern = None
        if qty > 1:
            # First try Stage 2's group_pattern, then fall back to Stage 1 fields
            if isinstance(gp, dict):
                patt = str(gp.get("pattern", "unknown"))
                sl = gp.get("start_lateral")
                el = gp.get("end_lateral")
                spacing = str(gp.get("spacing_bucket", "auto"))
            else:
                # Use Stage 1 fields
                patt = str(e.get("group_pattern", "unknown"))
                sl = e.get("start_lateral")
                el = e.get("end_lateral")
                spacing = "auto"
            
            # Map "diagonal" to "along_lane" with start/end laterals
            if patt == "diagonal":
                patt = "along_lane"
                
            if patt not in ALLOWED_GROUP_PATTERNS:
                patt = "along_lane"  # Default for multiple objects
            if sl is not None and str(sl) not in LATERAL_TO_M:
                sl = None
            if el is not None and str(el) not in LATERAL_TO_M:
                el = None
            if spacing not in ("tight", "normal", "sparse", "auto"):
                spacing = "auto"
            group_pattern = {"pattern": patt, "start_lateral": sl, "end_lateral": el, "spacing_bucket": spacing}

        # Motion type: PRIORITIZE Stage 1 motion_hint over LLM's Stage 2 output
        # Map Stage 1 motion_hint to motion type
        MOTION_HINT_TO_TYPE = {
            "static": "static",
            "crossing": "cross_perpendicular",
            "follow_lane": "follow_lane",
            "unknown": "unknown",
        }
        
        motion = raw.get("motion", {}) if isinstance(raw.get("motion", {}), dict) else {}
        stage1_motion_hint = str(e.get("motion_hint", "unknown"))
        
        # Use Stage 1 motion_hint as the authoritative source (mapped to motion type)
        # Only fall back to LLM's Stage 2 output if Stage 1 hint is unknown
        if stage1_motion_hint in MOTION_HINT_TO_TYPE and stage1_motion_hint != "unknown":
            mtype = MOTION_HINT_TO_TYPE[stage1_motion_hint]
        else:
            mtype = str(motion.get("type", "static"))
            if mtype not in ("static", "follow_lane", "cross_perpendicular", "straight_line", "zigzag_lane", "unknown"):
                mtype = "unknown"
        
        # Default motion for NPC vehicles and cyclists: follow_lane (they're driving/riding)
        if mtype in ("unknown", "static") and kind in ("npc_vehicle", "cyclist"):
            mtype = "follow_lane"
                
        sp = str(motion.get("speed_profile", e.get("speed_hint", "unknown")))
        if sp not in ("slow", "normal", "fast", "erratic", "stopped", "unknown"):
            sp = "unknown"
        
        # Default speed for NPC vehicles: normal
        if sp == "unknown" and kind == "npc_vehicle":
            sp = "normal"

        conf = raw.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        conf = max(0.0, min(1.0, conf))
        
        # Propagate direction_relative_to from Stage 1 entity (for NPC vehicles going opposite direction)
        direction_relative_to = e.get("direction_relative_to")
        if direction_relative_to is not None and not isinstance(direction_relative_to, dict):
            direction_relative_to = None

        lat_pref = lat

        # Cross direction: FIRST check if we can infer from vehicle lane (most reliable for conflict),
        # then fallback to LLM outputs (Stage 2 motion, Stage 1 crossing_direction)
        cross_dir = "unknown"
        
        # For crossing motion, infer direction from target vehicle's lane to maximize conflict
        # This takes priority because it's based on actual geometry, not LLM guessing
        if mtype == "cross_perpendicular":
            lane_id = _get_vehicle_lane_id(picked, tvs, phase)
            if lane_id is not None:
                # In CARLA: lane_id < 0 = right side of road, lane_id > 0 = left side
                # For maximal conflict, pedestrian should start from the side the vehicle is on
                if lane_id < 0:
                    # Vehicle in right lane -> pedestrian starts from right, crosses left
                    cross_dir = "left"
                elif lane_id > 0:
                    # Vehicle in left lane -> pedestrian starts from left, crosses right
                    cross_dir = "right"
                # lane_id == 0 is rare (center lane); fall through to LLM fallback
        
        # Fallback to LLM-specified direction if lane-based inference didn't work
        if cross_dir == "unknown":
            cross_dir = str(motion.get("cross_direction", "unknown"))
        if cross_dir == "unknown":
            stage1_cross_dir = e.get("crossing_direction")
            if stage1_cross_dir in ("left", "right"):
                cross_dir = stage1_cross_dir

        # If crossing and no lateral preference, infer a start side from cross direction.
        if mtype == "cross_perpendicular" and lat_pref == "unknown":
            if cross_dir == "left":
                lat_pref = "right_edge"
            elif cross_dir == "right":
                lat_pref = "left_edge"

        clean.append({
            "id": sid,
            "semantic": str(raw.get("semantic") or e.get("mention") or sid),
            "category": cat,
            "actor_kind": kind,
            "asset_id": asset_id,
            "quantity": qty,
            "anchor": {"target_vehicle": tvs, "phase": phase, "lateral_preference": lat_pref},
            "relations": rels,
            "group_pattern": group_pattern,
            "motion": {
                "type": mtype,
                "speed_profile": sp,
                "cross_direction": cross_dir,
            },
            "confidence": conf,
            "direction_relative_to": direction_relative_to,
        })
    return clean, warnings

def _phase_score(segment_index: int, veh_info: Dict[str, Any], phase: str) -> float:
    """Soft-but-strong preference for placing an actor at the requested route phase.

    The earlier version used small +/- scores, which made it easy for the solver to
    place 'after_exit' objects near the approach if other terms dominated. Here we
    keep the solver soft (to avoid UNSAT), but make phase mismatches much more costly.
    """
    if phase in ("any", "unknown", None, ""):
        return 0.0

    turn_indices = veh_info.get("turn_indices", set()) if isinstance(veh_info, dict) else set()
    exit_idx = veh_info.get("exit_index", None) if isinstance(veh_info, dict) else None
    nsegs = len(veh_info.get("seg_ids", [])) if isinstance(veh_info, dict) else 0
    nsegs = max(1, int(nsegs)) if nsegs else 1

    # progress ratio along segments (1..n)
    try:
        ratio = float(segment_index) / float(nsegs)
    except Exception:
        ratio = 0.0

    # Helper windows (fallback if geometry inference is noisy)
    early_cut = max(1, int(math.ceil(0.35 * nsegs)))
    late_cut = max(1, int(math.floor(0.65 * nsegs)))

    if phase == "in_intersection":
        if turn_indices:
            return 5.0 if segment_index in turn_indices else -5.0
        # If no inferred turn indices, prefer the middle.
        return 2.0 if (0.35 <= ratio <= 0.65) else -2.0

    if phase == "after_turn":
        if exit_idx is not None:
            ex = int(exit_idx)
            if segment_index == ex:
                return 5.0
            if segment_index > ex:
                return 2.5
            return -5.0
        return 2.5 if ratio >= 0.55 else -2.5

    if phase == "on_approach":
        if turn_indices:
            ft = min(turn_indices)
            return 5.0 if segment_index < ft else -5.0
        return 2.5 if segment_index <= early_cut else -2.5

    if phase == "after_exit":
        if exit_idx is not None:
            ex = int(exit_idx)
            # Strongly prefer strictly after the exit connector.
            return 5.0 if segment_index > ex else -5.0
        # Fallback for straight paths: prefer the last segment
        return 5.0 if segment_index == nsegs else -5.0

    if phase == "after_merge":
        # "After merge" means on the exit road, similar to after_exit
        # Merges happen on or after the intersection, so prefer exit segment
        if exit_idx is not None:
            ex = int(exit_idx)
            return 5.0 if segment_index >= ex else -5.0
        # Fallback for straight paths: prefer the last segment
        return 5.0 if segment_index == nsegs else -5.0

    return 0.0


def _lateral_score(lat: str, pref: str) -> float:
    if pref in ("unknown", None, ""):
        return 0.0
    if lat == pref:
        return 1.5
    try:
        v1 = float(LATERAL_TO_M.get(lat, 0.0))
        v2 = float(LATERAL_TO_M.get(pref, 0.0))
    except Exception:
        return 0.0
    if v1 == 0.0 or v2 == 0.0:
        return 0.0
    if (v1 > 0) == (v2 > 0):
        return 0.4
    return -0.4

def _relation_score(a: CandidatePlacement, b: CandidatePlacement, rel: Dict[str, Any]) -> float:
    rtype = rel["type"]
    target = DIST_BUCKET_TO_M.get(rel.get("distance","unknown"))
    dx = a.x - b.x
    dy = a.y - b.y
    d = float(math.hypot(dx, dy))

    if rtype == "near":
        if target is None:
            return 0.2 * (1.0 / (1.0 + d))
        return 1.0 * math.exp(-0.5 * ((d - target) / max(1.0, 0.5 * target)) ** 2)

    # For ahead_of/behind_of, use path_s_m comparison
    # Move this check BEFORE the vehicle_num check so we can enforce distance constraints
    delta = a.path_s_m - b.path_s_m
    if target is None:
        target = 5.0

    if rtype == "ahead_of":
        # Graduated scoring for ahead_of constraints:
        # - Full reward + bonus for exceeding target distance (encourages maximizing distance)
        # - Partial reward if we're ahead but not by enough (proportional to how close we are)
        # - Strong penalty if we're behind or at the same position
        if delta >= target:
            # Give extra bonus for being farther ahead (helps leave room for downstream entities)
            excess_bonus = min(1.0, (delta - target) / target) * 0.5  # up to 0.5 extra
            return 3.0 + excess_bonus
        elif delta > 0:
            # Partial satisfaction: give proportional credit for being ahead
            # Score scales from -2.0 (delta=0) to 2.5 (delta approaching target)
            ratio = delta / target
            return -2.0 + 4.5 * ratio  # ranges from -2.0 to +2.5
        else:
            # Behind the reference entity - strong penalty
            return -5.0
    if rtype == "behind_of":
        if delta <= -target:
            excess_bonus = min(1.0, (abs(delta) - target) / target) * 0.5
            return 3.0 + excess_bonus
        elif delta < 0:
            ratio = abs(delta) / target
            return -2.0 + 4.5 * ratio
        else:
            return -5.0

    # For left_of/right_of, require same vehicle path
    if a.vehicle_num != b.vehicle_num:
        return -0.2

    la = float(LATERAL_TO_M.get(a.lateral_relation, 0.0))
    lb = float(LATERAL_TO_M.get(b.lateral_relation, 0.0))
    if rtype == "right_of":
        return 0.8 if la > lb else -0.4
    if rtype == "left_of":
        return 0.8 if la < lb else -0.4
    return 0.0

def generate_candidates_for_actor(
    spec: Dict[str, Any],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    veh_info: Dict[int, Dict[str, Any]],
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    all_segments: Optional[List[Dict[str, Any]]] = None,
    max_candidates: int = 120,
    ds_m: float = 6.0,
    crop_margin_m: float = 1.0,
) -> List[CandidatePlacement]:
    anchor = spec.get("anchor", {})
    tv = str(anchor.get("target_vehicle", "unknown"))
    pref_phase = str(anchor.get("phase", "unknown"))
    pref_lat = str(anchor.get("lateral_preference", "unknown"))
    motion = spec.get("motion", {}) if isinstance(spec.get("motion", {}), dict) else {}
    mtype = str(motion.get("type", "unknown")).lower()
    cross_dir = str(motion.get("cross_direction", "unknown")).lower()
    preferred = _parse_vehicle_num(tv)
    
    # Check if this is an NPC vehicle with "opposite" direction constraint
    direction_rel = spec.get("direction_relative_to")
    is_opposite_npc = False
    opposite_ref_vehicle = None
    if direction_rel and isinstance(direction_rel, dict):
        if direction_rel.get("direction") == "opposite":
            is_opposite_npc = True
            opposite_ref_vehicle = _parse_vehicle_num(direction_rel.get("vehicle"))

    veh_domain: List[int] = []
    for p in picked_list:
        n = _parse_vehicle_num(p.get("vehicle"))
        if n is not None:
            veh_domain.append(n)
    veh_domain = sorted(list(set(veh_domain)))
    if preferred is not None:
        veh_domain = [preferred]

    cat = str(spec.get("category","static")).lower()
    if cat == "vehicle":
        laterals = ["center", "half_right", "half_left"]
    elif cat in ("walker","cyclist"):
        laterals = ["center", "half_right", "half_left", "right_edge", "left_edge"]
        if mtype == "cross_perpendicular" and pref_lat in ("unknown", None, ""):
            if cross_dir == "right":
                laterals = ["left_edge", "half_left", "right_edge", "half_right", "center"]
            elif cross_dir == "left":
                laterals = ["right_edge", "half_right", "left_edge", "half_left", "center"]
            else:
                laterals = ["right_edge", "left_edge", "half_right", "half_left", "center"]
    else:
        laterals = ["right_edge", "left_edge", "half_right", "half_left", "center", "offroad_right", "offroad_left"]
    if pref_lat in laterals:
        laterals = [pref_lat] + [x for x in laterals if x != pref_lat]

    cands: List[CandidatePlacement] = []
    fallback_cands: List[CandidatePlacement] = []

    merge_min_s_m = None
    if pref_phase == "after_merge" and preferred is not None and merge_min_s_by_vehicle:
        base_merge_s = merge_min_s_by_vehicle.get(preferred)
        if base_merge_s is not None:
            merge_min_s_m = float(base_merge_s) + AFTER_MERGE_CLEARANCE_M + _group_back_buffer_m(spec)
    
    # Special handling for NPC vehicles that travel in the OPPOSITE direction
    if is_opposite_npc and opposite_ref_vehicle is not None and all_segments:
        opposite_segs = _find_opposite_lane_segments(
            opposite_ref_vehicle, picked_list, all_segments, seg_by_id, crop_region
        )
        if opposite_segs:
            for seg_info in opposite_segs:
                seg_id = seg_info["seg_id"]
                pts = seg_info["points"]
                seg_len = seg_info["length_m"]
                if seg_len < 2.0:
                    continue
                
                step = max(2.0, float(ds_m))
                s_values_m = list(np.arange(min(crop_margin_m, 0.2*seg_len), max(seg_len - crop_margin_m, 0.8*seg_len), step))
                if len(s_values_m) < 3:
                    s_values_m = [0.2*seg_len, 0.5*seg_len, 0.8*seg_len]
                
                for s_m in s_values_m:
                    s_along = float(min(1.0, max(0.0, s_m / seg_len)))
                    for lat in laterals:
                        spawn = compute_spawn_from_anchor(pts, s_along, lat, None)
                        if not _inside_crop_xy(spawn["x"], spawn["y"], crop_region, margin_m=crop_margin_m):
                            continue
                        base = 3.0 + _lateral_score(lat, pref_lat)  # High bonus for matching opposite direction
                        cands.append(CandidatePlacement(
                            vehicle_num=0,  # 0 indicates "not on any ego path"
                            segment_index=1,  # Single segment
                            seg_id=int(seg_id),
                            s_along=float(s_along),
                            lateral_relation=str(lat),
                            x=float(spawn["x"]),
                            y=float(spawn["y"]),
                            yaw_deg=float(spawn["yaw_deg"]),
                            path_s_m=float(s_m),
                            base_score=float(base),
                        ))
            # If we found opposite-lane candidates, return them (don't mix with ego paths)
            if cands:
                cands.sort(key=lambda c: c.base_score, reverse=True)
                return cands[:max_candidates]
    
    # Standard candidate generation for ego vehicle paths
    for veh_num in veh_domain:
        picked_entry = next((p for p in picked_list if _parse_vehicle_num(p.get("vehicle")) == veh_num), None)
        if not picked_entry:
            continue
        seg_ids = picked_entry.get("signature", {}).get("segment_ids", [])
        if not isinstance(seg_ids, list) or not seg_ids:
            continue

        seg_lengths: List[float] = []
        seg_pts_cache: List[Optional[np.ndarray]] = []
        for seg_id_raw in seg_ids:
            try:
                seg_id = int(seg_id_raw)
            except Exception:
                seg_lengths.append(0.0)
                seg_pts_cache.append(None)
                continue
            pts = seg_by_id.get(seg_id)
            if pts is None or len(pts) < 2:
                seg_lengths.append(0.0)
                seg_pts_cache.append(None)
                continue
            cum = cumulative_dist(pts)
            seg_lengths.append(float(cum[-1]))
            seg_pts_cache.append(pts)

        total_path_len = float(sum(seg_lengths))
        if merge_min_s_m is not None and total_path_len > 0.0 and merge_min_s_m >= total_path_len:
            merge_min_s_m = None

        for idx0, seg_id_raw in enumerate(seg_ids):
            segment_index = idx0 + 1
            pts = seg_pts_cache[idx0]
            seg_len = seg_lengths[idx0]
            if pts is None or seg_len < 2.0:
                continue
            step = max(2.0, float(ds_m))
            s_values_m = list(np.arange(min(crop_margin_m, 0.2*seg_len), max(seg_len - crop_margin_m, 0.8*seg_len), step))
            if len(s_values_m) < 3:
                s_values_m = [0.2*seg_len, 0.5*seg_len, 0.8*seg_len]

            info = veh_info.get(veh_num, {})
            phase_bonus = _phase_score(segment_index, info, pref_phase)

            for s_m in s_values_m:
                s_along = float(min(1.0, max(0.0, s_m / seg_len)))
                for lat in laterals:
                    spawn = compute_spawn_from_anchor(pts, s_along, lat, None)
                    if not _inside_crop_xy(spawn["x"], spawn["y"], crop_region, margin_m=crop_margin_m):
                        continue
                    path_s_m = float(sum(seg_lengths[:idx0]) + float(s_m))
                    if merge_min_s_m is not None and path_s_m < merge_min_s_m:
                        fallback_cands.append(CandidatePlacement(
                            vehicle_num=int(veh_num),
                            segment_index=int(segment_index),
                            seg_id=int(seg_id_raw),
                            s_along=float(s_along),
                            lateral_relation=str(lat),
                            x=float(spawn["x"]),
                            y=float(spawn["y"]),
                            yaw_deg=float(spawn["yaw_deg"]),
                            path_s_m=float(path_s_m),
                            base_score=float(phase_bonus + _lateral_score(lat, pref_lat) + (1.0 if preferred is not None else 0.0)),
                        ))
                        continue
                    base = phase_bonus + _lateral_score(lat, pref_lat)
                    if preferred is not None:
                        base += 1.0
                    cands.append(CandidatePlacement(
                        vehicle_num=int(veh_num),
                        segment_index=int(segment_index),
                        seg_id=int(seg_id_raw),
                        s_along=float(s_along),
                        lateral_relation=str(lat),
                        x=float(spawn["x"]),
                        y=float(spawn["y"]),
                        yaw_deg=float(spawn["yaw_deg"]),
                        path_s_m=float(path_s_m),
                        base_score=float(base),
                    ))
    if merge_min_s_m is not None and not cands and fallback_cands:
        cands = fallback_cands
    cands.sort(key=lambda c: c.base_score, reverse=True)
    return cands[:max_candidates]

def solve_weighted_csp(
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    all_segments: Optional[List[Dict[str, Any]]] = None,
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    min_sep_scale: float = 1.0,
    max_backtrack: int = 30000,
) -> Tuple[Dict[str, CandidatePlacement], Dict[str, Any]]:
    veh_info = _infer_vehicle_turn_exit_indices(picked_list, seg_by_id)
    domains: Dict[str, List[CandidatePlacement]] = {}
    for s in specs:
        sid = str(s["id"])
        domains[sid] = generate_candidates_for_actor(
            s,
            picked_list,
            seg_by_id,
            crop_region,
            veh_info,
            merge_min_s_by_vehicle=merge_min_s_by_vehicle,
            all_segments=all_segments,
        )

    # Build dependency graph: if A has relation to B, B should be placed before A
    # Topological sort to respect dependencies, then sort by domain size within each level
    spec_by_id = {str(s["id"]): s for s in specs}
    all_ids = [str(s["id"]) for s in specs]
    
    # Build adjacency: depends_on[A] = set of entities A depends on (referenced in A's relations)
    depends_on: Dict[str, set] = {sid: set() for sid in all_ids}
    for s in specs:
        sid = str(s["id"])
        for rel in s.get("relations", []):
            other = str(rel.get("other_id", ""))
            if other in spec_by_id:
                depends_on[sid].add(other)
    
    # Topological sort using Kahn's algorithm
    in_degree = {sid: len(depends_on[sid]) for sid in all_ids}
    # Start with entities that have no dependencies
    queue = [sid for sid in all_ids if in_degree[sid] == 0]
    queue.sort(key=lambda k: len(domains.get(k, [])))  # Sort by domain size within level
    
    order = []
    while queue:
        sid = queue.pop(0)
        order.append(sid)
        # Decrease in_degree for entities that depend on this one
        for other_sid in all_ids:
            if sid in depends_on[other_sid]:
                in_degree[other_sid] -= 1
                if in_degree[other_sid] == 0:
                    queue.append(other_sid)
        queue.sort(key=lambda k: len(domains.get(k, [])))
    
    # If there's a cycle, just add remaining in domain-size order
    remaining = [sid for sid in all_ids if sid not in order]
    remaining.sort(key=lambda k: len(domains.get(k, [])))
    order.extend(remaining)
    
    # Use asset_id for more accurate radius calculation when bbox is available
    radii = {sid: _actor_radius_m(
        spec_by_id[sid]["category"], 
        spec_by_id[sid]["actor_kind"],
        spec_by_id[sid].get("asset_id")
    ) for sid in order}

    best: Dict[str, CandidatePlacement] = {}
    best_score = -1e18
    nodes = 0

    best_base = {sid: (domains[sid][0].base_score if domains.get(sid) else 0.0) for sid in order}
    suffix = [0.0]*(len(order)+1)
    for i in range(len(order)-1, -1, -1):
        suffix[i] = suffix[i+1] + float(best_base[order[i]])

    def ok_hard(c: CandidatePlacement, partial: Dict[str, CandidatePlacement], sid: str) -> bool:
        """
        Check if placing actor `sid` at candidate `c` would collide with any already-placed actors.
        Uses actor radii (derived from bboxes when available) plus a minimum separation margin.
        """
        rr = float(radii[sid]) * float(min_sep_scale)
        for oid, oc in partial.items():
            r = (rr + float(radii[oid]) * float(min_sep_scale)) + MIN_ACTOR_SEPARATION_M
            dx = c.x - oc.x
            dy = c.y - oc.y
            dist_sq = dx*dx + dy*dy
            if dist_sq < r*r:
                return False
        return True

    def _find_related_placement(oid: str, partial: Dict[str, CandidatePlacement]) -> Optional[CandidatePlacement]:
        """
        Find the placement for an entity referenced by other_id.
        Handles group expansion: if oid is "entity_2" but we have "entity_2_1", "entity_2_2", etc.,
        return the first matching expanded entity (they share the same base anchor).
        """
        if oid in partial:
            return partial[oid]
        # Check for expanded group members (entity_X_1, entity_X_2, etc.)
        for pid, placement in partial.items():
            if pid.startswith(oid + "_") and pid[len(oid)+1:].isdigit():
                return placement
        return None

    def soft_delta(c: CandidatePlacement, partial: Dict[str, CandidatePlacement], sid: str) -> float:
        s = spec_by_id[sid]
        sc = float(c.base_score)
        for rel in s.get("relations", []):
            oid = rel["other_id"]
            other_placement = _find_related_placement(oid, partial)
            if other_placement is not None:
                sc += float(_relation_score(c, other_placement, rel))
        return sc

    def bt(i: int, partial: Dict[str, CandidatePlacement], score: float) -> None:
        nonlocal best, best_score, nodes
        nodes += 1
        if nodes > max_backtrack:
            return
        if score + suffix[i] < best_score:
            return
        if i >= len(order):
            if score > best_score:
                best_score = score
                best = dict(partial)
            return
        sid = order[i]
        dom = domains.get(sid, [])
        if not dom:
            return
        for cand in dom:
            if not ok_hard(cand, partial, sid):
                continue
            d = soft_delta(cand, partial, sid)
            partial[sid] = cand
            bt(i+1, partial, score+d)
            partial.pop(sid, None)

    bt(0, {}, 0.0)

    dbg = {"nodes_searched": nodes, "best_score": best_score, "domain_sizes": {k: len(v) for k,v in domains.items()}}
    if len(best) != len(order):
        # greedy fallback
        fallback: Dict[str, CandidatePlacement] = {}
        for sid in order:
            dom = domains.get(sid, [])
            chosen = None
            for cand in dom:
                if ok_hard(cand, fallback, sid):
                    chosen = cand
                    break
            if chosen is None and dom:
                chosen = dom[0]
            if chosen is not None:
                fallback[sid] = chosen
        dbg["fallback_used"] = True
        return fallback, dbg
    dbg["fallback_used"] = False
    return best, dbg


# Distance mapping for checking constraint satisfaction
_DISTANCE_TO_M_TARGET = {
    "touching": 2.0,
    "close": 6.0,
    "medium": 12.0,
    "far": 20.0,  # Target for "Twenty meters later"
}


def _check_distance_constraints(
    chosen: Dict[str, CandidatePlacement],
    specs: List[Dict[str, Any]],
    threshold_ratio: float = 0.7,  # Allow 70% of target as acceptable
) -> List[Tuple[str, str, float, float]]:
    """
    Check if distance constraints between entities are satisfied.
    
    Returns list of (entity_id, other_id, actual_distance, target_distance) for unsatisfied constraints.
    """
    spec_by_id = {str(s["id"]): s for s in specs}
    violations = []
    
    def _find_placement(entity_id: str) -> Optional[CandidatePlacement]:
        """Find placement, handling group expansion."""
        if entity_id in chosen:
            return chosen[entity_id]
        # Check for expanded group members (entity_X_1, entity_X_2, etc.)
        for pid, p in chosen.items():
            if pid.startswith(entity_id + "_") and pid[len(entity_id)+1:].split("_")[0].isdigit():
                return p
        return None
    
    for spec in specs:
        eid = str(spec.get("id", ""))
        placement = _find_placement(eid)
        if placement is None:
            continue
        
        for rel in spec.get("relations", []):
            rel_type = rel.get("type", "")
            if rel_type not in ("ahead_of", "behind_of"):
                continue
            
            other_id = str(rel.get("other_id", ""))
            other_placement = _find_placement(other_id)
            
            if other_placement is None:
                continue
            
            # Check distance
            distance_bucket = rel.get("distance", "medium")
            target = _DISTANCE_TO_M_TARGET.get(distance_bucket, 12.0)
            
            if rel_type == "ahead_of":
                actual = placement.path_s_m - other_placement.path_s_m
            else:  # behind_of
                actual = other_placement.path_s_m - placement.path_s_m
            
            # Check if constraint is satisfied (within threshold)
            if actual < target * threshold_ratio:
                violations.append((eid, other_id, actual, target))
    
    return violations


def _compute_extension_needed(
    violations: List[Tuple[str, str, float, float]],
    chosen: Dict[str, CandidatePlacement],
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Compute how much additional path length is needed for each vehicle.
    
    Returns dict mapping vehicle_num -> additional_meters_needed
    """
    spec_by_id = {str(s["id"]): s for s in specs}
    extension_needed: Dict[int, float] = {}
    
    for eid, other_id, actual, target in violations:
        spec = spec_by_id.get(eid)
        if spec is None:
            # Try with group expansion (entity_2 -> entity_2_1)
            for sid, s in spec_by_id.items():
                if sid.startswith(eid + "_"):
                    spec = s
                    break
        if spec is None:
            continue
        
        # Get vehicle number from anchor.target_vehicle
        anchor = spec.get("anchor", {})
        target_vehicle = anchor.get("target_vehicle", "")
        veh_num = None
        if target_vehicle:
            # Extract number from "Vehicle 1", "Vehicle 2", etc.
            import re
            m = re.search(r'(\d+)', target_vehicle)
            if m:
                veh_num = int(m.group(1))
        
        if veh_num is None:
            # Fallback: get from placement
            placement = chosen.get(eid)
            if placement is None:
                for pid, p in chosen.items():
                    if pid.startswith(eid + "_"):
                        placement = p
                        break
            if placement is not None:
                veh_num = placement.vehicle_num
        
        if veh_num is None:
            continue
        
        # How much more do we need?
        shortfall = target - actual
        if shortfall > 0:
            current = extension_needed.get(veh_num, 0.0)
            extension_needed[veh_num] = max(current, shortfall + 10.0)  # Add margin
    
    return extension_needed


def solve_weighted_csp_with_extension(
    specs: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    crop_region: Any,
    all_segments: List[Dict[str, Any]],
    nodes: Optional[Dict[str, Any]] = None,
    merge_min_s_by_vehicle: Optional[Dict[int, float]] = None,
    min_sep_scale: float = 1.0,
    max_backtrack: int = 30000,
    max_extension_iterations: int = 3,
) -> Tuple[Dict[str, CandidatePlacement], Dict[str, Any], Any]:
    """
    Solve CSP with iterative path extension.
    
    After each CSP solve, checks if distance constraints are satisfied.
    If not, extends paths and re-solves.
    
    Returns:
        - chosen: dict mapping entity_id -> CandidatePlacement
        - dbg: debug info dict
        - crop_region: the (possibly expanded) crop region
    """
    # Make a mutable copy of crop_region so we can expand it
    if isinstance(crop_region, dict):
        crop_region = dict(crop_region)
    
    iteration = 0
    total_extensions: Dict[int, List[int]] = {}
    
    while iteration < max_extension_iterations:
        iteration += 1
        
        # Solve CSP
        chosen, dbg = solve_weighted_csp(
            specs, picked_list, seg_by_id, crop_region,
            all_segments=all_segments,
            merge_min_s_by_vehicle=merge_min_s_by_vehicle,
            min_sep_scale=min_sep_scale,
            max_backtrack=max_backtrack,
        )
        
        # Check if distance constraints are satisfied
        violations = _check_distance_constraints(chosen, specs)
        
        if not violations:
            dbg["extension_iterations"] = iteration
            dbg["extensions_made"] = total_extensions
            return chosen, dbg, crop_region
        
        # Log violations
        print(f"[INFO] CSP iteration {iteration}: {len(violations)} distance constraint violation(s)")
        for eid, other_id, actual, target in violations:
            print(f"  - {eid} should be {target:.0f}m from {other_id}, but is only {actual:.1f}m")
        
        # Compute needed extensions
        extension_needed = _compute_extension_needed(violations, chosen, specs, picked_list, seg_by_id)
        
        if not extension_needed:
            print(f"[INFO] No extensions needed (couldn't determine vehicle)")
            break
        
        # Extend paths
        any_extended = False
        for veh_num, extra_m in extension_needed.items():
            # Find picked entry for this vehicle
            picked_entry = None
            for p in picked_list:
                if _parse_vehicle_num(p.get("vehicle")) == veh_num:
                    picked_entry = p
                    break
            
            if picked_entry is None:
                continue
            
            current_len = _compute_path_length(picked_entry, seg_by_id)
            target_len = current_len + extra_m
            
            print(f"[INFO] Vehicle {veh_num}: extending from {current_len:.1f}m to {target_len:.1f}m")
            
            was_extended, new_len, parallel_extended = extend_path_if_needed(
                picked_entry, seg_by_id, all_segments, target_len, nodes=nodes, picked_list=picked_list
            )
            
            if was_extended:
                any_extended = True
                if veh_num not in total_extensions:
                    total_extensions[veh_num] = []
                total_extensions[veh_num].append(new_len)
                print(f"[INFO] Vehicle {veh_num}: extended to {new_len:.1f}m")
                
                # Track parallel vehicle extensions
                for pv in parallel_extended:
                    if pv not in total_extensions:
                        total_extensions[pv] = []
                    total_extensions[pv].append(new_len)  # Approximate length
                
                # Expand crop_region to include the extended path points (primary + parallel vehicles)
                if isinstance(crop_region, dict):
                    # Collect all segment IDs from primary and parallel vehicles
                    all_seg_ids_to_check = []
                    
                    # Primary vehicle
                    sig = picked_entry.get("signature", {})
                    all_seg_ids_to_check.extend(sig.get("segment_ids", []))
                    
                    # Parallel vehicles
                    for pv in parallel_extended:
                        for p in picked_list:
                            if _parse_vehicle_num(p.get("vehicle")) == pv:
                                pv_sig = p.get("signature", {})
                                all_seg_ids_to_check.extend(pv_sig.get("segment_ids", []))
                                break
                    
                    for seg_id in all_seg_ids_to_check:
                        pts = seg_by_id.get(int(seg_id))
                        if pts is not None and len(pts) > 0:
                            pts = np.asarray(pts)
                            xmin = float(np.min(pts[:, 0]))
                            xmax = float(np.max(pts[:, 0]))
                            ymin = float(np.min(pts[:, 1]))
                            ymax = float(np.max(pts[:, 1]))
                            if xmin < crop_region.get("xmin", float('inf')):
                                crop_region["xmin"] = xmin - 5.0  # Add margin
                            if xmax > crop_region.get("xmax", float('-inf')):
                                crop_region["xmax"] = xmax + 5.0
                            if ymin < crop_region.get("ymin", float('inf')):
                                crop_region["ymin"] = ymin - 5.0
                            if ymax > crop_region.get("ymax", float('-inf')):
                                crop_region["ymax"] = ymax + 5.0
                    print(f"[INFO] Expanded crop region to include extended paths")
        
        if not any_extended:
            print(f"[INFO] Could not extend any paths further")
            break
    
    dbg["extension_iterations"] = iteration
    dbg["extensions_made"] = total_extensions
    dbg["unresolved_violations"] = [(e, o, f"{a:.1f}m", f"{t:.0f}m") for e, o, a, t in violations] if violations else []
    return chosen, dbg, crop_region


def _spacing_bucket_to_m(bucket: str, category: str = "static", actor_kind: str = "static_prop", asset_id: Optional[str] = None) -> float:
    """
    Convert spacing bucket to meters, ensuring minimum separation for collision avoidance.
    The spacing must be at least 2 * actor_radius + MIN_ACTOR_SEPARATION_M to prevent overlaps.
    Uses bbox-based radius if asset_id is provided and has a bounding box.
    """
    b = str(bucket)
    # Get minimum spacing based on actor radius (uses bbox if available)
    actor_radius = _actor_radius_m(category, actor_kind, asset_id)
    min_spacing = 2 * actor_radius + MIN_ACTOR_SEPARATION_M
    
    if b == "tight":
        return max(min_spacing, 1.0)
    if b == "sparse":
        return max(min_spacing, 3.0)
    if b == "normal":
        return max(min_spacing, 2.0)
    return max(min_spacing, 2.0)

def _group_back_buffer_m(spec: Dict[str, Any]) -> float:
    """
    Conservative backward offset to keep expanded groups after a phase boundary.
    """
    try:
        qty = int(spec.get("quantity", 1))
    except Exception:
        qty = 1
    if qty <= 1:
        return 0.0
    gp = spec.get("group_pattern") or {}
    spacing_m = _spacing_bucket_to_m(
        gp.get("spacing_bucket", "auto"),
        category=str(spec.get("category", "static")),
        actor_kind=str(spec.get("actor_kind", "static_prop")),
        asset_id=spec.get("asset_id"),
    )
    return 0.5 * float(qty - 1) * float(spacing_m)

def expand_group_to_actors(
    base_actor: Dict[str, Any],
    spec: Dict[str, Any],
    chosen: CandidatePlacement,
    seg_by_id: Dict[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Expand quantity>1 into multiple actor dicts with custom lateral_offset_m and/or s_along offsets.
    Works without touching world conversion: we add placement.lateral_offset_m.
    """
    qty = int(spec.get("quantity", 1))
    if qty <= 1:
        return [base_actor]

    gp = spec.get("group_pattern") or {}
    pattern = str(gp.get("pattern", "along_lane"))
    if pattern not in ALLOWED_GROUP_PATTERNS:
        pattern = "along_lane"
    
    # Get category and actor_kind for collision radius calculation
    category = spec.get("category", "static")
    actor_kind = spec.get("actor_kind", "static_prop")
    asset_id = spec.get("asset_id")
    spacing_m = _spacing_bucket_to_m(gp.get("spacing_bucket", "auto"), category=category, actor_kind=actor_kind, asset_id=asset_id)

    pts = seg_by_id.get(int(chosen.seg_id))
    if pts is None or len(pts) < 2:
        return [base_actor]

    seg_len = float(cumulative_dist(pts)[-1])
    p, t = point_and_tangent_at_s(pts, chosen.s_along)
    rn = right_normal_world(t)  # +right

    start_lat = gp.get("start_lateral")
    end_lat = gp.get("end_lateral")
    a = float(LATERAL_TO_M.get(start_lat or "right_edge", +0.45*LANE_WIDTH_M))
    b = float(LATERAL_TO_M.get(end_lat or "left_edge", -0.45*LANE_WIDTH_M))
    
    # Check if we have diagonal placement (start_lateral != end_lateral)
    has_lateral_transition = (start_lat is not None and end_lat is not None and start_lat != end_lat)
    lateral_span = abs(b - a)
    lateral_step = lateral_span / float(qty - 1) if qty > 1 else lateral_span
    needs_s_offset = lateral_step < spacing_m

    out: List[Dict[str, Any]] = []
    for i in range(qty):
        child = json.loads(json.dumps(base_actor))  # deep-ish copy
        child["id"] = f"{base_actor.get('id','entity')}_{i+1}"
        
        # Compute interpolation factor
        u = 0.0 if qty == 1 else i / float(qty - 1)

        if pattern == "across_lane":
            # Side-by-side across lane width
            lat_m = a*(1-u) + b*u
            child["placement"]["lateral_offset_m"] = float(lat_m)
            if needs_s_offset:
                offset = (i - (qty-1)/2.0) * spacing_m
                child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
                child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))
        elif pattern == "along_lane":
            # Offset along segment
            offset = (i - (qty-1)/2.0) * spacing_m
            child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
            child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))
            # If we have lateral transition (diagonal), also interpolate lateral position
            if has_lateral_transition:
                lat_m = a*(1-u) + b*u
                child["placement"]["lateral_offset_m"] = float(lat_m)
        else:  # scatter
            # small lateral jitter
            jitter = random.uniform(-1.0, 1.0)
            child["placement"]["lateral_offset_m"] = float(LATERAL_TO_M.get(chosen.lateral_relation, 0.0) + jitter)
            offset = (i - (qty-1)/2.0) * spacing_m
            child_s = float(chosen.s_along) + float(offset / max(1e-6, seg_len))
            child["placement"]["s_along"] = float(min(1.0, max(0.0, child_s)))

        out.append(child)
    return out


# ======================================================================================
# Stage-1 hallucination guardrails (post-filter)
# ======================================================================================

_GEOM_WORDS = {
    "intersection", "junction", "tjunction", "roundabout", "road", "street", "highway", "freeway",
    "lane", "lanes", "shoulder", "median", "curb", "sidewalk", "crosswalk", "ramp", "exit", "entry",
    "approach", "connector", "merge", "turn", "turning", "intersectional", "roadway",
}
_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "into", "through", "from", "for", "with",
    "and", "or", "near", "by", "around", "across", "along", "before", "after", "during",
}
_DIR_WORDS = {
    "left", "right", "center", "middle", "same", "opposite", "oncoming", "main", "side",
    "straight", "forward", "behind", "ahead", "front", "back",
}

def _norm_ws_lower(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _contains_exact_quote(description: str, quote: str) -> bool:
    """Best-effort check that quote appears in description (case-insensitive, whitespace-normalized)."""
    d = _norm_ws_lower(description)
    q = _norm_ws_lower(quote)
    if not q:
        return False
    return q in d

def _is_ego_vehicle_mention(mention: str) -> bool:
    """True iff mention is exactly 'Vehicle N' (ego naming)."""
    m = _norm_ws_lower(mention)
    # Allow users to explicitly refer to NPCs like "NPC Vehicle 1"
    if "npc" in m or "non-ego" in m or "non ego" in m:
        return False
    return re.fullmatch(r"vehicle\s+\d+", m) is not None

def _is_pure_map_geometry_phrase(phrase: str) -> bool:
    """Heuristic: phrase is basically just a map/location noun like 'the intersection' or 'left lane'."""
    s = _norm_ws_lower(phrase)
    toks = re.findall(r"[a-z0-9]+", s)
    toks = [t for t in toks if t not in _STOP_WORDS and t not in _DIR_WORDS]
    if not toks:
        return False
    return all(t in _GEOM_WORDS for t in toks)



def _should_drop_stage1_entity(e: Dict[str, Any], description: str) -> Tuple[bool, str]:
    mention = str(e.get("mention") or "").strip()
    kind = str(e.get("actor_kind") or "").strip()
    evidence = str(e.get("evidence") or "").strip()

    if not mention:
        return True, "empty mention"

    # Hard guardrail: never spawn ego vehicles as entities.
    if _is_ego_vehicle_mention(mention):
        return True, "ego vehicle mention (Vehicle N)"

    # Evidence requirement (Fix D): require a quote that actually appears in the description.
    # If evidence is missing or mismatched, fall back to mention ONLY if mention appears.
    if evidence:
        if not _contains_exact_quote(description, evidence):
            if _contains_exact_quote(description, mention):
                e["evidence"] = mention
            else:
                return True, "evidence not found in description"
    else:
        if _contains_exact_quote(description, mention):
            e["evidence"] = mention
        else:
            return True, "no evidence/mention match in description"

    # Drop obvious map-geometry pseudo-entities (Fix A), but avoid over-filtering.
    # - Only apply aggressively to static props / parked vehicles where geometry confusions are common.
    # - Keep pedestrians/cyclists/NPC vehicles even if mention contains location words.
    if kind in ("static_prop", "parked_vehicle"):
        if _is_pure_map_geometry_phrase(mention):
            return True, "map-geometry phrase (not a spawnable actor)"

    return False, ""
def run_object_placer(args, model=None, tokenizer=None):
    """
    Main pipeline body, optionally reusing a provided model/tokenizer.
    """
    t_obj_start = time.time()
    # Set default values for optional args that may not be provided by SimpleNamespace
    if not hasattr(args, 'placement_mode'):
        args.placement_mode = "csp"  # default to CSP-based placement
    if not hasattr(args, 'do_sample'):
        args.do_sample = False
    if not hasattr(args, 'temperature'):
        args.temperature = 0.2
    if not hasattr(args, 'top_p'):
        args.top_p = 0.95

    t0 = time.time()
    with open(args.picked_paths, "r", encoding="utf-8") as f:
        picked_payload = json.load(f)

    picked = picked_payload.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths_detailed.json has no 'picked' list.")

    crop_region = picked_payload.get("crop_region")
    nodes_field = picked_payload.get("nodes")
    if not nodes_field:
        raise SystemExit("[ERROR] picked_paths_detailed.json missing 'nodes' field")

    all_assets = load_assets(args.carla_assets)

    # Build vehicle segment summaries for LLM
    vehicle_segments = build_vehicle_segment_summaries(picked)
    print(f"[TIMING] object_placer setup (load paths, assets, summaries): {time.time() - t0:.2f}s", flush=True)

    # Load HF model if not provided
    if tokenizer is None or model is None:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        print(f"[TIMING] object_placer model load: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Stage 1: extract entities
    # --------------------------
    t_stage1_start = time.time()
    stage1_prompt = build_stage1_prompt(args.description)
    t0 = time.time()
    stage1_text = generate_with_model(
        model=model,
        tokenizer=tokenizer,
        prompt=stage1_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[TIMING] Stage1 LLM generation: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time()
    # Stage 1 parse (with repair if the model didn't output JSON)
    try:
        stage1_obj = parse_llm_json(stage1_text, required_top_keys=["entities"])
    except Exception:
        repair_prompt = (
            "Return JSON ONLY with top-level key 'entities' (a list). No prose.\n"
            "If you previously wrote anything else, convert it into the required JSON now.\n\n"
            "RAW OUTPUT:\n" + stage1_text
        )
        repair_text = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        stage1_obj = parse_llm_json(repair_text, required_top_keys=["entities"])
    print(f"[TIMING] Stage1 parse+repair: {time.time() - t0:.2f}s", flush=True)

    entities = stage1_obj.get("entities", [])
    if not isinstance(entities, list):
        raise SystemExit("[ERROR] Stage1: 'entities' must be a list.")

    # Ensure each entity has a unique entity_id (normalize if LLM didn't provide one)
    valid_entity_ids = set()
    for idx, e in enumerate(entities):
        if not e.get("entity_id"):
            e["entity_id"] = f"entity_{idx + 1}"
        valid_entity_ids.add(e["entity_id"])

    
    # Post-filter Stage 1 entities to reduce hallucinations (Fix A + Fix D)
    t0 = time.time()
    dropped_stage1: List[Tuple[str, str, str]] = []
    filtered_entities: List[Dict[str, Any]] = []

    def _repair_evidence_with_llm(ent: Dict[str, Any]) -> None:
        """Best-effort: if Stage1 paraphrased, try to recover an EXACT supporting quote."""
        ev = str(ent.get("evidence") or "").strip()
        mention = str(ent.get("mention") or "").strip()
        if ev and _contains_exact_quote(args.description, ev):
            return
        if mention and _contains_exact_quote(args.description, mention):
            ent["evidence"] = mention
            return

        # One-shot repair: ask the model to point to an exact substring.
        try:
            prompt = (
                "Return JSON ONLY: {\"evidence\": \"...\"}.\n"
                "The evidence MUST be an EXACT substring (<=20 words) copied from DESCRIPTION that explicitly mentions the actor (not just a location).\n"
                "If you cannot find any supporting substring, return {\"evidence\": \"\"}.\n\n"
                f"ACTOR_KIND: {ent.get('actor_kind','')}\n"
                f"MENTION (may be paraphrase): {mention}\n\n"
                f"DESCRIPTION:\n{args.description}\n"
            )
            txt = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            obj = parse_llm_json(txt, required_top_keys=["evidence"])
            rep = str(obj.get("evidence") or "").strip()
            if rep and _contains_exact_quote(args.description, rep):
                ent["evidence"] = rep
        except Exception:
            return

    for e in entities:
        _repair_evidence_with_llm(e)

        drop, reason = _should_drop_stage1_entity(e, args.description)
        if drop:
            dropped_stage1.append((str(e.get("entity_id")), reason, str(e.get("mention") or "")))
            continue
        filtered_entities.append(e)
    if dropped_stage1:
        for ent_id, reason, mention in dropped_stage1[:50]:
            print(f"[WARNING] Stage1 drop: {ent_id}: {reason}; mention='{mention[:60]}'")
    entities = filtered_entities
    valid_entity_ids = set(str(e.get("entity_id")) for e in entities if e.get("entity_id"))
    print(f"[TIMING] Stage1 entity filtering+evidence repair: {time.time() - t0:.2f}s", flush=True)
    print(f"[TIMING] Stage1 total: {time.time() - t_stage1_start:.2f}s", flush=True)

    print(f"[INFO] Stage1 extracted {len(entities)} entities: {list(valid_entity_ids)}")
    # Debug: show Stage 1 entity details including motion_hint
    for e in entities:
        print(f"  - {e.get('entity_id')}: kind={e.get('actor_kind')}, motion_hint={e.get('motion_hint')}, mention='{e.get('mention', '')[:50]}'")

    # Keyword synonyms for better asset matching
    t0 = time.time()
    KEYWORD_SYNONYMS = {
        "cyclist": ["bike", "bicycle", "crossbike"],
        "bicyclist": ["bike", "bicycle", "crossbike"],
        "biker": ["bike", "bicycle", "crossbike", "motorcycle"],
        "motorcyclist": ["motorcycle", "harley", "yamaha", "kawasaki"],
        "pedestrian": ["pedestrian", "walker", "person"],
        "person": ["pedestrian", "walker"],
        "cone": ["cone", "trafficcone", "constructioncone"],
        "cones": ["cone", "trafficcone", "constructioncone"],
        "traffic cone": ["cone", "trafficcone", "constructioncone"],
        "barrier": ["barrier", "streetbarrier", "construction"],
        "truck": ["truck", "firetruck", "cybertruck", "pickup"],
        "police": ["police", "charger", "crown"],
        "ambulance": ["ambulance"],
        "firetruck": ["firetruck"],
    }

    # For each entity, build small asset option list (keyed by entity_id)
    per_entity_options: Dict[str, List[Dict[str, Any]]] = {}
    for idx, e in enumerate(entities):
        entity_id = e.get("entity_id", f"entity_{idx+1}")
        mention = str(e.get("mention", f"entity_{idx+1}"))
        kind = str(e.get("actor_kind", "static_prop"))
        low = mention.lower()

        # Extract keywords: split mention into words and check synonyms
        kws = set()
        words = [w.strip(".,!?") for w in low.split()]
        
        # Add all meaningful words from mention
        for word in words:
            if len(word) > 2:  # Skip tiny words
                kws.add(word)
            # Expand synonyms
            if word in KEYWORD_SYNONYMS:
                kws.update(KEYWORD_SYNONYMS[word])
        
        # Also check multi-word phrases
        for phrase, synonyms in KEYWORD_SYNONYMS.items():
            if phrase in low:
                kws.update(synonyms)

        # Add kind-based keywords (always, not just as fallback)
        if kind == "walker":
            categories = ["walker"]
            kws.update(["pedestrian", "walker"])
        elif kind == "cyclist":
            categories = ["vehicle"]
            kws.update(["bike", "bicycle", "crossbike"])
        elif kind in ("parked_vehicle", "npc_vehicle"):
            categories = ["vehicle"]
            # Keep mention-based keywords; add generic fallbacks only if empty
            if not any(w in kws for w in ["car", "truck", "bus", "van", "vehicle"]):
                kws.update(["car", "vehicle"])
        else:
            categories = ["static"]
            kws.update(["prop", "static"])

        kws_list = list(kws)
        options = keyword_filter_assets(all_assets, kws_list, categories=categories, k=12)

        # last-resort fallback options
        if not options:
            # choose a small default set by category
            options = keyword_filter_assets(all_assets, ["vehicle"], categories=categories, k=12) or all_assets[:12]

        # Key by entity_id for Stage 2
        per_entity_options[entity_id] = [
            {"asset_id": a.asset_id, "category": a.category, "tags": a.tags[:6]} for a in options
        ]
    print(f"[TIMING] asset matching for {len(entities)} entities: {time.time() - t0:.2f}s", flush=True)

    # Handle empty entities case early
    if not entities:
        print("[INFO] No entities extracted in Stage 1. Skipping Stage 2.")
        actors = []
        stage2_obj = {"actors": []}
    else:
        # --------------------------
        # Stage 2: resolve anchors
        # --------------------------
        t_stage2_start = time.time()

        if args.placement_mode == "llm_anchor":
            stage2_prompt = build_stage2_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("\n[DEBUG] Stage2 raw output (full):\n" + stage2_text + "\n", flush=True)

            stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actors"])
            actors = stage2_obj.get("actors", [])
            if not isinstance(actors, list):
                raise SystemExit("[ERROR] Stage2: 'actors' must be a list.")

            errs = validate_stage2_output(actors, vehicle_segments)
            if errs:
                repair_prompt = build_repair_prompt(stage2_text, errs)
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
                actors = stage2_obj.get("actors", [])
                if not isinstance(actors, list):
                    raise SystemExit("[ERROR] Stage2 repair: 'actors' must be a list.")
            print(f"[TIMING] Stage2 (llm_anchor mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)
        else:
            # CSP mode: LLM emits symbolic preferences; solver chooses anchors.
            # We need geometry (seg_by_id) to enumerate candidates; build it here.
            t0 = time.time()
            resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
            if not os.path.exists(resolved_nodes_path):
                raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                                 f"Tip: pass --nodes-root to resolve relative paths.")
            nodes = load_nodes(resolved_nodes_path)
            all_segments = build_segments_from_nodes(nodes)
            seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
            # Override with refined polylines from picked paths (so CSP uses accurate path lengths)
            seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
            merge_min_s_by_vehicle = _compute_merge_min_s_by_vehicle(picked_payload, picked, seg_by_id)
            print(f"[TIMING] Stage2 load nodes+build segments: {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            stage2_prompt = build_stage2_constraints_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"[TIMING] Stage2 LLM generation: {time.time() - t0:.2f}s", flush=True)
            print("\n[DEBUG] Stage2(CSP) raw output (full):\n" + stage2_text + "\n", flush=True)

            # Parse with a simple repair-on-failure
            t0 = time.time()
            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actor_specs"])
            except Exception:
                repair_prompt = (
                    "Return JSON ONLY with top-level key 'actor_specs' (a list). No prose.\n"
                    "If needed, convert your previous output into the required JSON now.\n\n"
                    "RAW OUTPUT:\n" + stage2_text
                )
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actor_specs"])
            print(f"[TIMING] Stage2 parse+repair: {time.time() - t0:.2f}s", flush=True)

            actor_specs_raw = stage2_obj.get("actor_specs", [])
            actor_specs, warns = validate_actor_specs(actor_specs_raw, entities, per_entity_options, picked=picked)
            for w in warns[:50]:
                print("[WARNING] Stage2(CSP): " + w)

            if not actor_specs:
                actors = []
            else:
                t0 = time.time()
                print(f"[TIMING] Starting CSP solve with {len(actor_specs)} actor specs...", flush=True)
                # Use iterative CSP with path extension - extends paths on-demand when distance constraints can't be met
                # Returns expanded crop_region to include extended path areas
                chosen, dbg, crop_region = solve_weighted_csp_with_extension(
                    actor_specs, picked, seg_by_id, crop_region,
                    all_segments=all_segments,
                    nodes=nodes,
                    merge_min_s_by_vehicle=merge_min_s_by_vehicle,
                    min_sep_scale=1.0,
                    max_extension_iterations=3,
                )
                print(f"[TIMING] CSP solve (with extension): {time.time() - t0:.2f}s", flush=True)
                print("[INFO] CSP solve debug: " + json.dumps(dbg, indent=2))

                actors = []
                for spec in actor_specs:
                    sid = str(spec["id"])
                    cand = chosen.get(sid)
                    if cand is None:
                        continue

                    # Get bounding box info if available
                    asset_id = spec["asset_id"]
                    bbox = get_asset_bbox(asset_id)
                    bbox_info = None
                    if bbox:
                        bbox_info = {
                            "length": bbox.length,
                            "width": bbox.width,
                            "height": bbox.height,
                        }

                    base_actor = {
                        "id": sid,
                        "semantic": spec.get("semantic", sid),
                        "category": spec["category"],
                        "asset_id": asset_id,
                        "placement": {
                            "target_vehicle": f"Vehicle {cand.vehicle_num}" if cand.vehicle_num > 0 else None,
                            "segment_index": int(cand.segment_index),
                            "s_along": float(cand.s_along),
                            "lateral_relation": str(cand.lateral_relation),
                            "seg_id": int(cand.seg_id),  # Store seg_id directly for opposite-lane NPCs
                        },
                        "motion": spec.get("motion", {"type": "static", "speed_profile": "normal"}),
                        "confidence": float(spec.get("confidence", 0.6)),
                        "csp": {
                            "base_score": float(cand.base_score),
                            "path_s_m": float(cand.path_s_m),
                            "relations": spec.get("relations", []),
                        },
                        "bbox": bbox_info,  # Include bounding box if available
                    }

                    expanded = expand_group_to_actors(base_actor, spec, cand, seg_by_id)
                    actors.extend(expanded)
            print(f"[TIMING] Stage2 (CSP mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)


    # --------------------------
    # Geometry reconstruction
    # --------------------------
    t0 = time.time()
    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")

    nodes = load_nodes(resolved_nodes_path)
    all_segments = build_segments_from_nodes(nodes)
    seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
    # If paths were refined (start/end trimming or synthetic segments), prefer those polylines.
    seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
    print(f"[TIMING] geometry reconstruction: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Convert anchors -> world
    # --------------------------
    t0 = time.time()
    # Guardrail: if Stage1 said "after_turn" but Stage2 placed the actor on a turning connector,
    # shift it onto the inferred post-turn (exit) segment before spawning.
    apply_after_turn_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)

    actors_world: List[Dict[str, Any]] = []
    for a in actors:
        placement = a["placement"]
        tv = placement.get("target_vehicle")
        seg_idx = int(placement.get("segment_index", 1))  # 1-based
        s_along = float(placement["s_along"])
        lat_rel = placement["lateral_relation"]
        
        # Check if seg_id is directly specified (for opposite-lane NPCs)
        direct_seg_id = placement.get("seg_id")
        
        if direct_seg_id is not None:
            # Direct seg_id: use it directly without looking up picked paths
            seg_id = int(direct_seg_id)
            seg_pts = seg_by_id.get(seg_id)
        else:
            # Standard lookup via target_vehicle and picked paths
            # Find seg_id from the picked path signature order
            # vehicle_segments contains seg_id list in order via picked signature; easiest: pull from picked itself
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                continue
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx < 1 or seg_idx > len(seg_ids):
                continue
            seg_id = int(seg_ids[seg_idx - 1])

            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                # fall back to polyline_sample if present
                segs_det = (picked_entry.get("signature", {}) or {}).get("segments_detailed", [])
                det = next((d for d in segs_det if int(d.get("seg_id", -1)) == seg_id), None)
                if det and isinstance(det.get("polyline_sample"), list) and det["polyline_sample"]:
                    seg_pts = np.array([[p["x"], p["y"]] for p in det["polyline_sample"]], dtype=float)

        if seg_pts is None:
            print(f"[WARNING] Missing segment geometry for seg_id={seg_id}; skipping actor {a.get('id')}")
            continue

        spawn = compute_spawn_from_anchor(seg_pts, s_along, lat_rel, placement.get("lateral_offset_m"))
        motion = a.get("motion", {}) if isinstance(a.get("motion", {}), dict) else {"type": "static"}
        # Let motion builder know anchor s for some types
        motion.setdefault("anchor_s_along", s_along)
        
        # Pass start_lateral for crossing direction inference
        motion.setdefault("start_lateral", lat_rel)

        # category normalization
        cat = str(a.get("category", "")).lower()
        if cat not in ("vehicle", "walker", "static", "cyclist"):
            # derive from asset category if needed
            # (we don't strictly enforce this)
            cat = "static"

        wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
        if isinstance(wps, list) and wps and isinstance(wps[0], dict) and "x" in wps[0] and "y" in wps[0]:
            # Align spawn to the first waypoint so spawn == trajectory start.
            spawn = {
                "x": float(wps[0]["x"]),
                "y": float(wps[0]["y"]),
                "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
            }

        actors_world.append({
            **a,
            "resolved": {
                "seg_id": seg_id,
                "nodes_path": resolved_nodes_path,
            },
            "spawn": spawn,
            "world_waypoints": wps,
        })
    print(f"[TIMING] anchor -> world conversion: {time.time() - t0:.2f}s", flush=True)

    out_payload = {
        "source_picked_paths": args.picked_paths,
        "nodes": resolved_nodes_path,
        "crop_region": crop_region,
        "ego_picked": picked,
        "actors": actors_world,
        "macro_plan": stage1_obj.get("entities", []),
    }

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote scene objects to: {args.out} (actors={len(actors_world)})")

    if args.viz:
        t_viz = time.time()
        visualize(
            picked=picked,
            seg_by_id=seg_by_id,
            actors_world=actors_world,
            crop_region=crop_region if isinstance(crop_region, dict) else None,
            out_path=args.viz_out,
            description=args.description,
            show=args.viz_show,
        )
        print(f"[TIMING] visualization: {time.time() - t_viz:.2f}s", flush=True)
    print(f"[TIMING] object_placer total (internal): {time.time() - t_obj_start:.2f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--carla-assets", required=True, help="carla_assets.json")

    ap.add_argument("--description", required=True, help="Natural-language scene description")
    ap.add_argument("--out", default="scene_objects.json", help="Output IR + placements JSON")
    ap.add_argument("--viz-out", default="scene_objects.png", help="Output visualization image")
    ap.add_argument("--viz", action="store_true", help="Enable visualization")
    ap.add_argument("--viz-show", action="store_true", help="Show plot window (if supported)")

    ap.add_argument("--nodes-root", default=None, help="Optional root to resolve relative nodes path")
    ap.add_argument("--placement-mode", default="csp", choices=["csp","llm_anchor"], help="Placement stage: weighted CSP (solver) or legacy LLM anchors")

    # LLM gen controls
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)

    args = ap.parse_args()
    run_object_placer(args)


if __name__ == "__main__":
    main()
