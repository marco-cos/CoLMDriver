#!/usr/bin/env python3
import argparse
import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Only used for visualization (geometry). Kept optional and isolated.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -------------------------
# Output parsing utilities
# -------------------------

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract any top-level JSON object from arbitrary text using a balanced-brace scan.
    Tries full-text JSON first; then scans each brace-balanced snippet; returns the
    first one that parses and contains a top-level 'vehicles'.
    """
    # Fast path: direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "vehicles" in obj:
            return obj
    except Exception:
        pass

    start_search = 0
    n = len(text)
    while True:
        start = text.find("{", start_search)
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
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start:i + 1]
                        try:
                            obj = json.loads(snippet)
                            if isinstance(obj, dict) and "vehicles" in obj:
                                return obj
                        except Exception:
                            break
                i += 1
        start_search = start + 1
    return None


def _extract_json_from_codeblocks(text: str) -> Optional[Dict[str, Any]]:
    """Try parsing JSON from ```json ... ``` blocks."""
    for m in re.finditer(r"```(?:json)?\s*\n([\s\S]*?)```", text):
        block = m.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "vehicles" in obj:
                return obj
        except Exception:
            continue
    return None


def _extract_vehicles_loose(text: str) -> Optional[Dict[str, Any]]:
    """
    Last-resort loose extraction: gather pairs of vehicle/path_name from text
    without requiring valid JSON. Also captures optional confidence if nearby.
    """
    items: List[Dict[str, Any]] = []
    # Prefer explicit "path_name": "..." pairs
    for pm in re.finditer(r"\"path_name\"\s*:\s*\"([^\"]+)\"", text):
        path_name = pm.group(1)
        # Look back a bit for vehicle and confidence
        window_start = max(0, pm.start() - 300)
        ctx = text[window_start:pm.start()]
        vm = re.search(r"\"vehicle\"\s*:\s*\"([^\"]+)\"", ctx)
        cm = re.search(r"\"confidence\"\s*:\s*([0-9]*\.?[0-9]+)", ctx)
        vehicle = vm.group(1) if vm else f"Vehicle {len(items) + 1}"
        entry: Dict[str, Any] = {"vehicle": vehicle, "path_name": path_name}
        if cm:
            try:
                entry["confidence"] = float(cm.group(1))
            except Exception:
                pass
        items.append(entry)

    # If none found, fall back to any path_### token found in order
    if not items:
        paths = [m.group(0) for m in re.finditer(r"path_\d{3}[^\s\"']*", text)]
        for i, p in enumerate(paths):
            items.append({"vehicle": f"Vehicle {i+1}", "path_name": p})

    return {"vehicles": items} if items else None


def _safe_parse_model_output(text: str) -> Optional[Dict[str, Any]]:
    """Robust parse order: (1) balanced-brace JSON (2) codeblock JSON (3) loose regex."""
    obj = _extract_first_json_object(text)
    if obj:
        return obj
    obj = _extract_json_from_codeblocks(text)
    if obj:
        return obj
    obj = _extract_vehicles_loose(text)
    if obj:
        return obj
    return None


# -------------------------
# Constraint extraction + CSP solver (minimal, deterministic)
# -------------------------

_OPPOSITE_CARDINAL = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _opposite_cardinal(c: str) -> str:
    return _OPPOSITE_CARDINAL.get(str(c).strip().upper(), "unknown")


# Canonical, unambiguous constraint names used everywhere (prompt + IR + solver)
_ALLOWED_CONSTRAINT_TYPES = {
    "same_approach_as",         # same ENTRY direction (approach direction)
    "opposite_approach_of",     # opposite ENTRY direction (oncoming)
    "perpendicular_right_of",   # entry is 90 degrees clockwise from reference vehicle's entry
    "perpendicular_left_of",    # entry is 90 degrees counter-clockwise from reference vehicle's entry
    "same_exit_as",             # same EXIT direction
    "same_road_as",             # same PHYSICAL road_id after the intersection (direction may differ)
    "follow_route_of",         # follower should take same route (entry + maneuver + exit corridor)
    "left_lane_of",             # adjacent lane immediately to the left (on approach)
    "right_lane_of",            # adjacent lane immediately to the right (on approach)
    "merges_into_lane_of",      # lane change: starts in adjacent lane, ends in same lane as reference
}


def _detect_npc_vehicles(description: str) -> set:
    """
    Detect vehicles that are explicitly marked as NPC vehicles in the description.
    Returns a set of vehicle names that should NOT be treated as ego vehicles.
    
    Patterns detected:
    - "Vehicle X is an NPC vehicle"
    - "Vehicle X is an NPC"
    - "Vehicle X is a non-player vehicle"
    """
    npc_vehicles = set()
    desc_lower = description.lower()
    
    # Pattern: "Vehicle N is an NPC" or "Vehicle N is an NPC vehicle"
    pattern = r"vehicle\s+(\d+)\s+is\s+(?:an?\s+)?(?:npc|non-?player)"
    for match in re.finditer(pattern, desc_lower):
        vehicle_name = f"Vehicle {match.group(1)}"
        npc_vehicles.add(vehicle_name)
    
    return npc_vehicles


def _extract_description_from_prompt(prompt: str) -> str:
    """
    Best-effort extraction of the scene description from a combined prompt.
    Expected marker: 'USER SCENARIO DESCRIPTION:' (used by run_scenario_pipeline).
    """
    if not prompt:
        return ""
    if "USER SCENARIO DESCRIPTION:" in prompt:
        desc = prompt.split("USER SCENARIO DESCRIPTION:", 1)[1].strip()
    else:
        desc = prompt.strip()

    desc = re.sub(r"\(Only assign paths to moving vehicles;.*?\)\s*$", "", desc, flags=re.S).strip()
    return desc


def _build_constraints_prompt(description: str) -> str:
    """
    Prompt the LLM to output ONLY constraints (no path names).
    Schema is intentionally small + aligned with natural language.
    """
    return (
        "You will read a short driving scene description.\n"
        "Extract constraints about EGO vehicles only (Vehicle 1, Vehicle 2, ...).\n"
        "Do not extract constraints about static objects such as parked vehicles, pedestrians, bicyclists, or any other prop.\n"
        "Do NOT choose path names here; only list constraints.\n"
        "\n"
        "IMPORTANT VEHICLE CLASSIFICATION:\n"
        "- EGO VEHICLES: Vehicles mentioned by name (Vehicle 1, Vehicle 2, etc.) that are NOT explicitly\n"
        "  described as \"NPC vehicle\", \"NPC\", or \"non-player character\". These need paths.\n"
        "- NPC VEHICLES: If the description says \"Vehicle X is an NPC vehicle\" or similar,\n"
        "  DO NOT include that vehicle in the 'vehicles' list. NPCs are placed as actors, not ego paths.\n"
        "  Still include constraints involving NPC vehicles if they help define other ego vehicle paths.\n"
        "\n"
        "KEY DEFINITIONS (use these exactly):\n"
        "- approach direction = the direction the vehicle APPROACHES the intersection from (ENTRY).\n"
        "- road role = whether a road is described as the \"main road\" or \"side road\".\n"
        "- lane relation = whether one ego vehicle starts in the lane to the left/right of another ego vehicle on approach.\n"
        "- same_road_as means the vehicles end up on the same PHYSICAL road (same road_id),\n"
        "  even if they travel opposite directions.\n"
        "\n"
        "Return JSON ONLY with this schema:\n"
        "{\n"
        "  \"vehicles\": [\n"
        "    {\n"
        "      \"vehicle\": \"Vehicle 1\",\n"
        "      \"maneuver\": \"left\" | \"right\" | \"straight\" | \"lane_change\" | \"unknown\",\n"
        "      \"lane_change_phase\": \"before_intersection\" | \"after_intersection\" | \"unknown\",\n"
        "      \"entry_road\": \"main\" | \"side\" | \"unknown\",\n"
        "      \"exit_road\": \"main\" | \"side\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"constraints\": [\n"
        "    {\n"
        "      \"type\": \"same_approach_as\" | \"opposite_approach_of\" | \"perpendicular_right_of\" | \"perpendicular_left_of\" | \"same_exit_as\" | \"same_road_as\" | \"follow_route_of\" | \"left_lane_of\" | \"right_lane_of\" | \"adjacent_lane_of\" | \"merges_into_lane_of\",\n"
        "      \"a\": \"Vehicle X\",\n"
        "      \"b\": \"Vehicle Y\",\n"
        "      \"evidence\": \"<COPIED VERBATIM from DESCRIPTION (no paraphrase)>\">\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "CRITICAL RULES (MUST FOLLOW):\n"
        "- Only include EGO vehicles in the 'vehicles' list. NPC vehicles are NOT ego vehicles.\n"
        "- Only extract constraints that are EXPLICITLY stated in the text.\n"
        "- evidence MUST be an EXACT substring copied from DESCRIPTION (no paraphrase, no synonyms).\n- keep evidence SHORT (<= 12 words).\n- keep the JSON compact: include at most 10 constraints total; omit low-confidence extras.\n"
        "- Do NOT output duplicate constraints: each (type,a,b) may appear at most once.\n"
        "- Do NOT infer entry_road or exit_road from the maneuver alone.\n"
        "- If the text does not literally say \"main road\" or \"side road\" for that vehicle, use \"unknown\".\n"
        "- Do NOT infer same_road_as from maneuvers. Only emit same_road_as if the description explicitly says\n"
        "  they are on the same road or one turns onto the road of the other.\n"
        "- Constraints without clear evidence will be discarded, so when unsure, omit it.\n"
        "- If a vehicle CHANGES LANES, set maneuver=\"lane_change\".\n"
        "- If the lane change happens \"after the intersection\" or \"on the exit road\", set lane_change_phase=\"after_intersection\".\n"
        "- If the lane change happens \"before the intersection\" or \"on approach\", set lane_change_phase=\"before_intersection\".\n"
        "- If lane change timing is not specified, use lane_change_phase=\"unknown\".\n"
        "\n"
        "DISAMBIGUATION RULES (MUST FOLLOW):\n"
        "- Phrases like \"following\", \"behind\", \"tailing\" refer to a ROUTE-FOLLOWING relation (spawn/route coupling).\n"
        "  => use type=\"follow_route_of\".\n"
        "- Phrases like \"going the same direction as\" refer to APPROACH direction only.\n"
        "  => use type=\"same_approach_as\".\n"
        "- Phrases like \"coming from the opposite direction\" refer to APPROACH direction.\n"
        "  => use type=\"opposite_approach_of\".\n"
        "- Phrases like \"approaches from the perpendicular road to the right of Vehicle A's approach\"\n"
        "  => use type=\"perpendicular_right_of\". This means Vehicle a's entry is 90 degrees clockwise\n"
        "     from Vehicle b's entry (e.g., if b is westbound, a is southbound).\n"
        "- Phrases like \"approaches from the perpendicular road to the left of Vehicle A's approach\"\n"
        "  => use type=\"perpendicular_left_of\". This means Vehicle a's entry is 90 degrees counter-clockwise\n"
        "     from Vehicle b's entry (e.g., if b is westbound, a is northbound).\n"
        "- Phrases like \"turns onto Vehicle A's exit road\" or \"onto the road Vehicle A exits on\"\n"
        "  => use type=\"same_exit_as\". This means both vehicles have the same EXIT direction.\n"
        "- Phrases like \"turns onto the road Vehicle A is traveling on\" refer to PHYSICAL ROAD identity.\n"
        "  => use type=\"same_road_as\" (do NOT guess direction).\n"
        "- Phrases like \"in the lane to the left/right of Vehicle A\" refer to a LANE relation on approach.\n"
        "  => use type=\"left_lane_of\" or type=\"right_lane_of\". Do NOT treat this as road role.\n"
        "- Phrases like \"in an adjacent lane\" or \"in a different lane\" (without specifying left/right)\n"
        "  => use type=\"adjacent_lane_of\". This means they must be in different lanes.\n"
        "- Phrases like \"changes lanes into Vehicle A's lane\" or \"merges into Vehicle A's lane\"\n"
        "  => use type=\"merges_into_lane_of\". This means Vehicle a STARTS in an adjacent lane but\n"
        "     ENDS in the same lane as Vehicle b (lane change/merge scenario).\n"
        "\n"
        "SAFE PROPAGATION (ALLOWED INFERENCE):\n"
        "- If Vehicle A has entry_road explicitly set (main/side) AND the text says Vehicle B is\n"
        "  going the same direction as / follow_route_of Vehicle A, then set Vehicle B entry_road\n"
        "  to match Vehicle A.\n"
        "- If Vehicle A has exit_road explicitly set (main/side) AND the text says Vehicle B turns onto\n"
        "  the road Vehicle A is traveling on, then set Vehicle B exit_road to match Vehicle A.\n"
        "- Still do NOT infer road roles from left/right/straight alone.\n"
        "\n"
        "GUIDANCE:\n"
        "- Prefer binary constraints when the text is relational.\n"
        "- Prefer \"unknown\" over guessing.\n"
        "\n"
        f"DESCRIPTION:\n{description}\n"
    )


def _safe_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse first JSON object from text (reuses existing robust extractors)."""
    obj = _extract_first_json_object(text)
    if obj:
        return obj
    obj = _extract_json_from_codeblocks(text)
    if obj:
        return obj
    return None


def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def _sanitize_constraints_obj(
    constraints_obj: Dict[str, Any],
    description: str,
    max_constraints: int = 8,
) -> Dict[str, Any]:
    """
    Deterministically drop hallucinated/invalid constraints:
    - evidence must be a literal substring of description (after whitespace-normalized lowercase compare)
    - a/b must refer to vehicles in vehicles[]
    - type must be in allowed set
    - dedupe (type,a,b)
    - cap total constraints
    - filter out NPC vehicles from the vehicle list
    """
    if not isinstance(constraints_obj, dict):
        return constraints_obj

    # Detect NPC vehicles from description
    npc_vehicles = _detect_npc_vehicles(description)
    if npc_vehicles:
        print(f"[INFO] Detected NPC vehicles (not ego): {npc_vehicles}")

    vehicles = constraints_obj.get("vehicles", [])
    if not isinstance(vehicles, list):
        vehicles = []

    # Filter out NPC vehicles from the vehicle list
    filtered_vehicles = []
    for v in vehicles:
        if isinstance(v, dict) and isinstance(v.get("vehicle"), str):
            vname = v["vehicle"].strip()
            if vname in npc_vehicles:
                print(f"[INFO] Removing NPC vehicle from ego list: {vname}")
                continue
            filtered_vehicles.append(v)
    
    constraints_obj["vehicles"] = filtered_vehicles

    valid_names = set()
    for v in filtered_vehicles:
        if isinstance(v, dict) and isinstance(v.get("vehicle"), str) and v["vehicle"].strip():
            valid_names.add(v["vehicle"].strip())

    desc_norm = _norm_ws(description)

    raw_constraints = constraints_obj.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raw_constraints = []

    kept: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for c in raw_constraints:
        if not isinstance(c, dict):
            continue
        t = str(c.get("type", "")).strip().lower()
        a = str(c.get("a", "")).strip()
        b = str(c.get("b", "")).strip()
        ev = c.get("evidence", "")

        # Backward-compatible lane relation recovery:
        # Some models confuse lane relations ("lane to the left/right") with same_road_as.
        # If evidence explicitly mentions a lane relation, rewrite to left/right_lane_of.
        # Convention: left_lane_of/right_lane_of are asymmetric, meaning:
        #   left_lane_of:  a is in the lane to the LEFT of b (on approach)
        #   right_lane_of: a is in the lane to the RIGHT of b (on approach)
        if isinstance(ev, str) and ev.strip():
            ev_l = ev.lower()
            ev_norm_tmp = _norm_ws(ev_l)
            if "lane" in ev_norm_tmp:
                if "left" in ev_norm_tmp:
                    t = "left_lane_of"
                elif "right" in ev_norm_tmp:
                    t = "right_lane_of"

                # Try to orient (a,b) consistently with phrases like "...left of Vehicle X".
                # If we can identify a referenced vehicle in the evidence, set it as b.
                m = re.search(r"(?:to\s+the\s+)?(?:left|right)\s+of\s+(vehicle\s*\d+)", ev_l)
                if m:
                    ref = " ".join(m.group(1).split()).title()  # "Vehicle 1"
                    if ref in (a, b):
                        other_name = b if a == ref else a
                        a, b = other_name, ref

        if t not in _ALLOWED_CONSTRAINT_TYPES:
            continue
        if not a or not b or a == b:
            continue
        if a not in valid_names or b not in valid_names:
            # drops pedestrians/props/etc even if the model tried
            continue
        if not isinstance(ev, str) or not ev.strip():
            continue

        ev_norm = _norm_ws(ev)
        if ev_norm not in desc_norm:
            # evidence is not copied verbatim from description => hallucination
            continue

        key = (t, a, b)
        if key in seen:
            continue
        seen.add(key)

        kept.append({"type": t, "a": a, "b": b, "evidence": ev})
        if len(kept) >= max_constraints:
            break

    constraints_obj["constraints"] = kept
    return constraints_obj


def _normalize_constraints_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize slightly different keys into the expected {'vehicles': [...], 'constraints': [...]}."""
    if not isinstance(obj, dict):
        return None

    vehicles = obj.get("vehicles")
    if vehicles is None and isinstance(obj.get("unary"), dict):
        vehicles = []
        for v, u in obj["unary"].items():
            if not isinstance(u, dict):
                continue
            vehicles.append({
                "vehicle": v,
                "maneuver": u.get("maneuver", "unknown"),
                "entry_road": u.get("entry_road", "unknown"),
                "exit_road": u.get("exit_road", "unknown"),
            })

    constraints = obj.get("constraints")
    if constraints is None and isinstance(obj.get("binary"), list):
        constraints = obj["binary"]

    if not isinstance(vehicles, list) or not vehicles:
        return None
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        constraints = []

    v_out = []
    for it in vehicles:
        if not isinstance(it, dict):
            continue
        name = it.get("vehicle")
        if not isinstance(name, str) or not name.strip():
            continue
        v_out.append({
            "vehicle": name.strip(),
            "maneuver": str(it.get("maneuver", "unknown")).strip().lower(),
            "lane_change_phase": str(it.get("lane_change_phase", "unknown")).strip().lower(),
            "entry_road": str(it.get("entry_road", "unknown")).strip().lower(),
            "exit_road": str(it.get("exit_road", "unknown")).strip().lower(),
        })

    c_out = []
    for it in constraints:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).strip().lower()
        a = str(it.get("a", "")).strip()
        b = str(it.get("b", "")).strip()
        ev = str(it.get("evidence", "")).strip()
        if not t or not a or not b:
            continue
        # Keep only the small canonical set (avoids ambiguous / unsupported constraints)
        if t not in _ALLOWED_CONSTRAINT_TYPES:
            continue
        entry = {"type": t, "a": a, "b": b}
        if ev:
            entry["evidence"] = ev
        c_out.append(entry)

    return {"vehicles": v_out, "constraints": c_out}


def _filter_constraints_with_evidence(description: str, norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop constraints that lack evidence or whose evidence/trigger is not present in the description.
    Also deduplicate identical (type,a,b) tuples.
    """
    desc_l = (description or "").lower()

    def has_trigger(c: Dict[str, Any]) -> bool:
        ev = c.get("evidence", "")
        ev_l = str(ev).lower()
        if not ev_l or ev_l not in desc_l:
            return False
        t = c.get("type", "")
        if t == "same_approach_as":
            triggers = ["same direction", "following", "behind", "same approach", "same lane", "coming from the same"]
        elif t == "opposite_approach_of":
            triggers = ["opposite direction", "oncoming", "coming toward", "from the opposite"]
        elif t == "perpendicular_right_of":
            triggers = ["perpendicular", "to the right of", "right of vehicle", "cross traffic"]
        elif t == "perpendicular_left_of":
            triggers = ["perpendicular", "to the left of", "left of vehicle", "cross traffic"]
        elif t == "same_exit_as":
            triggers = ["onto vehicle", "exit road", "onto the road", "turns onto"]
        elif t == "same_road_as":
            triggers = ["same road", "onto the road", "onto vehicle", "onto the roadway", "on the road of"]
        elif t == "follow_route_of":
            triggers = ["following", "behind", "tailing", "same route", "same path"]
        elif t == "left_lane_of":
            triggers = ["lane to the left", "left lane", "in the left lane"]
        elif t == "right_lane_of":
            triggers = ["lane to the right", "right lane", "in the right lane"]
        elif t == "merges_into_lane_of":
            triggers = ["changes lanes into", "merges into", "lane change", "into vehicle", "into the lane"]
        else:
            return False
        return any(tok in ev_l or tok in desc_l for tok in triggers)

    seen = set()
    filtered = []
    for c in norm.get("constraints", []):
        t = c.get("type")
        a = c.get("a")
        b = c.get("b")
        ev = c.get("evidence", "")

        # Heuristic: if evidence says "left/right of Vehicle X" but the extracted
        # constraint points the other way, swap a/b to align with the text.
        if t in ("left_lane_of", "right_lane_of") and isinstance(ev, str):
            ev_l = ev.lower()
            m = re.search(r"(left|right)[^\n]{0,40}?\bof\s+(vehicle\s*\d+)", ev_l)
            if m:
                side = m.group(1)
                ref = " ".join(m.group(2).split()).title()  # "Vehicle 1"
                # Only swap when the side matches the constraint type and ref matches one end.
                if ((t == "left_lane_of" and side == "left") or (t == "right_lane_of" and side == "right")):
                    if ref == a and ref != b:
                        a, b = b, a
                    elif ref == b and ref != a:
                        # already consistent
                        pass

        c_use = dict(c)
        c_use["a"], c_use["b"] = a, b

        key = (c_use.get("type"), c_use.get("a"), c_use.get("b"))
        if key in seen:
            continue
        if has_trigger(c_use):
            filtered.append(c_use)
            seen.add(key)
    out = dict(norm)
    out["constraints"] = filtered
    return out


def _infer_road_role_sets(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Infer which cardinal directions correspond to the 'main road' vs 'side road' using candidates:
    - Main road directions are the entry cardinals that have straight-through paths (entry==exit, maneuver==straight).
    - Side road directions are the remaining entry cardinals.
    """
    entry_set = set()
    exit_set = set()
    main_cardinals = set()

    for c in candidates:
        sig = (c or {}).get("signature", {})
        ent = (sig or {}).get("entry", {})
        ex = (sig or {}).get("exit", {})
        ent_c = str((ent or {}).get("cardinal4", "")).strip().upper()
        ex_c = str((ex or {}).get("cardinal4", "")).strip().upper()
        if ent_c:
            entry_set.add(ent_c)
        if ex_c:
            exit_set.add(ex_c)

        if str((sig or {}).get("entry_to_exit_turn", "")).strip().lower() == "straight" and ent_c and ent_c == ex_c:
            main_cardinals.add(ent_c)

    side_entry = set([c for c in entry_set if c not in main_cardinals])

    def expand_with_opposites(s: set, universe: set) -> set:
        out = set(s)
        for c in list(s):
            oc = _opposite_cardinal(c)
            if oc in universe:
                out.add(oc)
        return out

    main_exit = expand_with_opposites(main_cardinals, exit_set)
    side_exit = expand_with_opposites(side_entry, exit_set)

    # Road-role (main/side) is only meaningful when the intersection behaves like a T-junction.
    # In 4-way (or otherwise ambiguous) intersections, we intentionally disable this distinction.
    is_t_junction = (len(entry_set) == 3 and len(main_cardinals) >= 1 and len(side_entry) >= 1)
    if not is_t_junction:
        main_cardinals = set()
        side_entry = set()
        main_exit = set()
        side_exit = set()

    return {
        "entry_set": entry_set,
        "exit_set": exit_set,
        "main_entry": main_cardinals,
        "side_entry": side_entry,
        "main_exit": main_exit,
        "side_exit": side_exit,
    }


def _candidate_matches_unary(
    cand: Dict[str, Any],
    maneuver: str,
    entry_road: str,
    exit_road: str,
    role_sets: Dict[str, Any],
) -> bool:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    ex = (sig or {}).get("exit", {})
    ent_c = str((ent or {}).get("cardinal4", "")).strip().upper()
    ex_c = str((ex or {}).get("cardinal4", "")).strip().upper()
    man = str((sig or {}).get("entry_to_exit_turn", "")).strip().lower()

    if maneuver in ("left", "right", "straight") and man != maneuver:
        return False

    # Fix A2: If the inferred role set is empty (ambiguous intersection), ignore the road-role filter.
    main_entry = role_sets.get("main_entry", set())
    side_entry = role_sets.get("side_entry", set())
    main_exit = role_sets.get("main_exit", set())
    side_exit = role_sets.get("side_exit", set())

    if entry_road == "main" and main_entry and ent_c not in main_entry:
        return False
    if entry_road == "side" and side_entry and ent_c not in side_entry:
        return False

    if exit_road == "main" and main_exit and ex_c not in main_exit:
        return False
    if exit_road == "side" and side_exit and ex_c not in side_exit:
        return False

    return True


def _candidate_entry_lane_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    lid = (ent or {}).get("lane_id", None)
    try:
        return int(lid) if lid is not None else None
    except Exception:
        return None


def _candidate_exit_lane_id(cand: Dict[str, Any]) -> Optional[int]:
    """Get the exit lane ID from the path signature."""
    sig = (cand or {}).get("signature", {})
    ext = (sig or {}).get("exit", {})
    lid = (ext or {}).get("lane_id", None)
    try:
        return int(lid) if lid is not None else None
    except Exception:
        return None


def _candidate_entry_point(cand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    p = (ent or {}).get("point", None)
    if not isinstance(p, dict):
        return None
    try:
        return float(p.get("x")), float(p.get("y"))
    except Exception:
        return None


def _candidate_entry_heading_rad(cand: Dict[str, Any]) -> Optional[float]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    hd = (ent or {}).get("heading_deg", None)
    try:
        return math.radians(float(hd)) if hd is not None else None
    except Exception:
        return None


def _candidate_exit_road_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    rid = (ex or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_entry_road_id(cand: Dict[str, Any]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    rid = (ent or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_all_road_ids(cand: Dict[str, Any]) -> set:
    """Return set of all road_ids that this path travels through (entry + exit)."""
    roads = set()
    ent_rid = _candidate_entry_road_id(cand)
    ex_rid = _candidate_exit_road_id(cand)
    if ent_rid is not None:
        roads.add(ent_rid)
    if ex_rid is not None:
        roads.add(ex_rid)
    return roads


def _build_road_corridors(candidates: List[Dict[str, Any]]) -> Dict[int, set]:
    """
    Build a mapping from each road_id to its 'corridor' - the set of road_ids
    that are connected by straight-through paths. In a T-junction, the main road
    might have road_id=10 on one side and road_id=11 on the other, but they're
    the same logical corridor.
    """
    # Find pairs of roads connected by straight paths
    corridor_pairs = []
    for c in candidates:
        sig = (c or {}).get("signature", {})
        if str((sig or {}).get("entry_to_exit_turn", "")).strip().lower() == "straight":
            ent_rid = _candidate_entry_road_id(c)
            ex_rid = _candidate_exit_road_id(c)
            if ent_rid is not None and ex_rid is not None:
                corridor_pairs.append((ent_rid, ex_rid))

    # Build union-find to merge corridors
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # Merge all straight-through connected roads
    for a, b in corridor_pairs:
        union(a, b)

    # Also add all other roads as their own corridor
    for c in candidates:
        ent_rid = _candidate_entry_road_id(c)
        ex_rid = _candidate_exit_road_id(c)
        if ent_rid is not None:
            find(ent_rid)
        if ex_rid is not None:
            find(ex_rid)

    # Build corridor sets
    corridors = {}
    for rid in parent:
        root = find(rid)
        if root not in corridors:
            corridors[root] = set()
        corridors[root].add(rid)

    # Map each road to its full corridor set
    road_to_corridor = {}
    for rid in parent:
        root = find(rid)
        road_to_corridor[rid] = corridors[root]

    return road_to_corridor


def _candidate_entry_cardinal(cand: Dict[str, Any]) -> str:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    return str((ent or {}).get("cardinal4", "")).strip().upper()


def _candidate_exit_cardinal(cand: Dict[str, Any]) -> str:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    return str((ex or {}).get("cardinal4", "")).strip().upper()


def _candidate_length(cand: Dict[str, Any]) -> float:
    sig = (cand or {}).get("signature", {})
    try:
        return float((sig or {}).get("length_m", 0.0))
    except Exception:
        return 0.0


def _candidate_lane_change_count(cand: Dict[str, Any]) -> int:
    """Count lane-id transitions *within the same road/section corridor*.

    This avoids counting lane_id changes that happen simply because the vehicle turned
    onto a different road/section (which is not a 'lane change' in the behavioral sense).
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return 0
    roads = sig.get("roads", [])
    sections = sig.get("sections", [])

    count = 0
    prev_lane = None
    prev_road = None
    prev_section = None

    for i, lid in enumerate(lanes):
        road = roads[i] if isinstance(roads, list) and i < len(roads) else None
        sec = sections[i] if isinstance(sections, list) and i < len(sections) else None

        try:
            lane_int = int(lid)
        except Exception:
            prev_lane = lid
            prev_road = road
            prev_section = sec
            continue

        same_corridor = True
        try:
            if prev_road is not None and road is not None and int(road) != int(prev_road):
                same_corridor = False
            if prev_section is not None and sec is not None and int(sec) != int(prev_section):
                same_corridor = False
        except Exception:
            pass

        if (
            same_corridor
            and prev_lane is not None
            and isinstance(prev_lane, int)
            and lane_int != prev_lane
        ):
            count += 1

        prev_lane = lane_int
        prev_road = road
        prev_section = sec

    return count


def _candidate_straight_lateral_drift(cand: Dict[str, Any]) -> int:
    """For straight maneuvers, count how many intermediate segments deviate from entry/exit lane.

    Even if road IDs change (junction segments), a straight path that enters lane -2
    and exits lane -2 but uses lane -1 in the middle is 'drifting' and should be penalized.
    Returns the number of segments that deviate from the entry lane.
    """
    sig = (cand or {}).get("signature", {})
    man = str(sig.get("entry_to_exit_turn", "")).strip().lower()
    if man != "straight":
        return 0

    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 3:
        return 0

    try:
        entry_lane = int(lanes[0])
        exit_lane = int(lanes[-1])
    except Exception:
        return 0

    # Only penalize if entry and exit are in the same lane (true straight)
    if entry_lane != exit_lane:
        return 0

    drift_count = 0
    for lid in lanes[1:-1]:
        try:
            if int(lid) != entry_lane:
                drift_count += 1
        except Exception:
            pass

    return drift_count


def _candidate_lane_inconsistency(cand: Dict[str, Any]) -> int:
    """Detect lane inconsistency: entry lane != exit lane, or intermediate drift.
    
    This applies to ALL maneuvers (straight, left, right) and penalizes paths where:
    1. Entry and exit are in different lanes (unexpected lane change)
    2. Intermediate segments deviate from expected lane progression
    
    Returns a penalty count (0 = clean, 1+ = has issues).
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return 0

    try:
        entry_lane = int(lanes[0])
        exit_lane = int(lanes[-1])
    except Exception:
        return 0

    # For turning maneuvers, the "natural" exit lane depends on the turn:
    # - Right turn from lane -1 naturally exits to rightmost lane (-1 or -2 depending on road)
    # - Left turn naturally exits to leftmost lane
    # But for simplicity, we check if lanes are inconsistent within entry/exit segments
    man = str(sig.get("entry_to_exit_turn", "")).strip().lower()
    
    # Count how many unique lanes are used
    unique_lanes = set()
    for lid in lanes:
        try:
            unique_lanes.add(int(lid))
        except Exception:
            pass
    
    # If more than 2 unique lanes, definitely has drift
    if len(unique_lanes) > 2:
        return len(unique_lanes) - 1
    
    # For straight paths, entry should equal exit
    if man == "straight" and entry_lane != exit_lane:
        return 1
    
    # For turns, check if the exit segment lane is consistent
    # (i.e., doesn't change lanes AFTER completing the turn)
    if man in ("left", "right") and len(lanes) >= 3:
        # Check if the last two segments are in the same lane (no post-turn lane change)
        try:
            last_lane = int(lanes[-1])
            second_last = int(lanes[-2])
            if last_lane != second_last:
                return 1  # Lane change after turn
        except Exception:
            pass
    
    return 0


def _candidate_lane_change_phase(cand: Dict[str, Any]) -> str:
    """Detect WHERE in the path the lane change occurs.
    
    Returns:
    - "before_intersection": lane change happens in the first segment (approach)
    - "in_intersection": lane change happens in the middle segment(s) (junction)
    - "after_intersection": lane change happens in the last segment (exit)
    - "none": no lane change detected
    - "multiple": lane changes in multiple phases
    """
    sig = (cand or {}).get("signature", {})
    lanes = sig.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) < 2:
        return "none"
    
    # For a 3-segment path: [approach, junction, exit]
    # segment 0 = before intersection
    # segment 1 = in intersection (or junction connector)
    # segment 2 = after intersection
    
    change_phases = set()
    prev_lane = None
    
    for i, lid in enumerate(lanes):
        try:
            lane_int = int(lid)
        except Exception:
            prev_lane = lid
            continue
        
        if prev_lane is not None and isinstance(prev_lane, int) and lane_int != prev_lane:
            # Lane change detected between segment i-1 and i
            # This means the change "happens" as we enter segment i
            n_segs = len(lanes)
            if n_segs <= 2:
                # Can't determine phase with only 2 segments
                change_phases.add("unknown")
            elif i == 1:
                # Change entering segment 1 (from approach to junction) = before/in intersection
                change_phases.add("in_intersection")
            elif i == n_segs - 1:
                # Change entering the last segment = after intersection
                change_phases.add("after_intersection")
            else:
                # Change in middle segments = in intersection
                change_phases.add("in_intersection")
        
        prev_lane = lane_int
    
    if not change_phases:
        return "none"
    if len(change_phases) > 1:
        return "multiple"
    return change_phases.pop()


def _candidate_geometric_discontinuity(cand: Dict[str, Any], threshold_m: float = 2.0) -> float:
    """Detect geometric discontinuity between segments.
    
    Even if lane IDs are consistent, the actual geometry may have gaps
    where one segment ends and the next begins at a different position.
    This happens at junctions where lane mappings don't align geometrically.
    
    Returns total gap distance in meters (0 = smooth, >0 = has gaps).
    """
    sig = (cand or {}).get("signature", {})
    segs = sig.get("segments_detailed", [])
    if not isinstance(segs, list) or len(segs) < 2:
        return 0.0
    
    total_gap = 0.0
    for i in range(len(segs) - 1):
        seg_curr = segs[i]
        seg_next = segs[i + 1]
        
        # Get end point of current segment and start point of next
        end_pt = seg_curr.get("end", {}).get("point", {})
        start_pt = seg_next.get("start", {}).get("point", {})
        
        try:
            end_x = float(end_pt.get("x", 0))
            end_y = float(end_pt.get("y", 0))
            start_x = float(start_pt.get("x", 0))
            start_y = float(start_pt.get("y", 0))
            
            # Calculate Euclidean distance
            gap = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
            if gap > threshold_m:
                total_gap += gap
        except Exception:
            pass
    
    return total_gap


def _consistent_with_constraints(
    assignment: Dict[str, Dict[str, Any]],
    candidate_for_v: Dict[str, Any],
    v: str,
    constraints: List[Dict[str, str]],
    road_corridors: Optional[Dict[int, set]] = None,
) -> bool:
    ent_v = _candidate_entry_cardinal(candidate_for_v)
    exrid_v = _candidate_exit_road_id(candidate_for_v)
    all_roads_v = _candidate_all_road_ids(candidate_for_v)

    for c in constraints:
        t = str(c.get("type", "")).strip().lower()
        a = c.get("a")
        b = c.get("b")
        if not a or not b:
            continue

        other = None
        if a == v and b in assignment:
            other = (b, assignment[b])
        elif b == v and a in assignment:
            other = (a, assignment[a])
        else:
            continue

        _, o_cand = other
        ent_o = _candidate_entry_cardinal(o_cand)
        exit_o = _candidate_exit_cardinal(o_cand)
        exrid_o = _candidate_exit_road_id(o_cand)
        all_roads_o = _candidate_all_road_ids(o_cand)

        if t == "same_approach_as":
            if ent_v != ent_o:
                return False
        elif t == "opposite_approach_of":
            if ent_v != _opposite_cardinal(ent_o):
                return False
        elif t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            # W->S, S->E, E->N, N->W
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if a == v:
                # v is "a", we need v's entry to be 90° clockwise from other's entry
                expected = clockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                # v is "b", we need other's entry to be 90° clockwise from v's entry
                expected = clockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            # If b is W (westbound), a should be N (northbound) - 90° counter-clockwise
            # W->N, N->E, E->S, S->W
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if a == v:
                expected = counterclockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                expected = counterclockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "same_exit_as":
            # Both vehicles should have the same exit direction
            exit_v = _candidate_exit_cardinal(cand)
            if exit_v != exit_o:
                return False
        elif t == "same_road_as":
            # "same_road_as" means vehicles end up on the same logical road corridor.
            # This is satisfied if:
            # 1. Their exit road_ids are in the same corridor (connected by straight paths), OR
            # 2. One vehicle's exit road is among the roads the other travels on
            if exrid_v is None or exrid_o is None:
                # Cannot determine, don't over-constrain
                continue

            # Check if exit roads are in the same corridor
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:  # corridors overlap
                    same_corridor = True

            # Check if one's exit is in the other's traversed roads
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)

            if not same_corridor and not exit_in_other_path:
                return False

    return True


def _solve_paths_csp(
    constraints_obj: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    description: str = "",
) -> List[Dict[str, Any]]:
    """
    Soft CSP/backtracking over candidate paths.
    Returns list of {"vehicle": ..., "path_name": ..., "confidence": ...} for each vehicle.
    Soft penalties are applied for constraint violations; the best-scoring assignment is chosen.
    """
    norm = _normalize_constraints_obj(constraints_obj)
    if not norm:
        raise ValueError("Invalid constraints object (missing vehicles list).")

    norm = _filter_constraints_with_evidence(description, norm)
    vehicles = [v["vehicle"] for v in norm["vehicles"]]
    unary = {v["vehicle"]: v for v in norm["vehicles"]}
    constraints = norm.get("constraints", [])
    
    # DEBUG: print extracted constraints
    print(f"[DEBUG CSP] Extracted constraints: {constraints}")
    for v_obj in norm["vehicles"]:
        print(f"[DEBUG CSP] Vehicle {v_obj.get('vehicle')}: maneuver={v_obj.get('maneuver')}")

    # Identify which vehicles are explicitly performing lane changes
    # This is determined by the LLM's extracted maneuver field - no regex needed
    vehicles_with_lane_change: set = set()
    for v in vehicles:
        if unary.get(v, {}).get("maneuver", "").lower() == "lane_change":
            vehicles_with_lane_change.add(v)
    
    # Extract lane change phase requirements for vehicles doing lane changes
    lane_change_phase_req: Dict[str, str] = {}
    for v in vehicles_with_lane_change:
        phase = str(unary.get(v, {}).get("lane_change_phase", "unknown")).strip().lower()
        if phase in ("before_intersection", "after_intersection"):
            lane_change_phase_req[v] = phase
    
    if lane_change_phase_req:
        print(f"[INFO] Lane change phase requirements: {lane_change_phase_req}")

    def _candidate_effective_length(cand: Dict[str, Any], vehicle: str) -> float:
        base = _candidate_length(cand)
        lanes = (cand.get("signature") or {}).get("lanes", [])
        
        # Only skip penalty if THIS vehicle is doing a lane change
        if vehicle in vehicles_with_lane_change:
            # Check if there's a phase requirement
            required_phase = lane_change_phase_req.get(vehicle)
            if required_phase:
                actual_phase = _candidate_lane_change_phase(cand)
                # Heavy penalty if lane change happens at wrong phase
                if actual_phase not in ("none", "unknown", required_phase):
                    return 0.0  # Zero score for wrong phase
                # Bonus for matching phase
                if actual_phase == required_phase:
                    return base + 100.0  # Large bonus for correct phase
            return base
        
        # Geometric discontinuity (gaps between segments) is the PRIMARY indicator
        # If geometry is smooth (geo=0), the path is good regardless of lane IDs
        geo_gap = _candidate_geometric_discontinuity(cand, threshold_m=2.0)
        
        if geo_gap == 0:
            # Geometrically smooth path - minimal penalty
            # Only penalize if there's actual lane drift in straight paths
            drift_count = _candidate_straight_lateral_drift(cand)
            if drift_count > 0:
                return max(0.0, base - 200.0 * drift_count)
            return base
        
        # Path has geometric gaps - apply penalties
        lc_count = _candidate_lane_change_count(cand)
        drift_count = _candidate_straight_lateral_drift(cand)
        inconsistency = _candidate_lane_inconsistency(cand)
        
        # Combine penalties
        lane_penalty = max(lc_count + drift_count, inconsistency) * 200.0
        geo_penalty = geo_gap * 50.0  # 50m penalty per meter of gap
        
        total_penalty = lane_penalty + geo_penalty
        
        return max(0.0, base - total_penalty)

    role_sets = _infer_road_role_sets(candidates)
    road_corridors = _build_road_corridors(candidates)

    # Domains (filtered by unary constraints)
    domains: Dict[str, List[Dict[str, Any]]] = {}
    for v in vehicles:
        u = unary.get(v, {})
        man = str(u.get("maneuver", "unknown")).strip().lower()
        # Treat "lane_change" as "straight" for path filtering (they go straight but change lanes)
        if man == "lane_change":
            man = "straight"
        er = str(u.get("entry_road", "unknown")).strip().lower()
        xr = str(u.get("exit_road", "unknown")).strip().lower()
        dom = [c for c in candidates if _candidate_matches_unary(c, man, er, xr, role_sets)]
        dom.sort(key=lambda c: (-_candidate_effective_length(c, v), str(c.get("name", ""))))
        domains[v] = dom

    for v, dom in domains.items():
        if not dom:
            raise ValueError(f"No candidates satisfy unary constraints for {v}.")

    order = sorted(vehicles, key=lambda v: (len(domains[v]), v))

    # Soft scoring: maximize (total_length - penalty_weight * violations)
    penalty_weight = 100.0
    max_len_per_v = {v: max(_candidate_effective_length(c, v) for c in domains[v]) for v in vehicles}

    def _constraint_penalty(c_v: Dict[str, Any], c_o: Dict[str, Any], t: str, v_is_a: bool) -> float:
        ent_v = _candidate_entry_cardinal(c_v)
        ent_o = _candidate_entry_cardinal(c_o)
        exrid_v = _candidate_exit_road_id(c_v)
        exrid_o = _candidate_exit_road_id(c_o)
        all_roads_v = _candidate_all_road_ids(c_v)
        all_roads_o = _candidate_all_road_ids(c_o)

        def _lane_relation_ok(a_c: Dict[str, Any], b_c: Dict[str, Any], relation: str) -> Optional[bool]:
            """Return True/False if checkable, else None (insufficient signal => don't over-constrain)."""
            # Must be the same approach direction; lane relations are defined on approach.
            if _candidate_entry_cardinal(a_c) != _candidate_entry_cardinal(b_c):
                return False

            # If entry road_id is known for both, require them to match.
            rid_a = _candidate_entry_road_id(a_c)
            rid_b = _candidate_entry_road_id(b_c)
            if rid_a is not None and rid_b is not None and rid_a != rid_b:
                return False

            # If lane_id is known for both, prefer adjacent lanes.
            lid_a = _candidate_entry_lane_id(a_c)
            lid_b = _candidate_entry_lane_id(b_c)
            if lid_a is not None and lid_b is not None and abs(lid_a - lid_b) != 1:
                return False

            pa = _candidate_entry_point(a_c)
            pb = _candidate_entry_point(b_c)
            th = _candidate_entry_heading_rad(b_c)
            if pa is None or pb is None or th is None:
                return None

            dx = pa[0] - pb[0]
            dy = pa[1] - pb[1]
            # Left normal for b's heading.
            left_x = -math.sin(th)
            left_y = math.cos(th)
            lat = dx * left_x + dy * left_y

            # Small deadband to avoid numerical flicker.
            eps = 0.5
            if relation == "left":
                return lat > eps
            if relation == "right":
                return lat < -eps
            return None

        if t == "same_approach_as":
            return 0.0 if ent_v == ent_o else 1.0
        if t == "opposite_approach_of":
            return 0.0 if ent_v == _opposite_cardinal(ent_o) else 1.0
        if t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if v_is_a:
                expected = clockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = clockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if v_is_a:
                expected = counterclockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = counterclockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "same_exit_as":
            exit_v = _candidate_exit_cardinal(c_v)
            exit_o = _candidate_exit_cardinal(c_o)
            return 0.0 if exit_v == exit_o else 1.0
        if t == "same_road_as":
            if exrid_v is None or exrid_o is None:
                return 0.0
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:
                    same_corridor = True
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "follow_route_of":
            # Strong preference that the follower takes the same *route* as the leader:
            # same approach + same maneuver + same exit corridor (when checkable).
            turn_v = str(((c_v.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()
            turn_o = str(((c_o.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()

            if ent_v != ent_o:
                return 1.0

            # If either turn is unknown, don't over-constrain on turn; but if both known, require match.
            if turn_v and turn_o and (turn_v != turn_o):
                return 1.0

            if exrid_v is None or exrid_o is None:
                return 0.0

            corridor_v = road_corridors.get(exrid_v, {exrid_v}) if road_corridors else {exrid_v}
            corridor_o = road_corridors.get(exrid_o, {exrid_o}) if road_corridors else {exrid_o}
            same_corridor = bool(corridor_v & corridor_o)
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "left_lane_of":
            # Asymmetric: "a" is in the lane to the left of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "left") if v_is_a else _lane_relation_ok(c_o, c_v, "left")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "right_lane_of":
            # Asymmetric: "a" is in the lane to the right of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "right") if v_is_a else _lane_relation_ok(c_o, c_v, "right")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "adjacent_lane_of":
            # Symmetric: vehicles must be in different (adjacent) lanes on approach.
            # Same approach direction required.
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't check lane adjacency
            lid_v = _candidate_entry_lane_id(c_v)
            lid_o = _candidate_entry_lane_id(c_o)
            if lid_v is None or lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            if lid_v == lid_o:
                return 1.0  # Same lane = violation
            if abs(lid_v - lid_o) == 1:
                return 0.0  # Adjacent lanes = satisfied
            return 0.5  # Not adjacent but different = partial penalty
        if t == "merges_into_lane_of":
            # Asymmetric: "a" starts in adjacent lane but ENDS in same lane as "b"
            # This is for lane change scenarios where vehicle a merges into vehicle b's lane
            # Same approach direction required
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't merge
            
            if v_is_a:
                # c_v is the merger (a), c_o is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_v)
                entry_lid_o = _candidate_entry_lane_id(c_o)
                exit_lid_v = _candidate_exit_lane_id(c_v)
                exit_lid_o = _candidate_exit_lane_id(c_o)
            else:
                # c_o is the merger (a), c_v is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_o)
                entry_lid_o = _candidate_entry_lane_id(c_v)
                exit_lid_v = _candidate_exit_lane_id(c_o)
                exit_lid_o = _candidate_exit_lane_id(c_v)
            
            if entry_lid_v is None or entry_lid_o is None or exit_lid_v is None or exit_lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            
            penalty = 0.0
            # Entry lanes must be DIFFERENT (adjacent preferred)
            if entry_lid_v == entry_lid_o:
                penalty += 1.0  # Same entry lane = violation (should start in different lane)
            elif abs(entry_lid_v - entry_lid_o) != 1:
                penalty += 0.3  # Not adjacent = partial penalty
            
            # Exit lanes must be the SAME (merge into reference's lane)
            if exit_lid_v != exit_lid_o:
                penalty += 1.0  # Different exit lane = violation (should merge into same lane)
            
            return penalty
        return 0.0

    best_assignment: Optional[Dict[str, Dict[str, Any]]] = None
    best_score = -1e18

    def backtrack(i: int, asn: Dict[str, Dict[str, Any]], cur_len: float, cur_pen: float):
        nonlocal best_assignment, best_score
        if i >= len(order):
            score = cur_len - penalty_weight * cur_pen
            if score > best_score:
                best_score = score
                best_assignment = dict(asn)
            return

        # optimistic bound for pruning
        remaining = sum(max_len_per_v[v] for v in order[i:])
        bound = cur_len + remaining - penalty_weight * cur_pen
        if bound < best_score:
            return

        vname = order[i]
        for cand in domains[vname]:
            add_pen = 0.0
            for c in constraints:
                t = str(c.get("type", "")).strip().lower()
                a = c.get("a")
                b = c.get("b")
                if not a or not b:
                    continue
                other = None
                if a == vname and b in asn:
                    other = asn[b]
                elif b == vname and a in asn:
                    other = asn[a]
                if other is not None:
                    add_pen += _constraint_penalty(cand, other, t, v_is_a=(a == vname))
            asn[vname] = cand
            backtrack(i + 1, asn, cur_len + _candidate_effective_length(cand, vname), cur_pen + add_pen)
            del asn[vname]

    backtrack(0, {}, 0.0, 0.0)

    if not best_assignment:
        raise ValueError("No feasible assignment found.")

    # Print final assignment summary
    print("[DEBUG CSP] Final assignment:")
    for v in vehicles:
        c = best_assignment[v]
        sig = c.get("signature", {})
        lanes = sig.get("lanes", [])
        entry_lane = sig.get("entry", {}).get("lane_id", "?")
        entry_card = sig.get("entry", {}).get("cardinal4", "?")
        is_lc = v in vehicles_with_lane_change
        eff_len = _candidate_effective_length(c, v)
        incon = _candidate_lane_inconsistency(c)
        print(f"  {v}: entry={entry_card} lane={entry_lane} lanes={lanes} eff_len={eff_len:.0f}m incon={incon} lc_vehicle={is_lc}")

    out = []
    for v in vehicles:
        c = best_assignment[v]
        # Encode a soft confidence: 1.0 if no penalties, else lower
        confidence = 1.0
        out.append({"vehicle": v, "path_name": str(c.get("name", "")), "confidence": confidence})
    return out


# -------------------------
# Candidate name matching
# -------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def _best_fuzzy_match(requested: str, choices: List[str]) -> Optional[str]:
    """Very conservative fuzzy match for slightly mangled names."""
    if not choices:
        return None
    rn = _normalize(requested)
    best: Tuple[float, str] = (0.0, "")
    for c in choices:
        score = SequenceMatcher(None, rn, _normalize(c)).ratio()
        if score > best[0]:
            best = (score, c)
    return best[1] if best[0] >= 0.92 else None


# -------------------------
# Visualization (optional)
# -------------------------

def _load_nodes(nodes_path: str) -> Dict[str, Any]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{nodes_path} missing top-level 'payload'")
    return data


def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _build_segments_minimal(nodes: Dict[str, Any], min_points: int = 6) -> List[Dict[str, Any]]:
    """
    Minimal reimplementation of build_segments() just for visualization.
    Returns a list of dict segments:
      {seg_id, road_id, lane_id, section_id, points: [(x,y),...]}
    """
    import numpy as np
    from collections import defaultdict

    payload = nodes["payload"]
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    def unit_from_yaw(yaw_deg: float) -> np.ndarray:
        r = np.radians(_wrap180(float(yaw_deg)))
        return np.array([np.cos(r), np.sin(r)], dtype=float)

    def orient_polyline(pts: np.ndarray, yaws_deg: np.ndarray, idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(pts) < 2:
            return pts, yaws_deg, idxs
        vecs = pts[1:] - pts[:-1]
        norms = (np.linalg.norm(vecs, axis=1) + 1e-9)
        dir_vecs = vecs / norms[:, None]
        yaw_vecs = np.vstack([unit_from_yaw(y) for y in yaws_deg[:-1]])
        dots = np.sum(dir_vecs * yaw_vecs, axis=1)
        if float(np.nanmean(dots)) < 0.0:
            return pts[::-1].copy(), yaws_deg[::-1].copy(), idxs[::-1].copy()
        return pts, yaws_deg, idxs

    def split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws_deg: np.ndarray, gap_m: float = 6.0):
        if len(pts) < 2:
            return [(idxs_sorted, pts, yaws_deg)] if len(pts) > 0 else []
        jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        cuts = [0]
        for i, d in enumerate(jumps):
            if float(d) > gap_m:
                cuts.append(i + 1)
        cuts.append(len(pts))
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= 2:
                out.append((idxs_sorted[a:b], pts[a:b], yaws_deg[a:b]))
        return out

    segments = []
    seg_id_counter = 0

    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]
        for idxs_chunk, pts_chunk, yaws_chunk in split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue
            segments.append({
                "seg_id": int(seg_id_counter),
                "road_id": int(rid),
                "lane_id": int(lid),
                "section_id": int(sid),
                "points": [(float(p[0]), float(p[1])) for p in pts_o],
            })
            seg_id_counter += 1

    return segments


def _plot_paths_together(
    all_segments: List[Dict[str, Any]],
    picked: List[Dict[str, Any]],
    crop: Optional[Dict[str, Any]],
    out_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available; install it or disable --viz")

    seg_by_id = {s["seg_id"]: s for s in all_segments}

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")

    if crop and all(k in crop for k in ("xmin", "xmax", "ymin", "ymax")):
        xmin, xmax, ymin, ymax = crop["xmin"], crop["xmax"], crop["ymin"], crop["ymax"]
        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        ax.invert_xaxis()
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linestyle="--", linewidth=2)
        ax.add_patch(rect)

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Picked Paths (n={len(picked)})")

    cmap = plt.cm.get_cmap("tab20")

    for i, entry in enumerate(picked):
        sig = entry.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        color = cmap(i % 20)

        for sid in seg_ids:
            sid = int(sid)
            seg = seg_by_id.get(sid)
            if not seg:
                continue
            pts = seg["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.5, alpha=0.85, color=color)

        ent = sig.get("entry", {}).get("point", None)
        ex = sig.get("exit", {}).get("point", None)
        if ent:
            ax.plot(ent["x"], ent["y"], marker="o", markersize=8, color=color)
        if ex:
            ax.plot(ex["x"], ex["y"], marker="s", markersize=8, color=color)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Visualization saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


# -------------------------
# Main
# -------------------------

def pick_paths_with_model(
    prompt: str,
    aggregated_json: str,
    out_picked_json: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.95,
    allow_fuzzy_match: bool = False,
    viz: bool = False,
    viz_out: str = "picked_paths_viz.png",
    viz_show: bool = False,
    model_id: Optional[str] = None,
):
    """
    Run the path picker using a provided model/tokenizer if given.
    Falls back to loading from model_id when not supplied (keeps CLI behavior).
    """
    if tokenizer is None or model is None:
        if not model_id:
            raise ValueError("model_id is required when model/tokenizer are not provided")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

    with open(aggregated_json, "r", encoding="utf-8") as f:
        agg = json.load(f)

    candidates = agg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("[ERROR] aggregated-json has no 'candidates' list.")

    def _generate_text(local_prompt: str, local_max_tokens: Optional[int] = None) -> str:
        import time
        effective_max = local_max_tokens if local_max_tokens is not None else max_new_tokens
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": local_prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if torch.cuda.is_available():
                input_ids = input_ids.to(model.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            input_len = int(input_ids.shape[-1])
        else:
            enc = tokenizer(local_prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_kwargs = enc
            input_len = int(enc["input_ids"].shape[-1])

        # Build generation kwargs; omit temperature/top_p when not sampling to avoid warnings
        gen_config = {
            "max_new_tokens": effective_max,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_config["temperature"] = temperature
            gen_config["top_p"] = top_p

        print(f"[DEBUG] path_picker LLM: prompt_tokens={input_len}, max_new={effective_max}", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**gen_kwargs, **gen_config)
        elapsed = time.time() - t0
        out_tokens = out.shape[-1] - input_len
        print(f"[DEBUG] path_picker LLM: done in {elapsed:.1f}s, output_tokens={out_tokens}", flush=True)

        gen_tokens = out[0][input_len:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # -------------------------
    # Stage A: constraint extraction (LLM) + deterministic CSP solve
    # -------------------------
    import time as time_module
    parsed: Optional[Dict[str, Any]] = None
    description = _extract_description_from_prompt(prompt)

    if description:
        t0_csp_stage = time_module.time()
        constraints_prompt = _build_constraints_prompt(description)
        constraints_text = _generate_text(constraints_prompt)
        print(constraints_text)
        print(f"[TIMING] path_picker constraint extraction LLM: {time_module.time() - t0_csp_stage:.2f}s", flush=True)

        t0_parse = time_module.time()
        constraints_obj = _safe_parse_json_object(constraints_text)
        if constraints_obj:
            # Debug: show extracted lane_change_phase
            for v in constraints_obj.get("vehicles", []):
                lcp = v.get("lane_change_phase")
                man = v.get("maneuver")
                if man == "lane_change":
                    print(f"[DEBUG] {v.get('vehicle')}: maneuver={man}, lane_change_phase={lcp}")
            
            # Deterministic guardrail: drop hallucinated constraints/evidence + dedupe
            constraints_obj = _sanitize_constraints_obj(constraints_obj, description)
            print(f"[TIMING] path_picker parse+sanitize: {time_module.time() - t0_parse:.2f}s", flush=True)

            try:
                t0_csp_solve = time_module.time()
                csp_items = _solve_paths_csp(constraints_obj, candidates, description=description)
                print(f"[TIMING] path_picker CSP solve: {time_module.time() - t0_csp_solve:.2f}s", flush=True)
                parsed = {"vehicles": csp_items}
            except Exception as e:
                print(f"[WARNING] CSP solve failed; falling back to direct path picking. Reason: {e}")

    # -------------------------
    # Stage B (fallback): direct path picking (legacy behavior)
    # -------------------------
    if parsed is None:
        text = _generate_text(prompt)
        print(text)

        parsed = _safe_parse_model_output(text)
        if not parsed or "vehicles" not in parsed or not isinstance(parsed["vehicles"], list):
            raise SystemExit("[ERROR] Could not parse model output as JSON with top-level 'vehicles' list.")

    with open(aggregated_json, "r", encoding="utf-8") as f:
        agg = json.load(f)

    candidates = agg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("[ERROR] aggregated-json has no 'candidates' list.")

    name_to_cand = {c.get("name"): c for c in candidates if isinstance(c, dict) and c.get("name")}
    candidate_names = list(name_to_cand.keys())

    picked: List[Dict[str, Any]] = []
    for item in parsed["vehicles"]:
        if not isinstance(item, dict):
            continue
        veh = item.get("vehicle", "Vehicle")
        req_name = item.get("path_name", "")

        if not req_name:
            m = re.search(r"path_\d{3}[^\s\"']*", json.dumps(item))
            req_name = m.group(0) if m else ""

        if not req_name:
            print(f"[WARNING] Missing path_name for {veh}; skipping.")
            continue

        cand = name_to_cand.get(req_name)
        if cand is None and allow_fuzzy_match:
            alt = _best_fuzzy_match(req_name, candidate_names)
            cand = name_to_cand.get(alt) if alt else None

        if cand is None:
            print(f"[WARNING] Requested path not found: '{req_name}' for {veh}; skipping.")
            continue

        out_entry = {
            "vehicle": veh,
            "name": cand.get("name"),
            "signature": cand.get("signature", {}),
        }
        if "confidence" in item:
            out_entry["confidence"] = item["confidence"]
        picked.append(out_entry)

    out_payload = {
        "source_candidates": aggregated_json,
        "nodes": agg.get("nodes"),
        "crop_region": agg.get("crop_region"),
        "parameters": agg.get("parameters"),
        "picked": picked,
    }

    with open(out_picked_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[INFO] Wrote {len(picked)} picked paths to: {out_picked_json}")

    if viz:
        nodes_path = agg.get("nodes")
        if not nodes_path:
            raise SystemExit("[ERROR] aggregated-json missing 'nodes' path; cannot visualize.")
        nodes = _load_nodes(nodes_path)
        all_segments = _build_segments_minimal(nodes)
        _plot_paths_together(
            all_segments=all_segments,
            picked=picked,
            crop=agg.get("crop_region"),
            out_path=viz_out,
            show=viz_show,
        )

    return out_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local model dir (or HF id)")
    ap.add_argument("--prompt-file", required=True, help="Text file containing your prompt")

    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--do-sample", action="store_true", help="Enable sampling")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)

    ap.add_argument("--aggregated-json", required=True, help="legal_paths_detailed.json")
    ap.add_argument("--out-picked-json", required=True, help="Output subset JSON")

    ap.add_argument("--allow-fuzzy-match", action="store_true",
                    help="Allow conservative fuzzy matching when exact name not found.")

    ap.add_argument("--viz", action="store_true", help="If set, generate a combined plot of picked paths.")
    ap.add_argument("--viz-out", type=str, default="picked_paths_viz.png", help="Output image file for viz.")
    ap.add_argument("--viz-show", action="store_true", help="If set, also show the plot window.")

    args = ap.parse_args()

    prompt = open(args.prompt_file, "r", encoding="utf-8").read().strip()

    pick_paths_with_model(
        prompt=prompt,
        aggregated_json=args.aggregated_json,
        out_picked_json=args.out_picked_json,
        model=None,
        tokenizer=None,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_fuzzy_match=args.allow_fuzzy_match,
        viz=args.viz,
        viz_out=args.viz_out,
        viz_show=args.viz_show,
    )


if __name__ == "__main__":
    main()
