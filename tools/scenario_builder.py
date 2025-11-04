#!/usr/bin/env python3
"""Interactive CARLA scenario builder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from string import Template
from typing import Iterable, List

import pandas as pd
try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


COLOR_PALETTE = [
    "#8e44ad",
    "#f39c12",
    "#2ecc71",
    "#d35400",
    "#1abc9c",
    "#f1c40f",
    "#27ae60",
    "#9b59b6",
    "#ff6f61",
    "#2c3e50",
]


def classify_lane_direction(wp: carla.Waypoint, carla_map: carla.Map) -> str:
    """Return 'forward', 'opposing', or 'neutral' for the provided waypoint."""
    lane_type = wp.lane_type
    if not lane_type & carla.LaneType.Driving:
        return "neutral"

    reference_wp = None
    try:
        reference_wp = carla_map.get_waypoint_xodr(wp.road_id, 0, wp.s)
    except RuntimeError:
        reference_wp = None

    if reference_wp is not None:
        yaw = math.radians(wp.transform.rotation.yaw)
        ref_yaw = math.radians(reference_wp.transform.rotation.yaw)
        dot = math.cos(yaw - ref_yaw)
        if dot > 1e-3:
            return "forward"
        if dot < -1e-3:
            return "opposing"

    return "forward" if wp.lane_id < 0 else "opposing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Sampling distance between reference waypoints in metres.",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help="Legacy flag for a single town (use --towns for multiple).",
    )
    parser.add_argument(
        "--towns",
        nargs="*",
        default=None,
        help="Optional list of towns (e.g., Town05 Town07). When omitted, all installed towns are shown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scenario_builder.html"),
        help="Path to the generated HTML file.",
    )
    return parser.parse_args()


def list_available_towns(client: carla.Client) -> List[str]:
    return sorted({path.split("/")[-1] for path in client.get_available_maps()})


def sample_town(world: carla.World, distance: float) -> pd.DataFrame:
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(distance=distance)
    rows = []
    for wp in waypoints:
        tf = wp.transform
        loc = tf.location
        rows.append(
            dict(
                x=loc.x,
                y=loc.y,
                z=loc.z,
                yaw=tf.rotation.yaw,
                lane_id=wp.lane_id,
                road_id=wp.road_id,
                section_id=wp.section_id,
                lane_direction=classify_lane_direction(wp, carla_map),
            )
        )
    return pd.DataFrame(rows)


def compute_payload(df: pd.DataFrame) -> dict[str, float | list[float]]:
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    pad_x = max((xmax - xmin) * 0.05, 10.0)
    pad_y = max((ymax - ymin) * 0.05, 10.0)
    direction_colors: list[str] = []
    for direction in df["lane_direction"].tolist():
        if direction == "forward":
            direction_colors.append("#3498db")
        elif direction == "opposing":
            direction_colors.append("#e74c3c")
        else:
            direction_colors.append("#95a5a6")
    return {
        "x": df["x"].round(3).tolist(),
        "y": df["y"].round(3).tolist(),
        "z": df["z"].round(3).tolist(),
        "yaw": df["yaw"].round(3).tolist(),
        "lane_id": df["lane_id"].tolist(),
        "road_id": df["road_id"].tolist(),
        "section_id": df["section_id"].tolist(),
        "lane_direction": df["lane_direction"].tolist(),
        "lane_colors": direction_colors,
        "xmin": xmin - pad_x,
        "xmax": xmax + pad_x,
        "ymin": ymin - pad_y,
        "ymax": ymax + pad_y,
    }


def generate_html(town_payloads: dict[str, dict[str, float | list[float]]],
                   distance: float,
                   colors: Iterable[str],
                   output_path: Path) -> None:
    if not town_payloads:
        raise ValueError("No town payloads provided.")

    colors_json = json.dumps(list(colors))
    towns_json = json.dumps(town_payloads)
    initial_town = next(iter(town_payloads.keys()))

    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>CARLA Scenario Builder</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            background-color: #111;
            color: #e5e5e5;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .left-panel {
            flex: 1 1 70%;
            position: relative;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .right-panel {
            flex: 1 1 30%;
            background-color: #1a1a1a;
            padding: 16px;
            overflow-y: auto;
            border-left: 1px solid #2b2b2b;
        }
        #map {
            width: 100%;
            height: 100%;
        }
        h2 {
            margin-top: 0;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 6px;
            margin-bottom: 12px;
            border: 1px solid #333;
            border-radius: 4px;
            background-color: #222;
            color: #eee;
        }
        button {
            background-color: #2979ff;
            border: none;
            color: white;
            padding: 8px 12px;
            text-align: center;
            font-size: 14px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        button.secondary {
            background-color: #555;
        }
        button.danger {
            background-color: #c0392b;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #333;
            padding: 6px;
            text-align: left;
        }
        th {
            background-color: #252525;
        }
        td input {
            width: 100%;
        }
        textarea.xml-editor {
            width: 100%;
            min-height: 140px;
            margin-top: 6px;
            background-color: #181818;
            color: #dcdcdc;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 6px;
            font-family: "Courier New", monospace;
        }
        details {
            margin-bottom: 14px;
            background-color: #202020;
            border-radius: 4px;
            padding: 6px 10px;
        }
        summary {
            cursor: pointer;
            font-weight: 600;
        }
        .info {
            margin-bottom: 12px;
            font-size: 13px;
            color: #bbb;
        }
        .town-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .town-tab {
            background-color: #2d2d2d;
            border: none;
            color: #d0d0d0;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .town-tab.active {
            background-color: #2979ff;
            color: #fff;
        }
        .status-msg {
            font-size: 12px;
            margin-top: 4px;
        }
        .status-error {
            color: #ff6b6b;
        }
        .status-success {
            color: #2ecc71;
        }
        .lane-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 6px 0 12px 0;
            font-size: 12px;
            color: #bbb;
        }
        .lane-legend .lane-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .lane-swatch {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            display: inline-block;
        }
        .lane-forward { background-color: #3498db; }
        .lane-opposite { background-color: #e74c3c; }
        .lane-neutral { background-color: #95a5a6; }
        .lane-arrow {
            font-size: 14px;
            font-weight: 600;
        }
        .lane-arrow.forward { color: #3498db; }
        .lane-arrow.opposite { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <div class="town-tabs" id="townTabs"></div>
            <div id="map"></div>
        </div>
        <div class="right-panel">
            <h2>Scenario Builder</h2>
            <p class="info">Active town: <strong id="activeTownLabel"></strong> | Sampling: ${distance} m</p>
            <p class="info">Switching towns resets all agents and waypoints.</p>
            <label for="scenarioName">Scenario / folder name</label>
            <input type="text" id="scenarioName" value="" />

            <label for="routeId">Route ID</label>
            <input type="text" id="routeId" value="247" />

            <label for="actorSelect">Active agent</label>
            <select id="actorSelect"></select>

            <div>
                <button id="addEgoBtn">Add ego</button>
                <button id="addNpcBtn">Add NPC vehicle</button>
                <button id="addPedBtn">Add pedestrian</button>
                <button id="addBikeBtn">Add bicycle</button>
                <button id="removeActorBtn" class="danger">Remove agent</button>
                <button id="clearActorBtn" class="secondary">Clear current path</button>
                <button id="resetAllBtn" class="secondary">Reset all</button>
            </div>

            <p class="info">
                Click to drop waypoints. After placing one, click again in the desired direction to set its heading.
            </p>
            <div class="lane-legend">
                <span class="lane-item">
                    <span class="lane-swatch lane-forward"></span>
                    <span>Forward lane <span class="lane-arrow forward">&#8594;</span></span>
                </span>
                <span class="lane-item">
                    <span class="lane-swatch lane-opposite"></span>
                    <span>Opposing lane <span class="lane-arrow opposite">&#8592;</span></span>
                </span>
                <span class="lane-item">
                    <span class="lane-swatch lane-neutral"></span>
                    <span>Neutral / shoulder</span>
                </span>
            </div>

            <table id="waypointTable">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>x [m]</th>
                        <th>y [m]</th>
                        <th>yaw [°]</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody id="waypointTableBody">
                    <tr><td colspan="5" style="text-align:center; padding:8px;">No waypoints yet.</td></tr>
                </tbody>
            </table>

            <h3>Generated XML</h3>
            <div id="xmlOutputs"></div>
            <button id="downloadAllBtn">Download all as ZIP</button>
        </div>
    </div>

    <script>
    const colorPalette = ${colors_json};
    const townsData = ${towns_json};
    const townNames = Object.keys(townsData);
    let activeTown = ${initial_town};
    if (!townNames.includes(activeTown) && townNames.length > 0) {
        activeTown = townNames[0];
    }

    let actors = [];
    let activeActorId = null;
    let actorIdCounter = 0;
    let egoCounter = 0;
    let npcCounter = 0;
    let pedestrianCounter = 0;
    let bicycleCounter = 0;
    let colorIndex = 0;
    let pendingHeading = null;
    let orientationPreview = null;
    let plotReady = false;

    function nextActorName(kind) {
        if (kind === 'ego') {
            const name = 'ego_vehicle_' + egoCounter;
            egoCounter += 1;
            return name;
        }
        if (kind === 'pedestrian') {
            const name = 'pedestrian_' + pedestrianCounter;
            pedestrianCounter += 1;
            return name;
        }
        if (kind === 'bicycle') {
            const name = 'bicycle_' + bicycleCounter;
            bicycleCounter += 1;
            return name;
        }
        const name = 'npc_vehicle_' + npcCounter;
        npcCounter += 1;
        return name;
    }

    function createActor(kind) {
        const color = colorPalette[colorIndex % colorPalette.length];
        colorIndex += 1;
        const actor = {
            id: actorIdCounter,
            kind: kind,
            name: nextActorName(kind),
            color: color,
            waypoints: []
        };
        actorIdCounter += 1;
        return actor;
    }

    function getActiveActor() {
        return actors.find(a => a.id === activeActorId) || null;
    }

    function setActiveActor(id) {
        resetOrientationState();
        activeActorId = id;
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    }

    const mapDiv = document.getElementById('map');
    const baseTrace = {
        x: [],
        y: [],
        mode: 'markers',
        marker: {
            size: 5,
            color: 'rgba(200, 200, 200, 0.45)',
            line: {width: 0}
        },
        customdata: [],
        hovertemplate:
            'x=%{x:.2f} m' +
            '<br>y=%{y:.2f} m' +
            '<br>z=%{customdata[0]:.2f} m' +
            '<br>yaw=%{customdata[1]:.1f}°' +
            '<br>lane=%{customdata[2]} (%{customdata[4]})' +
            '<br>road=%{customdata[3]}' +
            '<extra></extra>',
        showlegend: false
    };
    const layout = {
        paper_bgcolor: '#111',
        plot_bgcolor: '#111',
        xaxis: {
            gridcolor: '#222',
            zerolinecolor: '#333',
            title: 'x [m]',
            autorange: true
        },
        yaxis: {
            gridcolor: '#222',
            zerolinecolor: '#333',
            title: 'y [m]',
            scaleanchor: 'x',
            scaleratio: 1,
            autorange: true
        },
        dragmode: 'zoom',
        hovermode: 'closest',
        margin: {l: 60, r: 20, t: 30, b: 60}
    };
    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        scrollZoom: true
    };
    Plotly.newPlot(mapDiv, [baseTrace], layout, config).then(() => {
        plotReady = true;
    });

    mapDiv.addEventListener('mousemove', function(event) {
        if (!pendingHeading) return;
        const coords = screenToData(event);
        if (!coords) return;
        orientationPreview = coords;
        updateOrientationOverlay();
        updateOrientationPreviewMarker();
    });

    mapDiv.on('plotly_click', function(data) {
        const evt = data.event || {};
        if (evt.button && evt.button !== 0) return;
        if (!data.points || data.points.length === 0) return;
        const pt = data.points[0];
        if (typeof pt.x !== 'number' || typeof pt.y !== 'number') return;
        handlePlotClick(pt.x, pt.y);
    });

    function computeHeadingDegrees(origin, target) {
        return Math.atan2(target.y - origin.y, target.x - origin.x) * 180 / Math.PI;
    }

    function yawForDisplay(actor, index) {
        const wp = actor.waypoints[index];
        if (!wp) return 0;
        if (
            pendingHeading &&
            orientationPreview &&
            pendingHeading.actorId === actor.id &&
            pendingHeading.index === index
        ) {
            return computeHeadingDegrees(wp, orientationPreview);
        }
        return wp.yaw;
    }

    function getMarkerStyle(actor) {
        if (actor.kind === 'pedestrian') {
            return {symbol: 'circle', size: 14};
        }
        if (actor.kind === 'bicycle') {
            return {symbol: 'diamond', size: 16};
        }
        return {symbol: 'triangle-up', size: 20};
    }

    function updateOrientationPreviewMarker() {
        if (!plotReady) return;
        let traceIndex = 1; // baseTrace is index 0
        actors.forEach((actor) => {
            if (actor.waypoints.length === 0) return;
            const angles = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90);
            const yawValues = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx));
            Plotly.restyle(mapDiv, {
                'marker.angle': [angles],
                'customdata': [yawValues]
            }, traceIndex);
            traceIndex += 1;
        });
    }

    function screenToData(event) {
        const xaxis = mapDiv._fullLayout.xaxis;
        const yaxis = mapDiv._fullLayout.yaxis;
        if (!xaxis || !yaxis || !Plotly || !Plotly.Axes) return null;
        const rect = mapDiv.getBoundingClientRect();
        const xPixel = event.clientX - rect.left;
        const yPixel = event.clientY - rect.top;
        const xData = Plotly.Axes.p2c(xaxis, xPixel);
        const yData = Plotly.Axes.p2c(yaxis, yPixel);
        if (!isFinite(xData) || !isFinite(yData)) return null;
        return {x: xData, y: yData};
    }

    function resetOrientationState() {
        pendingHeading = null;
        orientationPreview = null;
        updateOrientationOverlay();
        updateOrientationPreviewMarker();
    }

    function updateOrientationOverlay() {
        if (!plotReady) return;
        const shapes = [];
        if (pendingHeading) {
            const actor = actors.find(a => a.id === pendingHeading.actorId);
            if (actor) {
                const wp = actor.waypoints[pendingHeading.index];
                if (wp) {
                    const radius = 4;
                    shapes.push({
                        type: 'circle',
                        xref: 'x',
                        yref: 'y',
                        x0: wp.x - radius,
                        x1: wp.x + radius,
                        y0: wp.y - radius,
                        y1: wp.y + radius,
                        line: {color: '#f1c40f', width: 2, dash: 'dot'}
                    });
                }
            }
        }
        Plotly.relayout(mapDiv, {shapes: shapes, annotations: []});
    }

    function handlePlotClick(x, y) {
        const actor = getActiveActor();
        if (!actor) return;

        if (pendingHeading && pendingHeading.actorId === actor.id) {
            const base = actor.waypoints[pendingHeading.index];
            if (base) {
                const yaw = Math.atan2(y - base.y, x - base.x) * 180 / Math.PI;
                if (isFinite(yaw)) {
                    base.yaw = yaw;
                }
            }
            pendingHeading = null;
            orientationPreview = null;
            updatePlot();
            renderWaypointTable();
            renderXmlOutputs();
            return;
        }

        const prev = actor.waypoints[actor.waypoints.length - 1];
        let yaw = prev ? prev.yaw : 0;
        actor.waypoints.push({x: x, y: y, z: 0.0, yaw: yaw});
        pendingHeading = {actorId: actor.id, index: actor.waypoints.length - 1};
        orientationPreview = null;
        updatePlot();
        renderWaypointTable();
        renderXmlOutputs();
    }

    function updatePlot() {
        const traces = [baseTrace];
        actors.forEach(actor => {
            if (actor.waypoints.length === 0) return;
            const orderLabels = actor.waypoints.map((_, idx) => String(idx + 1));
            const markerStyle = getMarkerStyle(actor);
            traces.push({
                x: actor.waypoints.map(w => w.x),
                y: actor.waypoints.map(w => w.y),
                customdata: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx)),
                text: orderLabels,
                textposition: 'middle center',
                textfont: {
                    color: '#111',
                    size: 12,
                    family: '"Segoe UI Semibold", "Segoe UI", Arial, sans-serif'
                },
                mode: 'lines+markers+text',
                name: actor.name + ' (' + actor.kind.toUpperCase() + ')',
                line: {
                    color: actor.color,
                    width: 3
                },
                marker: {
                    color: actor.color,
                    size: markerStyle.size,
                    symbol: markerStyle.symbol,
                    angle: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90),
                    line: {color: '#000', width: 0.5}
                },
                hovertemplate:
                    'Agent: ' + actor.name + ' (' + actor.kind.toUpperCase() + ')<br>' +
                    'x=%{x:.2f} m<br>' +
                    'y=%{y:.2f} m<br>' +
                    'yaw=%{customdata:.1f}°<extra></extra>'
            });
        });
        Plotly.react(mapDiv, traces, layout, config).then(() => {
            updateOrientationOverlay();
            updateOrientationPreviewMarker();
        });
    }
    function renderActorSelect() {
        const select = document.getElementById('actorSelect');
        select.innerHTML = '';
        actors.forEach(actor => {
            const opt = document.createElement('option');
            opt.value = actor.id;
            opt.textContent = actor.name + ' (' + actor.kind.toUpperCase() + ')';
            if (actor.id === activeActorId) opt.selected = true;
            select.appendChild(opt);
        });
        select.disabled = actors.length === 0;
        document.getElementById('removeActorBtn').disabled = actors.length <= 1;
        const active = getActiveActor();
        document.getElementById('clearActorBtn').disabled = !active || active.waypoints.length === 0;
    }

    function renderWaypointTable() {
        const tbody = document.getElementById('waypointTableBody');
        tbody.innerHTML = '';
        const actor = getActiveActor();
        if (!actor || actor.waypoints.length === 0) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 5;
            cell.style.textAlign = 'center';
            cell.style.padding = '8px';
            cell.textContent = 'No waypoints yet.';
            row.appendChild(cell);
            tbody.appendChild(row);
            document.getElementById('clearActorBtn').disabled = true;
            return;
        }

        actor.waypoints.forEach((wp, idx) => {
            const row = document.createElement('tr');

            const cellIdx = document.createElement('td');
            cellIdx.textContent = idx;
            row.appendChild(cellIdx);

            const cellX = document.createElement('td');
            cellX.textContent = wp.x.toFixed(3);
            row.appendChild(cellX);

            const cellY = document.createElement('td');
            cellY.textContent = wp.y.toFixed(3);
            row.appendChild(cellY);

            const cellYaw = document.createElement('td');
            const yawInput = document.createElement('input');
            yawInput.type = 'number';
            yawInput.step = '0.1';
            yawInput.value = wp.yaw.toFixed(2);
            yawInput.addEventListener('change', () => {
                const val = parseFloat(yawInput.value);
                wp.yaw = Number.isFinite(val) ? val : 0;
                updatePlot();
                renderXmlOutputs();
            });
            cellYaw.appendChild(yawInput);
            row.appendChild(cellYaw);

            const cellActions = document.createElement('td');
            const removeBtn = document.createElement('button');
            removeBtn.textContent = 'Delete';
            removeBtn.className = 'danger';
            removeBtn.addEventListener('click', () => {
                actor.waypoints.splice(idx, 1);
                resetOrientationState();
                renderWaypointTable();
                updatePlot();
                renderXmlOutputs();
            });
            cellActions.appendChild(removeBtn);
            row.appendChild(cellActions);

            tbody.appendChild(row);
        });
        document.getElementById('clearActorBtn').disabled = false;
    }

    function generateXml(actor) {
        const scenarioName = document.getElementById('scenarioName').value || 'custom_scenario';
        const routeId = document.getElementById('routeId').value || '0';
        const lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            '<routes>',
            '  <route id="' + routeId + '" town="' + activeTown + '" role="' + actor.kind + '">'
        ];
        actor.waypoints.forEach(wp => {
            lines.push(formatWaypoint(wp, '    '));
        });
        lines.push('  </route>');
        lines.push('</routes>');
        return lines.join('');
    }

    function formatWaypoint(wp, indent) {
        if (indent === undefined) indent = '      ';
        const yaw = wp.yaw.toFixed(6);
        const x = wp.x.toFixed(6);
        const y = wp.y.toFixed(6);
        const z = (wp.z || 0).toFixed(6);
        return indent + '<waypoint pitch="360.000000" roll="0.000000" x="' + x + '" y="' + y + '" yaw="' + yaw + '" z="' + z + '" />';
    }

    function parseXmlToWaypoints(xmlText) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(xmlText, 'application/xml');
        const errorNode = doc.getElementsByTagName('parsererror');
        if (errorNode.length) {
            throw new Error(errorNode[0].textContent || 'Invalid XML');
        }
        const nodes = Array.from(doc.getElementsByTagName('waypoint'));
        if (!nodes.length) {
            throw new Error('No <waypoint> elements found.');
        }
        return nodes.map(node => {
            const x = parseFloat(node.getAttribute('x'));
            const y = parseFloat(node.getAttribute('y'));
            const yaw = parseFloat(node.getAttribute('yaw') || '0');
            const z = parseFloat(node.getAttribute('z') || '0');
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(yaw)) {
                throw new Error('Waypoints must include numeric x, y, yaw.');
            }
            return { x: x, y: y, z: Number.isFinite(z) ? z : 0, yaw: yaw };
        });
    }

    function renderXmlOutputs() {
        const container = document.getElementById('xmlOutputs');
        container.innerHTML = '';
        const scenarioName = document.getElementById('scenarioName').value || 'custom_scenario';

        actors.forEach(actor => {
            const details = document.createElement('details');
            details.open = actors.length <= 1;
            const summary = document.createElement('summary');
            summary.textContent = actor.name + ' [' + actor.kind.toUpperCase() + '] (' + actor.waypoints.length + ' waypoints)';
            details.appendChild(summary);

            const textarea = document.createElement('textarea');
            textarea.className = 'xml-editor';
            let programmaticUpdate = true;
            textarea.value = generateXml(actor);
            programmaticUpdate = false;

            const status = document.createElement('div');
            status.className = 'status-msg';

            let debounceId = null;
            textarea.addEventListener('input', () => {
                if (programmaticUpdate) {
                    programmaticUpdate = false;
                    return;
                }
                clearTimeout(debounceId);
                status.textContent = 'Applying...';
                status.className = 'status-msg';
                debounceId = setTimeout(() => {
                    try {
                        const waypoints = parseXmlToWaypoints(textarea.value);
                        actor.waypoints = waypoints;
                        resetOrientationState();
                        renderWaypointTable();
                        updatePlot();
                        programmaticUpdate = true;
                        textarea.value = generateXml(actor);
                        programmaticUpdate = false;
                        status.textContent = 'Applied';
                        status.className = 'status-msg status-success';
                    } catch (err) {
                        status.textContent = 'Error: ' + (err.message || err);
                        status.className = 'status-msg status-error';
                    }
                }, 400);
            });

            const buttonsRow = document.createElement('div');
            const downloadBtn = document.createElement('button');
            downloadBtn.textContent = 'Download XML';
            downloadBtn.addEventListener('click', () => {
                const blob = new Blob([textarea.value], {type: 'application/xml'});
                const fileName = scenarioName + '_' + actor.name + '.xml';
                saveAs(blob, fileName);
            });
            buttonsRow.appendChild(downloadBtn);

            details.appendChild(textarea);
            details.appendChild(buttonsRow);
            details.appendChild(status);
            container.appendChild(details);
        });
    }

    function renderTownTabs() {
        const tabs = document.getElementById('townTabs');
        tabs.innerHTML = '';
        townNames.forEach(name => {
            const btn = document.createElement('button');
            btn.textContent = name;
            btn.className = 'town-tab' + (name === activeTown ? ' active' : '');
            btn.addEventListener('click', () => loadTown(name));
            tabs.appendChild(btn);
        });
        document.getElementById('activeTownLabel').textContent = activeTown;
    }

    function loadTown(name) {
        activeTown = name;
        const data = townsData[name];
        baseTrace.x = data.x;
        baseTrace.y = data.y;
        baseTrace.marker.color = data.lane_colors || baseTrace.marker.color;
        baseTrace.customdata = data.z.map((_, idx) => [
            data.z[idx],
            data.yaw[idx],
            data.lane_id[idx],
            data.road_id[idx],
            data.lane_direction ? data.lane_direction[idx] : ''
        ]);
        layout.xaxis.range = [data.xmin, data.xmax];
        layout.yaxis.range = [data.ymin, data.ymax];
        layout.xaxis.autorange = false;
        layout.yaxis.autorange = false;

        actors = [];
        actorIdCounter = 0;
        egoCounter = 0;
        npcCounter = 0;
        pedestrianCounter = 0;
        bicycleCounter = 0;
        colorIndex = 0;
        const actor = createActor('ego');
        actors.push(actor);
        activeActorId = actor.id;

        document.getElementById('scenarioName').value = name.toLowerCase() + '_custom';
        document.getElementById('routeId').value = '247';

        pendingHeading = null;
        orientationPreview = null;

        renderTownTabs();
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    }

    function init() {
        renderTownTabs();
        if (townNames.length > 0) {
            loadTown(activeTown);
        } else {
            document.getElementById('activeTownLabel').textContent = 'None';
        }
    }

    document.getElementById('actorSelect').addEventListener('change', (evt) => {
        setActiveActor(parseInt(evt.target.value));
    });

    document.getElementById('addEgoBtn').addEventListener('click', () => {
        const actor = createActor('ego');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addNpcBtn').addEventListener('click', () => {
        const actor = createActor('npc');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addPedBtn').addEventListener('click', () => {
        const actor = createActor('pedestrian');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addBikeBtn').addEventListener('click', () => {
        const actor = createActor('bicycle');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('removeActorBtn').addEventListener('click', () => {
        if (actors.length <= 1) return;
        resetOrientationState();
        const actor = getActiveActor();
        const idx = actors.findIndex(a => a.id === actor.id);
        actors.splice(idx, 1);
        const next = actors[Math.max(0, idx - 1)];
        setActiveActor(next.id);
    });

    document.getElementById('clearActorBtn').addEventListener('click', () => {
        const actor = getActiveActor();
        if (!actor) return;
        actor.waypoints = [];
        resetOrientationState();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    });

    document.getElementById('resetAllBtn').addEventListener('click', () => {
        if (!confirm('Clear all agents and waypoints?')) return;
        if (townNames.length === 0) return;
        actors = [];
        actorIdCounter = 0;
        egoCounter = 0;
        npcCounter = 0;
        pedestrianCounter = 0;
        bicycleCounter = 0;
        colorIndex = 0;
        const actor = createActor('ego');
        actors.push(actor);
        activeActorId = actor.id;
        resetOrientationState();
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    });

    document.getElementById('scenarioName').addEventListener('input', renderXmlOutputs);
    document.getElementById('routeId').addEventListener('input', renderXmlOutputs);

    document.getElementById('downloadAllBtn').addEventListener('click', () => {
        if (actors.length === 0) {
            alert('No agents to export.');
            return;
        }
        const scenarioName = document.getElementById('scenarioName').value || 'custom_scenario';
        const zip = new JSZip();

        actors.forEach(actor => {
            const xmlText = generateXml(actor);
            const fileName = scenarioName + '_' + actor.name + '.xml';
            zip.file(fileName, xmlText);
        });

        zip.generateAsync({type:'blob'}).then(content => {
            saveAs(content, scenarioName + '_routes.zip');
        });
    });

    init();
    </script>
</body>
</html>
""")

    html = template.substitute(
        colors_json=colors_json,
        towns_json=towns_json,
        initial_town=json.dumps(initial_town),
        distance=f"{distance:.2f}",
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved scenario builder to {output_path}")


def main() -> None:
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    available = list_available_towns(client)
    if not available:
        raise RuntimeError("No CARLA maps found on the server.")

    if args.towns:
        requested = args.towns
    elif args.town:
        requested = [args.town]
    else:
        requested = available

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested town(s) not available: {', '.join(missing)}")

    print("Available CARLA towns:", ", ".join(available))
    print("Building scenario builder for:", ", ".join(requested))

    payloads: dict[str, dict[str, float | list[float]]] = {}
    for town in requested:
        print(f"Sampling {town} with spacing {args.distance:.2f} m ...")
        world = client.load_world(town)
        df = sample_town(world, args.distance)
        print(f"  collected {len(df)} reference points.")
        payloads[town] = compute_payload(df)

    generate_html(payloads, args.distance, COLOR_PALETTE, args.output)


if __name__ == "__main__":
    main()
