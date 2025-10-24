import argparse
import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_TIME_PATTERN = re.compile(r"step infer time:\s*([0-9.]+)")
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_PADDING = 14
VEHICLE_DIR_PATTERN = re.compile(r"ego_vehicle_(\d+)$")


@dataclass
class VehicleStream:
    index: int
    label: str
    image_paths: List[Path]


@dataclass
class NegotiationOverlay:
    id: int
    frame_index: int
    hold_frames: int
    messages_by_vehicle: Dict[int, List[str]]
    global_messages: List[str]
    color: Tuple[int, int, int]


def list_vehicle_streams(root: Path) -> List[VehicleStream]:
    """Discover vehicle-specific image folders within the given root."""
    streams: List[VehicleStream] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = re.search(r"(\d+)$", entry.name)
        if not match:
            continue
        index = int(match.group(1))
        image_paths = sorted(
            (entry / fname for fname in os.listdir(entry)),
            key=lambda p: p.stem,
        )
        image_paths = [p for p in image_paths if p.suffix.lower() in FRAME_EXTENSIONS]
        if not image_paths:
            continue
        streams.append(
            VehicleStream(index=index, label=f"Vehicle {index}", image_paths=image_paths)
        )

    if not streams:
        raise FileNotFoundError(
            f"No vehicle folders with images were found in {root}. "
            "Expected folders like ego_vehicle_0, ego_vehicle_1, etc."
        )

    streams.sort(key=lambda s: s.index)
    return streams


def _scenario_has_vehicle_images(scenario_dir: Path) -> bool:
    """Return True if scenario_dir contains at least one ego_vehicle_* folder with images."""
    if not scenario_dir.is_dir():
        return False

    for child in scenario_dir.iterdir():
        if not (child.is_dir() and VEHICLE_DIR_PATTERN.match(child.name)):
            continue
        for item in child.iterdir():
            if item.is_file() and item.suffix.lower() in FRAME_EXTENSIONS:
                return True
    return False


def discover_scenario_dirs(root: Path) -> List[Path]:
    """Recursively find scenario directories underneath root."""
    scenarios: List[Path] = []
    seen: set[Path] = set()
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if candidate in seen:
            continue
        if _scenario_has_vehicle_images(candidate):
            scenarios.append(candidate)
            seen.add(candidate)
    scenarios.sort()
    return scenarios


def remove_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def chunk_word(word: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
    """Split an oversized word into chunks that fit within max_width."""
    chunks: List[str] = []
    current = ""
    for char in word:
        tentative = current + char
        width = cv2.getTextSize(tentative, font, font_scale, thickness)[0][0]
        if width <= max_width or not current:
            current = tentative
        else:
            chunks.append(current)
            current = char
    if current:
        chunks.append(current)
    return chunks


def wrap_line(
    line: str, font, font_scale: float, thickness: int, max_width: int
) -> List[str]:
    """Wrap a line of text to fit within max_width."""
    words = line.split()
    if not words:
        return [""]

    wrapped: List[str] = []
    current = ""
    for word in words:
        tentative = word if not current else f"{current} {word}"
        width = cv2.getTextSize(tentative, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = tentative
            continue

        if current:
            wrapped.append(current)
            current = ""

        word_width = cv2.getTextSize(word, font, font_scale, thickness)[0][0]
        if word_width <= max_width:
            current = word
            continue

        chunks = chunk_word(word, font, font_scale, thickness, max_width)
        if chunks:
            wrapped.extend(chunks[:-1])
            current = chunks[-1]

    if current:
        wrapped.append(current)
    return wrapped


def discover_log_path(root: Path) -> Optional[Path]:
    """Return the first .log file found within the directory."""
    nested = root / "log"
    if nested.is_dir():
        candidate = nested / "log.log"
        if candidate.exists():
            return candidate

    for path in root.rglob("*.log"):
        return path
    return None


def _stable_color_for_event(event_id: int) -> Tuple[int, int, int]:
    """Pick a readable BGR color that is stable per event."""
    rng = np.random.default_rng(event_id + 2024)
    # Sample until the luminance is within a readable range.
    for _ in range(8):
        color = rng.integers(60, 230, size=3, dtype=np.int32)
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if 80 <= luminance <= 220:
            return (b, g, r)
    color = rng.integers(90, 210, size=3, dtype=np.int32)
    return (int(color[0]), int(color[1]), int(color[2]))


def parse_negotiation_events(
    log_path: Optional[Path],
    fps: float,
    min_hold_seconds: float,
    words_per_minute: float,
    reading_factor: float,
) -> List[NegotiationOverlay]:
    if not log_path or not log_path.exists():
        return []

    overlays: List[NegotiationOverlay] = []
    last_step_time = 0.0
    capturing = False
    negotiation_time = 0.0
    event_counter = 0

    current_actor: Optional[int] = None
    current_messages: List[Tuple[Optional[int], str]] = []
    current_action_list: Optional[str] = None
    current_short_analysis: Optional[str] = None

    words_per_second = max(1e-3, words_per_minute / 60.0)
    reading_factor = max(0.05, reading_factor)

    def finalize_event() -> None:
        nonlocal event_counter
        if (
            not current_messages
            and not current_action_list
            and not current_short_analysis
        ):
            return

        messages_by_vehicle: Dict[int, List[str]] = defaultdict(list)
        global_lines: List[str] = []

        turn_index = 0
        for car_id, raw_text in current_messages:
            clean_text = (raw_text or "").strip()
            turn_index += 1
            vehicle_prefix = f"Vehicle {car_id}: " if car_id is not None else ""
            numbered_text = f"{turn_index}. {vehicle_prefix}{clean_text if clean_text else '(no message)'}"
            if car_id is None:
                global_lines.append(numbered_text)
            else:
                messages_by_vehicle[car_id].append(numbered_text)

        if current_action_list:
            global_lines.append(f"Action List: {current_action_list}")

        if current_short_analysis:
            global_lines.append(f"Short Analysis: {current_short_analysis}")

        if not messages_by_vehicle and not global_lines:
            return

        word_count = 0
        for lines in messages_by_vehicle.values():
            word_count += sum(len(line.split()) for line in lines if line.strip())
        word_count += sum(len(line.split()) for line in global_lines if line.strip())

        hold_seconds = max(min_hold_seconds, (word_count / words_per_second) * reading_factor)
        hold_frames = max(1, int(round(hold_seconds * fps)))
        frame_index = max(0, int(round(negotiation_time * fps)))
        color = _stable_color_for_event(event_counter)

        overlays.append(
            NegotiationOverlay(
                id=event_counter,
                frame_index=frame_index,
                hold_frames=hold_frames,
                messages_by_vehicle={k: v for k, v in sorted(messages_by_vehicle.items())},
                global_messages=global_lines[:],
                color=color,
            )
        )
        event_counter += 1

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            clean_line = remove_ansi(raw_line.rstrip("\n"))
            step_match = STEP_TIME_PATTERN.search(clean_line)
            if step_match:
                try:
                    last_step_time = float(step_match.group(1))
                except ValueError:
                    pass

            if "Conflict exist, start negotiation" in clean_line:
                capturing = True
                negotiation_time = last_step_time
                current_actor = None
                current_messages.clear()
                current_action_list = None
                current_short_analysis = None
                continue

            if not capturing:
                continue

            car_match = re.search(r"car_id:\s*(\d+)", clean_line)
            if car_match:
                try:
                    current_actor = int(car_match.group(1))
                except ValueError:
                    current_actor = None
                continue

            stripped = clean_line.strip()
            if stripped.startswith("message:"):
                message_text = stripped.split("message:", 1)[1].strip()
                message_text = message_text.strip('" ')
                current_messages.append((current_actor, message_text))
                continue

            if stripped.startswith("action_list:"):
                raw_value = stripped.split("action_list:", 1)[1].strip()
                formatted = raw_value
                try:
                    parsed = ast.literal_eval(raw_value)
                    if isinstance(parsed, dict):
                        pieces: List[str] = []
                        for key in sorted(
                            parsed.keys(),
                            key=lambda x: int(x) if str(x).isdigit() else str(x),
                        ):
                            detail = parsed[key]
                            if isinstance(detail, dict):
                                nav = detail.get("nav")
                                speed = detail.get("speed")
                                parts = [part for part in (nav, speed) if part]
                                desc = " / ".join(parts) if parts else str(detail)
                            else:
                                desc = str(detail)
                            pieces.append(f"Vehicle {key}: {desc}")
                        if pieces:
                            formatted = "; ".join(pieces)
                except Exception:
                    formatted = raw_value
                current_action_list = formatted
                continue

            if stripped.startswith("Short analysis"):
                summary = stripped.split("Short analysis:", 1)[1].strip()
                current_short_analysis = summary
                continue

            lowered = stripped.lower()
            if (
                "comm_results" in lowered
                or "negotiation done" in lowered
                or lowered.startswith("vlm using negotiation results")
            ):
                finalize_event()
                capturing = False

    return overlays


def add_label(image: np.ndarray, label: str) -> Tuple[np.ndarray, int]:
    """Overlay a larger label in the top-left corner and return the bottom of the label box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    padding = LABEL_PADDING
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    box_pt1 = (padding - 6, padding - 10)
    box_pt2 = (padding + text_size[0] + 6, padding + text_size[1])

    overlay = image.copy()
    cv2.rectangle(overlay, box_pt1, box_pt2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    text_origin = (padding, padding + text_size[1] - 4)
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    label_bottom = text_origin[1] + 6
    return image, label_bottom


def render_negotiation_slot(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    lines: List[str],
    color: Tuple[int, int, int],
) -> None:
    """Render a negotiation box anchored within the negotiation column."""
    base_color = (30, 30, 30)
    border_thickness = max(2, int(round(height * 0.01)))
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), base_color, -1)
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 0, 0),
        border_thickness,
    )

    cleaned = [line.strip() for line in lines if line.strip()]
    if not cleaned:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_ratio = 0.05
    margin_x = max(6, int(round(width * margin_ratio)))
    margin_y = max(6, int(round(height * margin_ratio)))
    available_width = max(20, width - 2 * margin_x)
    available_height = max(20, height - 2 * margin_y)

    min_scale = 0.1
    max_scale = 6.0

    def layout(scale: float) -> Tuple[List[str], int, int, int, int, int]:
        """Return (wrapped_lines, thickness, line_height, spacing, max_width, total_height)."""
        thickness = max(1, int(round(max(scale * 0.85, 1.0))))
        baseline = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
        spacing = max(4, int(round(baseline * 0.3)))
        wrapped: List[str] = []
        for raw_line in cleaned:
            segments = wrap_line(raw_line, font, scale, thickness, available_width)
            if not segments:
                continue
            wrapped.extend(segments)

        if not wrapped:
            wrapped = [cleaned[0]]

        max_width = 0
        for segment in wrapped:
            max_width = max(max_width, cv2.getTextSize(segment, font, scale, thickness)[0][0])

        total_height = len(wrapped) * baseline
        if len(wrapped) > 1:
            total_height += (len(wrapped) - 1) * spacing
        return wrapped, thickness, baseline, spacing, max_width, total_height

    def fits(scale: float) -> Tuple[bool, Tuple[List[str], int, int, int, int, int]]:
        layout_result = layout(scale)
        max_width, total_height = layout_result[4], layout_result[5]
        return (max_width <= available_width and total_height <= available_height), layout_result

    scale = min_scale
    layout_result = layout(scale)
    attempts = 0
    while (layout_result[4] > available_width or layout_result[5] > available_height) and attempts < 8:
        scale *= 0.7
        layout_result = layout(scale)
        attempts += 1

    best_scale = scale
    best_layout = layout_result

    high = max(max_scale, best_scale * 1.5)
    fits_high, layout_high = fits(high)
    while fits_high and high < 24.0:
        best_scale = high
        best_layout = layout_high
        high *= 1.5
        fits_high, layout_high = fits(high)

    low = best_scale
    if high < low:
        high = low

    for _ in range(30):
        if high - low <= 1e-3:
            break
        mid = (low + high) / 2
        fits_mid, layout_mid = fits(mid)
        if fits_mid:
            best_scale = mid
            best_layout = layout_mid
            low = mid
        else:
            high = mid

    for _ in range(5):
        trial = best_scale * 1.08
        fits_trial, layout_trial = fits(trial)
        if fits_trial:
            best_scale = trial
            best_layout = layout_trial
        else:
            break

    wrapped_lines, thickness, line_height, spacing, _, total_height = best_layout

    b, g, r = color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance < 185:
        blend = 0.55
        r = int(r + (255 - r) * blend)
        g = int(g + (255 - g) * blend)
        b = int(b + (255 - b) * blend)
        color = (b, g, r)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)

    text_color = (0, 0, 0)

    used_height = total_height
    vertical_extra = max(0, available_height - used_height)
    cursor_y = y + margin_y + line_height + vertical_extra // 2

    for line in wrapped_lines:
        cursor_x = x + margin_x
        cv2.putText(
            canvas,
            line,
            (cursor_x, cursor_y),
            font,
            best_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        bold_prefix: Optional[str] = None
        number_match = re.match(r"(\d+\.\s)", line)
        if number_match:
            bold_prefix = number_match.group(1)
        else:
            lowered_line = line.lower()
            if lowered_line.startswith("action list:"):
                bold_prefix = line[: len("Action List:")]
            elif lowered_line.startswith("short analysis:"):
                bold_prefix = line[: len("Short Analysis:")]

        if bold_prefix:
            bold_thickness = max(thickness + 1, thickness * 2)
            cv2.putText(
                canvas,
                bold_prefix,
                (cursor_x, cursor_y),
                font,
                best_scale,
                text_color,
                bold_thickness,
                cv2.LINE_AA,
            )

        cursor_y += line_height + spacing


def build_video(
    output_path: Path,
    streams: List[VehicleStream],
    fps: float,
    resize_factor: int,
    negotiation_events: List[NegotiationOverlay],
) -> None:
    first_frame = cv2.imread(str(streams[0].image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Unable to read {streams[0].image_paths[0]}")

    base_height, base_width = first_frame.shape[:2]
    target_size = (max(1, base_width // resize_factor), max(1, base_height // resize_factor))
    combined_height = target_size[1] * len(streams)
    column_width = max(220, int(round(target_size[0] * 0.4)))
    total_width = target_size[0] + column_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (total_width, combined_height)
    )

    max_frames = max(len(stream.image_paths) for stream in streams)

    event_map: Dict[int, List[NegotiationOverlay]] = defaultdict(list)
    for event in negotiation_events:
        event_map[event.frame_index].append(event)

    active_events: List[Dict[str, object]] = []

    for frame_idx in tqdm(range(max_frames), desc="Compositing frames"):
        starting_events = event_map.get(frame_idx, [])
        for event in starting_events:
            active_events.append({"event": event, "remaining": event.hold_frames})

        frames: List[np.ndarray] = []
        stream_overlays: List[List[Tuple[int, Tuple[int, int, int], List[str]]]] = []
        for stream in streams:
            image_index = min(frame_idx, len(stream.image_paths) - 1)
            image = cv2.imread(str(stream.image_paths[image_index]))
            if image is None:
                raise RuntimeError(f"Unable to read {stream.image_paths[image_index]}")
            image = cv2.resize(image, target_size)
            image, _ = add_label(image, stream.label)

            overlays = []
            for active in active_events:
                event = active["event"]  # type: ignore[assignment]
                lines: List[str] = []
                vehicle_lines = event.messages_by_vehicle.get(stream.index, [])
                if vehicle_lines:
                    lines.extend(vehicle_lines)
                if event.global_messages:
                    lines.extend(event.global_messages)
                if lines:
                    overlays.append((event.id, event.color, lines))

            overlays.sort(key=lambda item: item[0])
            if overlays:
                overlays = [overlays[-1]]

            frames.append(image)
            stream_overlays.append(overlays)

        combined = frames[0] if len(frames) == 1 else cv2.vconcat(frames)
        column = np.zeros((combined_height, column_width, 3), dtype=np.uint8)
        column[:] = (24, 24, 24)

        for stream_idx, overlays in enumerate(stream_overlays):
            top = int(round(stream_idx * target_size[1]))
            bottom = int(round(top + target_size[1]))
            slot_height = max(1, bottom - top)
            if slot_height <= 0:
                continue

            event_data: Optional[Tuple[int, Tuple[int, int, int], List[str]]] = (
                overlays[0] if overlays else None
            )

            if event_data is None:
                border_thickness = max(2, int(round(slot_height * 0.01)))
                cv2.rectangle(
                    column,
                    (0, top),
                    (column_width - 1, top + slot_height - 1),
                    (45, 45, 45),
                    -1,
                )
                cv2.rectangle(
                    column,
                    (0, top),
                    (column_width - 1, top + slot_height - 1),
                    (0, 0, 0),
                    border_thickness,
                )
                continue

            _, color, slot_lines = event_data
            render_negotiation_slot(
                column,
                0,
                top,
                column_width,
                slot_height,
                slot_lines,
                color,
            )

        final_frame = cv2.hconcat([column, combined])
        writer.write(final_frame)

        for active in list(active_events):
            active["remaining"] = int(active["remaining"]) - 1  # type: ignore[index]
            if active["remaining"] <= 0:
                active_events.remove(active)

    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a composite video from vehicle image folders, annotating each feed "
            "and presenting negotiation transcripts in a sidebar aligned with the speaking vehicle."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Path to the scenario directory containing ego_vehicle_* (or similar) folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path. Defaults to <input_dir>/output_v3.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where generated videos are saved when processing multiple scenarios. "
            "Defaults to <input_dir>/videos when batching."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Output video frames per second (default: 5.0).",
    )
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=2,
        help="Factor by which to downscale each feed before stacking (default: 2).",
    )
    parser.add_argument(
        "--hold-seconds",
        "--min-hold-seconds",
        dest="min_hold_seconds",
        type=float,
        default=1.2,
        help="Minimum duration (in seconds) to display negotiation transcripts (default: 1.2).",
    )
    parser.add_argument(
        "--words-per-minute",
        type=float,
        default=240.0,
        help="Reading speed used to time transcript overlays (default: 240 WPM).",
    )
    parser.add_argument(
        "--reading-factor",
        type=float,
        default=0.5,
        help="Multiplier applied to the computed display time (default: 0.5).",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional path to a log file. If omitted, the script searches within the input directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory {root} does not exist.")

    is_single_scenario = _scenario_has_vehicle_images(root)
    if is_single_scenario:
        scenario_dirs = [root]
    else:
        scenario_dirs = discover_scenario_dirs(root)
        if not scenario_dirs:
            raise FileNotFoundError(
                f"No scenario directories with ego_vehicle_* images were found under {root}."
            )

    if len(scenario_dirs) > 1 and args.output:
        raise ValueError(
            "The --output argument is only supported when a single scenario directory is processed. "
            "Use --output-dir to specify the destination for batch generation."
        )

    if len(scenario_dirs) > 1 and args.log_path:
        raise ValueError(
            "A custom --log-path cannot be combined with batch processing. "
            "Logs are discovered per scenario automatically."
        )

    if args.output_dir:
        batch_output_dir = Path(args.output_dir).expanduser().resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if len(scenario_dirs) > 1:
            batch_output_dir = (root / "videos").resolve()
            batch_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            batch_output_dir = None

    print(f"Discovered {len(scenario_dirs)} scenario directory(ies) to process.")

    for scenario_dir in scenario_dirs:
        if len(scenario_dirs) > 1:
            print(f"\nProcessing scenario: {scenario_dir}")

        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        elif batch_output_dir:
            output_path = batch_output_dir / f"{scenario_dir.name}.mp4"
        else:
            output_path = scenario_dir / "output_v3.mp4"

        streams = list_vehicle_streams(scenario_dir)

        log_path = (
            Path(args.log_path).expanduser().resolve()
            if args.log_path
            else discover_log_path(scenario_dir)
        )
        if log_path and not log_path.exists():
            log_path = None

        negotiation_events = parse_negotiation_events(
            log_path,
            args.fps,
            args.min_hold_seconds,
            args.words_per_minute,
            args.reading_factor,
        )

        if not negotiation_events:
            print("No negotiation events found; proceeding without transcript overlays.")
        else:
            print(f"Configured {len(negotiation_events)} negotiation overlay(s).")

        build_video(
            output_path=output_path,
            streams=streams,
            fps=args.fps,
            resize_factor=args.resize_factor,
            negotiation_events=negotiation_events,
        )
        print(f"Wrote video to {output_path}")


if __name__ == "__main__":
    main()
