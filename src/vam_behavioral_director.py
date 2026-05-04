"""VaM Behavioral AI Director.

Production-ready behavioral synthesis pipeline for Virt-a-Mate Timeline JSON.
Implements:
1) Offline behavior model extraction
2) Markov state brain
3) Biomechanical layered synthesis
4) customtkinter GUI with async Generate & Export
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation, Slerp

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox

    GUI_AVAILABLE = True
except Exception:  # pragma: no cover
    ctk = None
    filedialog = None
    messagebox = None
    GUI_AVAILABLE = False

from motion_analyzer import MotionResampler, TimelineParser
from motion_synthesizer import BehavioralSynthesizer
from body_awareness_engine import BodyAwareSynthesizer, BodyAwarenessScanner


LOGGER = logging.getLogger("vam_behavioral_director")
AXES = ("x", "y", "z")
CURVE_TYPE_SMOOTH_LOCAL = 3
INTERNAL_FPS = 60.0
EXPORT_FPS = 30.0

KINEMATIC_CONTROLLERS = (
    "hipControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
    "lThighControl",
    "rThighControl",
)
STATE_NAMES = (
    "Warmup_Teasing",
    "Steady_Riding",
    "Intense_LeaningBack",
    "Exhausted_LeaningFwd",
)

DEFAULT_MOCAP_DIR = Path(r"G:\VAM_Fresh\SavedMocaps")
DEFAULT_MODEL_PATH = Path(r"G:\VAM_Fresh\behavior_model.json")
DEFAULT_BODYAWARE_MODEL_PATH = Path(r"G:\VAM_Fresh\body_awareness_model.json")
DEFAULT_PRESET_DIR = Path(r"G:\VAM_Fresh\Custom")

LOCKED_EXTREMITIES = (
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
    "lThighControl",
    "rThighControl",
)
FALLBACK_STATIC_LOCKS: dict[str, dict[str, float]] = {
    "lThighControl": {"x": -0.18, "y": 0.90, "z": 0.10},
    "rThighControl": {"x": 0.18, "y": 0.90, "z": 0.10},
    "lKneeControl": {"x": -0.30, "y": 0.60, "z": 0.30},
    "rKneeControl": {"x": 0.30, "y": 0.60, "z": 0.30},
    "lFootControl": {"x": -0.32, "y": 0.20, "z": 0.44},
    "rFootControl": {"x": 0.32, "y": 0.20, "z": 0.44},
}
FALLBACK_ANCHOR_POSE: dict[str, dict[str, Any]] = {
    "hipControl": {"x": 0.0, "y": 1.00, "z": 0.0},
    "chestControl": {"x": 0.0, "y": 1.25, "z": 0.03},
    "headControl": {"x": 0.0, "y": 1.55, "z": 0.05},
    "lHandControl": {"x": -0.28, "y": 1.08, "z": 0.10},
    "rHandControl": {"x": 0.28, "y": 1.08, "z": 0.10},
    **FALLBACK_STATIC_LOCKS,
}
for _ctrl in FALLBACK_ANCHOR_POSE:
    FALLBACK_ANCHOR_POSE[_ctrl]["quat"] = [0.0, 0.0, 0.0, 1.0]


@dataclass(frozen=True)
class DirectorParameters:
    context: str
    duration: float
    base_intensity: float
    playfulness: float
    stamina: float
    seed: int = 42
    target_anchor: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class StateSegment:
    state: str
    start: float
    end: float
    intensity: float


@dataclass(frozen=True)
class SecondaryEvent:
    kind: str
    controller: str
    start: float
    end: float
    target: dict[str, float]
    rot_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


def finite_float(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm <= 1e-12, 1.0, norm)
    return q / norm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication in scipy order x,y,z,w."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def parse_vam_pose_anchor(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a VaM pose-like JSON and extract controller anchors."""
    with path.open("r", encoding="utf-8-sig") as h:
        data = json.load(h)

    out: dict[str, dict[str, Any]] = {
        c: {
            "x": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("x", 0.0)),
            "y": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("y", 1.0)),
            "z": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("z", 0.0)),
            "quat": list(FALLBACK_ANCHOR_POSE.get(c, {}).get("quat", [0.0, 0.0, 0.0, 1.0])),
        }
        for c in KINEMATIC_CONTROLLERS
    }

    def try_ingest(obj: dict[str, Any]) -> None:
        cid = str(obj.get("id") or obj.get("Controller") or "")
        if cid not in out:
            return
        lp = obj.get("localPosition") or obj.get("position")
        lr = obj.get("localRotation") or obj.get("rotation")
        if isinstance(lp, dict):
            out[cid]["x"] = finite_float(lp.get("x"), out[cid]["x"])
            out[cid]["y"] = finite_float(lp.get("y"), out[cid]["y"])
            out[cid]["z"] = finite_float(lp.get("z"), out[cid]["z"])
        if isinstance(lr, dict):
            if "w" in lr:
                q = np.array(
                    [
                        finite_float(lr.get("x"), 0.0),
                        finite_float(lr.get("y"), 0.0),
                        finite_float(lr.get("z"), 0.0),
                        finite_float(lr.get("w"), 1.0),
                    ],
                    dtype=float,
                )
                out[cid]["quat"] = normalize_quat(q).tolist()
            else:
                euler = np.array(
                    [
                        finite_float(lr.get("x"), 0.0),
                        finite_float(lr.get("y"), 0.0),
                        finite_float(lr.get("z"), 0.0),
                    ],
                    dtype=float,
                )
                out[cid]["quat"] = Rotation.from_euler("xyz", euler, degrees=True).as_quat().tolist()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            try_ingest(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return out


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True-runs [start, end) from a boolean mask."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(mask.astype(np.int8), prepend=0, append=0))
    if len(idx) < 2:
        return []
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def _resample_xyz(segment: np.ndarray, out_len: int) -> np.ndarray:
    """Resample Nx3 segment to out_len x 3 with cubic splines fallback to linear."""
    n = segment.shape[0]
    if n <= 1:
        return np.repeat(segment[:1], out_len, axis=0)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, out_len)
    out = np.zeros((out_len, 3), dtype=float)
    for a in range(3):
        if n >= 4:
            out[:, a] = CubicSpline(x_old, segment[:, a], bc_type="natural")(x_new)
        else:
            out[:, a] = np.interp(x_new, x_old, segment[:, a])
    return out


def _analyze_file_worker(path_str: str, fps: float) -> dict[str, Any]:
    """Parallel worker for deep mocap motif extraction."""
    parser = TimelineParser(KINEMATIC_CONTROLLERS)
    resampler = MotionResampler(fps=fps)
    path = Path(path_str)
    context_hint = infer_context_label(path)
    clips_out: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for clip in parser.parse_file(path):
        res = resampler.resample(clip)
        if res is None or "hipControl" not in res.signals:
            continue
        signals = res.signals
        t = res.time
        hip = signals["hipControl"][list(AXES)].to_numpy(dtype=float)
        anchor: dict[str, Any] = {}
        deltas: dict[str, np.ndarray] = {}
        for ctrl in KINEMATIC_CONTROLLERS:
            df = signals.get(ctrl)
            if df is None or df.empty:
                continue
            v = df[list(AXES)].to_numpy(dtype=float)
            a0 = v[0]
            anchor[ctrl] = {"x": float(a0[0]), "y": float(a0[1]), "z": float(a0[2]), "quat": [0.0, 0.0, 0.0, 1.0]}
            deltas[ctrl] = v - a0

        hip_d = deltas.get("hipControl")
        if hip_d is None:
            continue
        y = hip_d[:, 1] - float(np.mean(hip_d[:, 1]))
        peaks, _ = find_peaks(y, distance=max(1, int(fps * 0.20)), prominence=max(float(np.std(y)) * 0.2, 1e-6))
        hip_cycles: list[dict[str, Any]] = []
        motion_chunks: list[dict[str, Any]] = []
        if len(peaks) > 2:
            for i in range(len(peaks) - 1):
                s, e = int(peaks[i]), int(peaks[i + 1])
                if e - s < 4:
                    continue
                seg = hip_d[s:e, :]
                rs = _resample_xyz(seg, 96)
                chunk_controllers: dict[str, dict[str, list[float]]] = {}
                for ctrl in KINEMATIC_CONTROLLERS:
                    arr = signals.get(ctrl)
                    if arr is None or arr.empty:
                        continue
                    ctrl_abs = arr[list(AXES)].to_numpy(dtype=float)[s:e, :]
                    if ctrl_abs.shape[0] < 2:
                        continue
                    ctrl_rel = ctrl_abs - hip[s:e, :]
                    chunk_controllers[ctrl] = {
                        "x": ctrl_rel[:, 0].astype(float).tolist(),
                        "y": ctrl_rel[:, 1].astype(float).tolist(),
                        "z": ctrl_rel[:, 2].astype(float).tolist(),
                    }
                chunk = {
                    "id": f"{path.name}:{clip.clip_name}:{i}",
                    "source_file": path.name,
                    "clip_name": clip.clip_name,
                    "duration_s": float((e - s) / fps),
                    "fps": float(fps),
                    "context": context_hint,
                    "hip_delta": {
                        "x": seg[:, 0].astype(float).tolist(),
                        "y": seg[:, 1].astype(float).tolist(),
                        "z": seg[:, 2].astype(float).tolist(),
                    },
                    "controllers": chunk_controllers,
                }
                motion_chunks.append(chunk)
                hip_cycles.append(
                    {
                        "duration_s": float((e - s) / fps),
                        "traj": rs.tolist(),
                        "amp_y": float(np.max(seg[:, 1]) - np.min(seg[:, 1])),
                    }
                )

        hand_profiles: dict[str, list[dict[str, Any]]] = {"lHandControl": [], "rHandControl": []}
        chest_d = deltas.get("chestControl")
        for hand in ("lHandControl", "rHandControl"):
            hd = deltas.get(hand)
            if hd is None or chest_d is None:
                continue
            vel = np.linalg.norm(np.gradient(hd, axis=0) * fps, axis=1)
            thr = float(np.percentile(vel, 90))
            runs = _contiguous_runs(vel > thr)
            for s, e in runs:
                if e - s < int(0.35 * fps):
                    continue
                # Learn hand motifs in chest-local frame:
                # P_local = R_chest^-1 * (P_hand_world - P_chest_world)
                rel = hd[s:e, :] - chest_d[s:e, :]
                rs = _resample_xyz(rel, 72)
                hand_profiles[hand].append({"duration_s": float((e - s) / fps), "traj_local_chest": rs.tolist()})

        head_intentional: list[dict[str, Any]] = []
        hd = deltas.get("headControl")
        cd = deltas.get("chestControl")
        if hd is not None:
            smooth = gaussian_filter1d(hd, sigma=max(1.0, fps * 0.8), axis=0)
            gaze = smooth - smooth[0]
            # Keep slow directional turns as intentional gaze profiles.
            speed = np.linalg.norm(np.gradient(gaze, axis=0) * fps, axis=1)
            runs = _contiguous_runs(speed > float(np.percentile(speed, 70)))
            for s, e in runs:
                if e - s < int(1.0 * fps):
                    continue
                rs = _resample_xyz(gaze[s:e, :], 72)
                head_intentional.append({"duration_s": float((e - s) / fps), "traj": rs.tolist()})

        # Cross-correlation phase rows (all controller pairs).
        ctrls = [c for c in KINEMATIC_CONTROLLERS if c in deltas]
        for i, a in enumerate(ctrls):
            for b in ctrls[i + 1 :]:
                sa = deltas[a][:, 1] - np.mean(deltas[a][:, 1])
                sb = deltas[b][:, 1] - np.mean(deltas[b][:, 1])
                if np.std(sa) <= 1e-8 or np.std(sb) <= 1e-8:
                    continue
                corr = np.correlate(sb, sa, mode="full")
                lag = int(np.argmax(corr) - (len(sa) - 1))
                phase_rows.append({"a": a, "b": b, "delay_s": float(lag / fps)})

        # Bone lengths from anchor.
        def dist(c1: str, c2: str) -> float | None:
            if c1 not in anchor or c2 not in anchor:
                return None
            p1 = np.array([anchor[c1]["x"], anchor[c1]["y"], anchor[c1]["z"]], dtype=float)
            p2 = np.array([anchor[c2]["x"], anchor[c2]["y"], anchor[c2]["z"]], dtype=float)
            return float(np.linalg.norm(p1 - p2))

        bone_lengths = {
            "hip_chest": dist("hipControl", "chestControl"),
            "chest_head": dist("chestControl", "headControl"),
            "l_hip_thigh": dist("hipControl", "lThighControl"),
            "l_thigh_knee": dist("lThighControl", "lKneeControl"),
            "l_knee_foot": dist("lKneeControl", "lFootControl"),
            "r_hip_thigh": dist("hipControl", "rThighControl"),
            "r_thigh_knee": dist("rThighControl", "rKneeControl"),
            "r_knee_foot": dist("rKneeControl", "rFootControl"),
        }
        clips_out.append(
            {
                "clip_name": clip.clip_name,
                "source_file": path.name,
                "context_hint": context_hint,
                "anchor": anchor,
                "bone_lengths": bone_lengths,
                "hip_cycles": hip_cycles,
                "hand_profiles": hand_profiles,
                "head_intentional": head_intentional,
                "hip_delta_summary": {
                    "amp_y": float(np.percentile(hip_d[:, 1], 95) - np.percentile(hip_d[:, 1], 5)),
                    "amp_x": float(np.percentile(hip_d[:, 0], 95) - np.percentile(hip_d[:, 0], 5)),
                    "amp_z": float(np.percentile(hip_d[:, 2], 95) - np.percentile(hip_d[:, 2], 5)),
                },
                "motion_chunks": motion_chunks,
            }
        )
    return {"file": path.name, "file_path": str(path), "context_hint": context_hint, "clips": clips_out, "phases": phase_rows}


def resolve_context(requested: str, available: list[str]) -> str:
    def norm(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    r = norm(requested)
    for c in available:
        if norm(c) == r:
            return c
    for c in available:
        cn = norm(c)
        if r in cn or cn in r:
            return c
    raise KeyError(f"Unknown context '{requested}'. Available: {', '.join(available)}")


def infer_context_label(path: Path, clip_name: str = "") -> str:
    """Infer context from folder/file naming for strict data siloing."""
    text = " ".join([str(path.parent.name), path.stem, clip_name]).lower()
    token_map = {
        "Cowgirl": ("cowgirl", "ride", "riding"),
        "Missionary": ("missionary", "thrust"),
        "Oral": ("blowjob", "oral", "bj"),
        "Grinding": ("grind", "circular"),
    }
    for label, tokens in token_map.items():
        if any(tok in text for tok in tokens):
            return label
    parent = path.parent.name.strip()
    if parent and parent.lower() not in {"savedmocaps", "animations"}:
        return parent
    stem = path.stem.strip()
    return stem.split("_")[0] if "_" in stem else stem


class BehavioralFeatureExtractor:
    """Offline learning: pose motifs + secondary track probabilities."""

    def __init__(self, mocap_dir: Path, output_path: Path, fps: float = 30.0) -> None:
        self.mocap_dir = mocap_dir
        self.output_path = output_path
        self.fps = fps
        self.parser = TimelineParser(KINEMATIC_CONTROLLERS)
        self.resampler = MotionResampler(fps=fps)

    def learn(self, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        progress = progress or (lambda _v, _m: None)
        files = sorted(self.mocap_dir.rglob("*.json"))
        if not files:
            raise RuntimeError(f"No JSON files found in {self.mocap_dir}")
        progress(0.0, f"Parallel deep scan over {len(files)} files")
        workers = max(1, min(len(files), (os.cpu_count() or 8)))
        results: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            fut_to_file = {pool.submit(_analyze_file_worker, str(path), self.fps): path for path in files}
            done = 0
            for fut in concurrent.futures.as_completed(fut_to_file):
                done += 1
                path = fut_to_file[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    LOGGER.warning("Worker failed for %s: %s", path.name, exc)
                progress(done / max(len(files), 1), f"Deep-scanned {path.name}")

        clips = [clip for res in results for clip in res.get("clips", [])]
        if not clips:
            raise RuntimeError("No usable mocap content found.")
        context_to_clips: dict[str, list[dict[str, Any]]] = {}
        for clip in clips:
            label = str(clip.get("context_hint") or "Unsorted")
            context_to_clips.setdefault(label, []).append(clip)
        all_phase_rows = [row for res in results for row in res.get("phases", [])]
        contexts_model: dict[str, Any] = {}
        for label, cset in context_to_clips.items():
            context = self._build_context_from_worker_results(cset, all_phase_rows)
            contexts_model[label] = context
        all_chunks = [ch for c in clips for ch in c.get("motion_chunks", [])]
        model = {
            "version": 3,
            "sample_count": int(len(all_chunks)),
            "source_dir": str(self.mocap_dir),
            "contexts": contexts_model,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as h:
            json.dump(model, h, indent=2, ensure_ascii=False)
        progress(1.0, f"Saved {self.output_path.name}")
        return model
    def _build_context_from_worker_results(self, clips: list[dict[str, Any]], phases: list[dict[str, Any]]) -> dict[str, Any]:
        # Rhythm and motif pools from real micro-cycles.
        hip_cycles = [cy for c in clips for cy in c.get("hip_cycles", [])]
        motion_chunks = [ch for c in clips for ch in c.get("motion_chunks", [])]
        hand_pool = {
            "lHandControl": [p for c in clips for p in c.get("hand_profiles", {}).get("lHandControl", [])],
            "rHandControl": [p for c in clips for p in c.get("hand_profiles", {}).get("rHandControl", [])],
        }
        head_pool = [p for c in clips for p in c.get("head_intentional", [])]

        bpm_vals = []
        amp_vals = []
        for cy in hip_cycles:
            d = finite_float(cy.get("duration_s"), 0.0)
            if d > 1e-6:
                bpm_vals.append(60.0 / d)
            amp_vals.append(finite_float(cy.get("amp_y"), np.nan))
        bpm = self._summary(bpm_vals, (60.0, 150.0, 95.0))
        amp = self._summary(amp_vals, (0.03, 0.15, 0.08))

        # Bone lengths aggregated from anchors.
        lengths: dict[str, list[float]] = {}
        for c in clips:
            for k, v in c.get("bone_lengths", {}).items():
                if v is None:
                    continue
                lengths.setdefault(k, []).append(float(v))
        bone_lengths = {k: float(np.median(v)) for k, v in lengths.items() if v}

        # Dominant hidden phase delays by controller pair.
        phase_map: dict[str, float] = {}
        if phases:
            dfp = pd.DataFrame(phases)
            for (a, b), grp in dfp.groupby(["a", "b"]):
                phase_map[f"{a}->{b}"] = float(np.median(grp["delay_s"].to_numpy(dtype=float)))

        # Delta-centric pose state priors from cycle amplitudes (kept simple but data-driven).
        pose_states = self._fallback_states()
        if hip_cycles:
            amps = np.array([finite_float(cy.get("amp_y"), 0.0) for cy in hip_cycles], dtype=float)
            q1, q2, q3 = np.percentile(amps, [25, 50, 75])
            pose_states["Warmup_Teasing"]["probability"] = float(np.mean(amps <= q1))
            pose_states["Steady_Riding"]["probability"] = float(np.mean((amps > q1) & (amps <= q2)))
            pose_states["Intense_LeaningBack"]["probability"] = float(np.mean(amps > q3))
            pose_states["Exhausted_LeaningFwd"]["probability"] = float(np.mean((amps > q2) & (amps <= q3)))

        secondary = {
            "hands": {
                "lHandControl": {"events_per_minute": float(len(hand_pool["lHandControl"]) / max(len(clips), 1)), "profiles": hand_pool["lHandControl"]},
                "rHandControl": {"events_per_minute": float(len(hand_pool["rHandControl"]) / max(len(clips), 1)), "profiles": hand_pool["rHandControl"]},
            },
            "head_look": {"events_per_minute": float(len(head_pool) / max(len(clips), 1)), "profiles": head_pool},
        }
        return {
            "rhythm": {"bpm": bpm, "amplitude": amp, "dominant_axis": "y", "lead_controller": "hipControl"},
            "delta_mode": True,
            "motion_dictionary": {"fps": float(self.fps), "chunks": motion_chunks},
            "motifs": {"hip_cycles": hip_cycles},
            "secondary_actions": secondary,
            "pose_states": pose_states,
            "markov": self._default_markov(),
            "kinematic_chain": {
                "hip_to_chest_delay_s": [0.08, 0.15],
                "hip_to_chest_decay": [0.40, 0.50],
                "chest_to_head_delay_s": [0.05, 0.12],
                "chest_to_head_decay": [0.45, 0.60],
            },
            "bone_lengths": bone_lengths,
            "phase_delays": phase_map,
        }


    @staticmethod
    def _summary(values: list[float], default: tuple[float, float, float]) -> dict[str, float]:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            return {"min": default[0], "max": default[1], "mean": default[2]}
        return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}

    @staticmethod
    def _default_markov() -> dict[str, dict[str, float]]:
        return {
            "Warmup_Teasing": {"Warmup_Teasing": 0.30, "Steady_Riding": 0.60, "Intense_LeaningBack": 0.10},
            "Steady_Riding": {"Steady_Riding": 0.35, "Intense_LeaningBack": 0.45, "Exhausted_LeaningFwd": 0.20},
            "Intense_LeaningBack": {"Steady_Riding": 0.45, "Intense_LeaningBack": 0.30, "Exhausted_LeaningFwd": 0.25},
            "Exhausted_LeaningFwd": {"Warmup_Teasing": 0.15, "Steady_Riding": 0.55, "Exhausted_LeaningFwd": 0.30},
        }

    @staticmethod
    def _fallback_states() -> dict[str, Any]:
        return {
            s: {"probability": 0.25, "pose_offsets": {c: {a: 0.0 for a in AXES} for c in KINEMATIC_CONTROLLERS}}
            for s in STATE_NAMES
        }


class MarkovBrain:
    """Behavior state scheduler with stamina-dependent fatigue drift."""

    def __init__(self, context: dict[str, Any], params: DirectorParameters) -> None:
        self.context = context
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def plan(self) -> list[StateSegment]:
        trans = self._adjust_transitions()
        t = 0.0
        state = "Warmup_Teasing"
        segments: list[StateSegment] = []
        while t < self.params.duration - 1e-6:
            dur = float(self.rng.uniform(9.0, 22.0))
            if state == "Intense_LeaningBack":
                dur *= 0.80
            if state == "Exhausted_LeaningFwd":
                dur *= 1.20
            end = min(self.params.duration, t + dur)
            segments.append(StateSegment(state, t, end, self._state_intensity(state)))
            state = self._next(state, trans)
            t = end
        return segments

    def _adjust_transitions(self) -> dict[str, dict[str, float]]:
        base_raw = self.context.get("markov", BehavioralFeatureExtractor._default_markov())
        # Backward compatibility: accept old flat rows like {"Steady_Riding": 0.8, ...}
        if base_raw and all(isinstance(v, (int, float)) for v in base_raw.values()):
            base: dict[str, dict[str, float]] = BehavioralFeatureExtractor._default_markov()
        else:
            base = {
                state: row if isinstance(row, dict) else BehavioralFeatureExtractor._default_markov().get(state, {})
                for state, row in base_raw.items()
            }
        out: dict[str, dict[str, float]] = {}
        fatigue = 1.0 - np.clip(self.params.stamina, 0.0, 1.0)
        for state, row in base.items():
            r = dict(row)
            r["Exhausted_LeaningFwd"] = r.get("Exhausted_LeaningFwd", 0.0) + 0.32 * fatigue
            if self.params.stamina > 0.65:
                r["Intense_LeaningBack"] = r.get("Intense_LeaningBack", 0.0) + 0.10
            s = sum(max(v, 0.0) for v in r.values())
            out[state] = {k: max(v, 0.0) / s for k, v in r.items()}
        return out

    def _state_intensity(self, state: str) -> float:
        base = float(np.clip(self.params.base_intensity, 0.1, 1.0))
        delta = {"Warmup_Teasing": -0.22, "Steady_Riding": 0.0, "Intense_LeaningBack": 0.24, "Exhausted_LeaningFwd": -0.20}.get(state, 0.0)
        return float(np.clip(base + delta + self.rng.normal(0, 0.03), 0.08, 1.0))

    def _next(self, state: str, trans: dict[str, dict[str, float]]) -> str:
        row = trans.get(state, trans["Steady_Riding"])
        keys = list(row)
        p = np.array([row[k] for k in keys], dtype=float)
        p = p / p.sum()
        return str(self.rng.choice(keys, p=p))


class BiomechanicalSynthesisEngine:
    """Fourier motor + wave propagation + anchors + playful spline idles."""

    def __init__(self, context: dict[str, Any], params: DirectorParameters) -> None:
        self.context = context
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def synthesize(self, segments: list[StateSegment]) -> dict[str, Any]:
        t = np.linspace(0.0, self.params.duration, int(self.params.duration * INTERNAL_FPS) + 1)
        if self.context.get("motion_dictionary", {}).get("chunks"):
            positions = self._init_positions(t)
            rotations = self._init_rotations(t)
            events = self._synthesize_from_motion_chunks(t, positions)
            self._layer4_micro_noise(t, positions, rotations)
            self._enforce_static_locks(t, positions)
            self._ik_distance_check(positions)
            self._normalize_rotations(rotations)
            return {"time": t, "positions": positions, "rotations": rotations, "segments": segments, "events": events}
        intensity = self._intensity_curve(t, segments)
        positions = self._init_positions(t)
        rotations = self._init_rotations(t)

        self._layer1_hip_motor(t, intensity, positions)
        self._layer1_wave_propagation(t, intensity, positions, rotations)
        self._layer2_pose_offsets(t, segments, positions, rotations)
        events = self._layer3_secondary_tracks(t, intensity, positions, rotations)
        self._layer4_micro_noise(t, positions, rotations)
        self._enforce_static_locks(t, positions)
        self._ik_distance_check(positions)
        self._normalize_rotations(rotations)
        return {"time": t, "positions": positions, "rotations": rotations, "segments": segments, "events": events}

    def _synthesize_from_motion_chunks(
        self, t: np.ndarray, positions: dict[str, dict[str, np.ndarray]]
    ) -> list[SecondaryEvent]:
        """Replay learned chunk trajectories 1:1 (resampled), no synthetic limb invention."""
        chunks = list(self.context.get("motion_dictionary", {}).get("chunks", []))
        if not chunks:
            return []
        events: list[SecondaryEvent] = []
        n = len(t)
        cur = 0
        while cur < n:
            chunk = chunks[int(self.rng.integers(0, len(chunks)))]
            hip = chunk.get("hip_delta", {})
            hx = np.asarray(hip.get("x", []), dtype=float)
            hy = np.asarray(hip.get("y", []), dtype=float)
            hz = np.asarray(hip.get("z", []), dtype=float)
            if min(len(hx), len(hy), len(hz)) < 4:
                cur += 1
                continue
            chunk_dur = max(float(chunk.get("duration_s", 0.0)), 0.15)
            frame_len = max(4, int(round(chunk_dur * INTERNAL_FPS)))
            end = min(n, cur + frame_len)
            out_len = end - cur
            rs_hip = _resample_xyz(np.column_stack([hx, hy, hz]), out_len)
            positions["hipControl"]["x"][cur:end] = rs_hip[:, 0]
            positions["hipControl"]["y"][cur:end] = rs_hip[:, 1]
            positions["hipControl"]["z"][cur:end] = rs_hip[:, 2]

            ctrls = chunk.get("controllers", {})
            for ctrl in KINEMATIC_CONTROLLERS:
                if ctrl == "hipControl":
                    continue
                c = ctrls.get(ctrl)
                if not isinstance(c, dict):
                    continue
                cx = np.asarray(c.get("x", []), dtype=float)
                cy = np.asarray(c.get("y", []), dtype=float)
                cz = np.asarray(c.get("z", []), dtype=float)
                if min(len(cx), len(cy), len(cz)) < 4:
                    continue
                rs_rel = _resample_xyz(np.column_stack([cx, cy, cz]), out_len)
                positions[ctrl]["x"][cur:end] = positions["hipControl"]["x"][cur:end] + rs_rel[:, 0]
                positions[ctrl]["y"][cur:end] = positions["hipControl"]["y"][cur:end] + rs_rel[:, 1]
                positions[ctrl]["z"][cur:end] = positions["hipControl"]["z"][cur:end] + rs_rel[:, 2]
            events.append(
                SecondaryEvent(
                    "chunk_replay",
                    "hipControl",
                    float(t[cur]),
                    float(t[end - 1]),
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                )
            )
            cur = end
        return events

    def _intensity_curve(self, t: np.ndarray, segments: list[StateSegment]) -> np.ndarray:
        kt = [0.0]
        kv = [segments[0].intensity if segments else self.params.base_intensity]
        for s in segments:
            kt.extend([s.start, s.end])
            kv.extend([s.intensity, s.intensity])
        kt.append(self.params.duration)
        kv.append(kv[-1])
        uk, idx = np.unique(np.asarray(kt), return_index=True)
        vv = np.asarray(kv)[idx]
        out = CubicSpline(uk, vv, bc_type="natural")(t) if len(uk) >= 4 else np.interp(t, uk, vv)
        fatigue = np.linspace(0.0, 1.0 - np.clip(self.params.stamina, 0.0, 1.0), len(t))
        return np.clip(out * (1.0 - fatigue * 0.15), 0.05, 1.0)

    def _init_positions(self, t: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        pos: dict[str, dict[str, np.ndarray]] = {}
        for c in KINEMATIC_CONTROLLERS:
            pos[c] = {}
            for a in AXES:
                pos[c][a] = np.zeros_like(t, dtype=float)
        return pos

    @staticmethod
    def _init_rotations(t: np.ndarray) -> dict[str, np.ndarray]:
        return {c: np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(t), 1)) for c in KINEMATIC_CONTROLLERS}

    def _layer1_hip_motor(self, t: np.ndarray, intensity: np.ndarray, positions: dict[str, dict[str, np.ndarray]]) -> None:
        r = self.context.get("rhythm", {})
        bpm_cfg = r.get("bpm", {})
        amp_cfg = r.get("amplitude", {})
        bpm = np.interp(
            intensity,
            [0.0, 0.5, 1.0],
            [finite_float(bpm_cfg.get("min"), 60.0), finite_float(bpm_cfg.get("mean"), 95.0), finite_float(bpm_cfg.get("max"), 155.0)],
        )
        stamina_decay = 1.0 - np.clip(self.params.stamina, 0.0, 1.0)
        bpm *= self._tempo_noise(t, depth=0.02 + 0.09 * stamina_decay)
        amp = np.interp(
            intensity,
            [0.0, 0.5, 1.0],
            [finite_float(amp_cfg.get("min"), 0.03), finite_float(amp_cfg.get("mean"), 0.08), finite_float(amp_cfg.get("max"), 0.15)],
        )
        amp *= (0.86 + 0.28 * intensity)

        phase = 2.0 * np.pi * np.cumsum((bpm / 60.0) * np.diff(t, prepend=t[0]))
        # Fourier push-pull asymmetry
        y_wave = (
            0.78 * np.sin(phase)
            + 0.24 * np.sin(2.0 * phase - 0.38)
            + 0.10 * np.sin(3.0 * phase + 0.22)
        )
        y_wave = np.tanh(1.18 * y_wave)  # slightly asymmetric muscle push profile
        positions["hipControl"]["y"] += amp * y_wave
        positions["hipControl"]["x"] += amp * 0.17 * np.sin(phase + np.pi / 2.0)
        positions["hipControl"]["z"] += amp * 0.30 * np.sin(phase + np.pi * 0.78)

    def _layer1_wave_propagation(
        self,
        t: np.ndarray,
        intensity: np.ndarray,
        positions: dict[str, dict[str, np.ndarray]],
        rotations: dict[str, np.ndarray],
    ) -> None:
        chain = self.context.get("kinematic_chain", {})
        d_hip_chest = float(self.rng.uniform(*chain.get("hip_to_chest_delay_s", [0.08, 0.15])))
        d_chest_head = float(self.rng.uniform(*chain.get("chest_to_head_delay_s", [0.05, 0.12])))
        a_hip_chest = float(self.rng.uniform(*chain.get("hip_to_chest_decay", [0.40, 0.50])))
        a_chest_head = float(self.rng.uniform(*chain.get("chest_to_head_decay", [0.45, 0.60])))

        hip_y = positions["hipControl"]["y"]
        chest_y = np.interp(np.clip(t - d_hip_chest, t[0], t[-1]), t, hip_y)
        head_y = np.interp(np.clip(t - d_hip_chest - d_chest_head, t[0], t[-1]), t, hip_y)
        positions["chestControl"]["y"] += (chest_y - np.median(chest_y)) * a_hip_chest
        positions["headControl"]["y"] += (head_y - np.median(head_y)) * a_hip_chest * a_chest_head

        hip_z = positions["hipControl"]["z"]
        positions["chestControl"]["z"] += np.interp(np.clip(t - d_hip_chest, t[0], t[-1]), t, hip_z - np.median(hip_z)) * a_hip_chest * 0.9
        positions["headControl"]["z"] += np.interp(np.clip(t - d_hip_chest - d_chest_head, t[0], t[-1]), t, hip_z - np.median(hip_z)) * a_hip_chest * a_chest_head * 0.9

        chest_pitch = np.deg2rad(4.8 * np.sin(np.linspace(0, 2 * np.pi, len(t))) * intensity)
        chest_rot = Rotation.from_euler("xyz", np.column_stack([chest_pitch, np.zeros_like(chest_pitch), np.zeros_like(chest_pitch)]))
        rotations["chestControl"] = chest_rot.as_quat()

        # Vestibulo-ocular reflex: partial counter-rotation of head against chest pitch.
        vor_pitch = -0.55 * chest_pitch
        head_base_rot = Rotation.from_euler("xyz", np.column_stack([vor_pitch, np.zeros_like(vor_pitch), np.zeros_like(vor_pitch)]))
        # Slerp blend towards stabilized target
        result = np.empty_like(rotations["headControl"])
        for i in range(len(t)):
            key = Rotation.from_quat([rotations["headControl"][i], head_base_rot.as_quat()[i]])
            s = Slerp([0.0, 1.0], key)
            result[i] = s([0.72]).as_quat()[0]
        rotations["headControl"] = result

    def _layer2_pose_offsets(
        self,
        t: np.ndarray,
        segments: list[StateSegment],
        positions: dict[str, dict[str, np.ndarray]],
        rotations: dict[str, np.ndarray],
    ) -> None:
        states = self.context.get("pose_states", {})
        for ctrl in KINEMATIC_CONTROLLERS:
            if ctrl in LOCKED_EXTREMITIES:
                continue
            for ax in AXES:
                kt: list[float] = []
                kv: list[float] = []
                for s in segments:
                    off = finite_float(states.get(s.state, {}).get("pose_offsets", {}).get(ctrl, {}).get(ax), 0.0)
                    kt.extend([s.start, s.end])
                    kv.extend([off, off])
                if not kt:
                    continue
                uk, idx = np.unique(np.asarray(kt), return_index=True)
                vv = np.asarray(kv)[idx]
                add = CubicSpline(uk, vv, bc_type="natural")(t) if len(uk) >= 4 else np.interp(t, uk, vv)
                positions[ctrl][ax] += add

        # Intensity posture coupling: high intensity leans back and extends stroke.
        high = np.clip(self.params.base_intensity - 0.8, 0.0, 0.2) / 0.2
        if high > 0:
            positions["chestControl"]["z"] += 0.08 * high
            back_rot = Rotation.from_euler("xyz", [-14.0 * high, 0.0, 0.0], degrees=True).as_quat()
            for i in range(len(t)):
                key = Rotation.from_quat([rotations["chestControl"][i], back_rot])
                rotations["chestControl"][i] = Slerp([0.0, 1.0], key)([0.62]).as_quat()[0]

    def _layer3_secondary_tracks(
        self,
        t: np.ndarray,
        intensity: np.ndarray,
        positions: dict[str, dict[str, np.ndarray]],
        rotations: dict[str, np.ndarray],
    ) -> list[SecondaryEvent]:
        secondary = self.context.get("secondary_actions", {})
        play = float(np.clip(self.params.playfulness, 0.0, 1.0))
        duration_min = self.params.duration / 60.0
        events: list[SecondaryEvent] = []

        # Head look side events
        look_rate = finite_float(secondary.get("head_look", {}).get("events_per_minute"), 1.5) * (0.2 + play)
        n_look = int(self.rng.poisson(max(0.0, look_rate * duration_min)))
        for _ in range(n_look):
            d = float(self.rng.uniform(2.0, 3.0))
            s = float(self.rng.uniform(0.0, max(0.0, self.params.duration - d)))
            yaw = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(8.0, 24.0))
            ev = SecondaryEvent("head_look", "headControl", s, s + d, {"x": 0.0, "y": 0.0, "z": 0.0}, (0.0, yaw, self.rng.uniform(-4.0, 4.0)))
            events.append(ev)
            self._apply_head_look_event(t, rotations["headControl"], ev)

        # Hand leave-anchor to body and return via CubicSpline C2 path
        hands = secondary.get("hands", {})
        for hand in ("lHandControl", "rHandControl"):
            info = hands.get(hand, {})
            rate = finite_float(info.get("events_per_minute"), 0.8) * (0.15 + 1.25 * play)
            n = int(self.rng.poisson(max(0.0, rate * duration_min)))
            target = info.get("body_target", {"x": 0.0, "y": 0.0, "z": 0.0})
            profiles = info.get("profiles", [])
            for _ in range(n):
                d = float(self.rng.uniform(2.6, 4.4))
                s = float(self.rng.uniform(0.0, max(0.0, self.params.duration - d)))
                ev = SecondaryEvent("hand_body", hand, s, s + d, {a: finite_float(target.get(a), 0.0) for a in AXES})
                events.append(ev)
                if profiles:
                    profile = profiles[int(self.rng.integers(0, len(profiles)))]
                    traj_local = np.asarray(profile.get("traj_local_chest", []), dtype=float)
                    if traj_local.ndim == 2 and traj_local.shape[0] >= 4:
                        self._apply_hand_local_profile(t, positions, rotations, hand, ev, traj_local)
                        continue
                self._apply_hand_event(t, positions[hand], ev)

        # Anchoring default: keep hands mostly stationary with subtle muscle jitter if not in event.
        for hand in ("lHandControl", "rHandControl"):
            for ax in AXES:
                positions[hand][ax] = gaussian_filter1d(positions[hand][ax], sigma=1.2)
        return sorted(events, key=lambda e: e.start)

    def _apply_head_look_event(self, t: np.ndarray, quats: np.ndarray, event: SecondaryEvent) -> None:
        mask = (t >= event.start) & (t <= event.end)
        if not np.any(mask):
            return
        local_t = (t[mask] - event.start) / max(1e-6, event.end - event.start)
        weight = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t)
        target = Rotation.from_euler("xyz", event.rot_euler_deg, degrees=True).as_quat()
        for i, idx in enumerate(np.where(mask)[0]):
            key = Rotation.from_quat([quats[idx], target])
            quats[idx] = Slerp([0.0, 1.0], key)([float(weight[i])]).as_quat()[0]

    def _apply_hand_event(self, t: np.ndarray, hand_axes: dict[str, np.ndarray], event: SecondaryEvent) -> None:
        mask = (t >= event.start) & (t <= event.end)
        if np.count_nonzero(mask) < 4:
            return
        idx = np.where(mask)[0]
        tt = t[idx]
        mid = event.start + 0.55 * (event.end - event.start)
        for ax in AXES:
            start_v = float(hand_axes[ax][idx[0]])
            end_v = float(hand_axes[ax][idx[-1]])
            peak = float(event.target[ax])
            k_t = np.array([event.start, event.start + 0.25 * (event.end - event.start), mid, event.end])
            k_v = np.array([start_v, start_v * 0.65 + peak * 0.35, peak, end_v])
            spl = CubicSpline(k_t, k_v, bc_type="natural")
            hand_axes[ax][idx] = spl(tt)

    def _apply_hand_local_profile(
        self,
        t: np.ndarray,
        positions: dict[str, dict[str, np.ndarray]],
        rotations: dict[str, np.ndarray],
        hand_ctrl: str,
        event: SecondaryEvent,
        traj_local_chest: np.ndarray,
    ) -> None:
        """Replay chest-local hand profile in world frame:
        P_hand_world = P_chest_world + R_chest_world * P_hand_local
        """
        mask = (t >= event.start) & (t <= event.end)
        idx = np.where(mask)[0]
        if len(idx) < 4:
            return
        local = _resample_xyz(traj_local_chest, len(idx))
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.linspace(0.0, 1.0, len(idx)))
        for j, k in enumerate(idx):
            chest_p = np.array(
                [
                    positions["chestControl"]["x"][k],
                    positions["chestControl"]["y"][k],
                    positions["chestControl"]["z"][k],
                ],
                dtype=float,
            )
            chest_r = Rotation.from_quat(normalize_quat(rotations["chestControl"][k]))
            hand_world_target = chest_p + chest_r.apply(local[j])
            for a_idx, ax in enumerate(AXES):
                current = positions[hand_ctrl][ax][k]
                positions[hand_ctrl][ax][k] = current * (1.0 - w[j]) + hand_world_target[a_idx] * w[j]

    def _layer4_micro_noise(self, t: np.ndarray, positions: dict[str, dict[str, np.ndarray]], rotations: dict[str, np.ndarray]) -> None:
        for ctrl in KINEMATIC_CONTROLLERS:
            for ax in AXES:
                # tiny on anchored limbs, larger on torso
                scale = 0.0008 if ctrl in LOCKED_EXTREMITIES else (0.0012 if "Hand" in ctrl else 0.0035)
                positions[ctrl][ax] += self._smooth_noise(t, scale, knot_s=1.8)
            euler_noise = np.column_stack(
                [self._smooth_noise(t, 0.8, knot_s=2.2), self._smooth_noise(t, 0.7, knot_s=2.0), self._smooth_noise(t, 0.6, knot_s=2.1)]
            )
            rotations[ctrl] = (Rotation.from_quat(rotations[ctrl]) * Rotation.from_euler("xyz", euler_noise, degrees=True)).as_quat()

    def _enforce_static_locks(self, t: np.ndarray, positions: dict[str, dict[str, np.ndarray]]) -> None:
        for ctrl in LOCKED_EXTREMITIES:
            for ax in AXES:
                positions[ctrl][ax][:] = 0.0
        # Hand anchoring in delta-space: around zero with micro tremor only.
        for hand in ("lHandControl", "rHandControl"):
            for ax in AXES:
                positions[hand][ax] = 0.12 * positions[hand][ax]
                positions[hand][ax] += self._smooth_noise(t, 0.0009, knot_s=1.9)

    @staticmethod
    def _normalize_rotations(rotations: dict[str, np.ndarray]) -> None:
        for ctrl in list(rotations):
            rotations[ctrl] = normalize_quat(rotations[ctrl])

    def _tempo_noise(self, t: np.ndarray, depth: float) -> np.ndarray:
        return np.clip(1.0 + self._smooth_noise(t, depth, knot_s=2.6), 0.80, 1.20)

    def _smooth_noise(self, t: np.ndarray, scale: float, knot_s: float) -> np.ndarray:
        knots = max(4, int(math.ceil(max(float(t[-1]), 1.0) / knot_s)) + 2)
        kt = np.linspace(t[0], t[-1], knots)
        kv = self.rng.normal(0.0, scale, knots)
        out = np.interp(t, kt, kv)
        out = gaussian_filter1d(out, sigma=max(1.0, INTERNAL_FPS * 0.08))
        return out - float(np.mean(out))

    def _ik_distance_check(self, positions: dict[str, dict[str, np.ndarray]]) -> None:
        """Clamp hip Y when hip-foot distance exceeds Dmax = L_thigh + L_calf."""
        bones = self.context.get("bone_lengths", {})
        l_thigh = finite_float(bones.get("l_thigh_knee"), 0.0)
        l_calf = finite_float(bones.get("l_knee_foot"), 0.0)
        r_thigh = finite_float(bones.get("r_thigh_knee"), 0.0)
        r_calf = finite_float(bones.get("r_knee_foot"), 0.0)
        dmax_l = l_thigh + l_calf if l_thigh > 0 and l_calf > 0 else None
        dmax_r = r_thigh + r_calf if r_thigh > 0 and r_calf > 0 else None
        n = len(positions["hipControl"]["x"])
        for i in range(n):
            hip = np.array([positions["hipControl"]["x"][i], positions["hipControl"]["y"][i], positions["hipControl"]["z"][i]], dtype=float)
            corr_y = 0.0
            if dmax_l is not None:
                foot_l = np.array([positions["lFootControl"]["x"][i], positions["lFootControl"]["y"][i], positions["lFootControl"]["z"][i]], dtype=float)
                d = float(np.linalg.norm(hip - foot_l))
                if d > dmax_l:
                    corr_y = min(corr_y, -abs(d - dmax_l))
            if dmax_r is not None:
                foot_r = np.array([positions["rFootControl"]["x"][i], positions["rFootControl"]["y"][i], positions["rFootControl"]["z"][i]], dtype=float)
                d = float(np.linalg.norm(hip - foot_r))
                if d > dmax_r:
                    corr_y = min(corr_y, -abs(d - dmax_r))
            if corr_y != 0.0:
                positions["hipControl"]["y"][i] += corr_y


class TimelineExporter:
    """Export generated arrays as VaM Timeline t/v/c curves (Bezier curve type 3)."""

    def assemble(self, generated: dict[str, Any], context: str) -> dict[str, Any]:
        t = generated["time"]
        duration = float(t[-1])
        et = np.linspace(0.0, duration, int(duration * EXPORT_FPS) + 1)
        controllers = []
        pos = generated["positions"]
        rot = generated["rotations"]
        for ctrl in KINEMATIC_CONTROLLERS:
            q = np.column_stack([np.interp(et, t, rot[ctrl][:, i]) for i in range(4)])
            q = normalize_quat(q)
            controllers.append(
                {
                    "Controller": ctrl,
                    "TargetsPosition": "1",
                    "TargetsRotation": "1",
                    "ControlPosition": "1",
                    "ControlRotation": "1",
                    "X": self._curve(et, np.interp(et, t, pos[ctrl]["x"])),
                    "Y": self._curve(et, np.interp(et, t, pos[ctrl]["y"])),
                    "Z": self._curve(et, np.interp(et, t, pos[ctrl]["z"])),
                    "RotX": self._curve(et, q[:, 0]),
                    "RotY": self._curve(et, q[:, 1]),
                    "RotZ": self._curve(et, q[:, 2]),
                    "RotW": self._curve(et, q[:, 3]),
                }
            )
        return {
            "SerializeVersion": "283",
            "SerializeMode": "1",
            "AtomType": "Person",
            "GeneratedBy": "vam_behavioral_director.py",
            "SourceContext": context,
            "BehaviorSegments": [s.__dict__ for s in generated["segments"]],
            "SecondaryEvents": [e.__dict__ for e in generated["events"]],
            "Clips": [
                {
                    "AnimationName": f"Behavioral AI {context}",
                    "AnimationLength": self._f(duration),
                    "BlendDuration": "1",
                    "Loop": "1",
                    "PreserveLastFrame": "0",
                    "LoopSelfBlendDuration": "0",
                    "NextAnimationRandomizeWeight": "1",
                    "AutoTransitionPrevious": "0",
                    "AutoTransitionNext": "0",
                    "SyncTransitionTime": "1",
                    "SyncTransitionTimeNL": "0",
                    "EnsureQuaternionContinuity": "1",
                    "AnimationLayer": "Main",
                    "Speed": "1",
                    "Weight": "1",
                    "Uninterruptible": "0",
                    "AnimationSegment": "BehavioralGenerated",
                    "Controllers": controllers,
                    "FloatParams": [],
                    "Triggers": [],
                }
            ],
        }

    def export(self, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as h:
            json.dump(payload, h, indent=2, ensure_ascii=False)

    @staticmethod
    def _curve(t: np.ndarray, v: np.ndarray) -> list[dict[str, str]]:
        # High-density bake + low-pass smoothing to avoid Unity bezier overshoot jitter.
        if len(v) > 4:
            v = gaussian_filter1d(np.asarray(v, dtype=float), sigma=1.5)
        return [{"t": TimelineExporter._f(tt), "v": TimelineExporter._f(vv), "c": str(CURVE_TYPE_SMOOTH_LOCAL)} for tt, vv in zip(t, v)]

    @staticmethod
    def _f(x: float) -> str:
        return f"{float(x):.6f}".rstrip("0").rstrip(".")


class BehavioralDirectorSystem:
    """High-level orchestrator called by GUI and CLI."""

    def __init__(self, mocap_dir: Path, model_path: Path) -> None:
        self.mocap_dir = mocap_dir
        self.model_path = model_path

    def load_or_learn(self, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        if self.model_path.exists():
            with self.model_path.open("r", encoding="utf-8") as h:
                return json.load(h)
        return BehavioralFeatureExtractor(self.mocap_dir, self.model_path).learn(progress)

    def learn(self, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        return BehavioralFeatureExtractor(self.mocap_dir, self.model_path).learn(progress)

    def generate(self, params: DirectorParameters, output: Path, progress: Callable[[float, str], None] | None = None) -> Path:
        progress = progress or (lambda _v, _m: None)
        model = self.load_or_learn(progress)
        context_name = resolve_context(params.context, list(model["contexts"]))
        context = model["contexts"][context_name]
        progress(0.20, "Planning behavioral states")
        segments = MarkovBrain(context, params).plan()
        progress(0.45, "Synthesizing biomechanical layers")
        generated = BiomechanicalSynthesisEngine(context, params).synthesize(segments)
        anchor = self._resolve_anchor(params.target_anchor)
        generated = self._merge_deltas_with_anchor(generated, anchor)
        progress(0.82, "Assembling Timeline payload")
        payload = TimelineExporter().assemble(generated, context_name)
        TimelineExporter().export(payload, output)
        progress(1.0, f"Exported {output.name}")
        return output

    @staticmethod
    def _resolve_anchor(user_anchor: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        anchor = {
            c: {
                "x": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("x", 0.0)),
                "y": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("y", 1.0)),
                "z": float(FALLBACK_ANCHOR_POSE.get(c, {}).get("z", 0.0)),
                "quat": list(FALLBACK_ANCHOR_POSE.get(c, {}).get("quat", [0.0, 0.0, 0.0, 1.0])),
            }
            for c in KINEMATIC_CONTROLLERS
        }
        if user_anchor:
            for c in KINEMATIC_CONTROLLERS:
                if c not in user_anchor:
                    continue
                for a in AXES:
                    anchor[c][a] = finite_float(user_anchor[c].get(a), anchor[c][a])
                q = np.array(user_anchor[c].get("quat", anchor[c]["quat"]), dtype=float)
                anchor[c]["quat"] = normalize_quat(q).tolist()
        return anchor

    @staticmethod
    def _merge_deltas_with_anchor(generated: dict[str, Any], anchor: dict[str, dict[str, Any]]) -> dict[str, Any]:
        pos = generated["positions"]
        rot = generated["rotations"]
        out_pos: dict[str, dict[str, np.ndarray]] = {}
        out_rot: dict[str, np.ndarray] = {}
        n = len(generated["time"])
        for c in KINEMATIC_CONTROLLERS:
            out_pos[c] = {}
            if c in LOCKED_EXTREMITIES:
                for a in AXES:
                    out_pos[c][a] = np.full(n, float(anchor[c][a]), dtype=float)
                q0 = np.array(anchor[c]["quat"], dtype=float)
                out_rot[c] = np.tile(normalize_quat(q0), (n, 1))
                continue
            for a in AXES:
                out_pos[c][a] = np.asarray(pos[c][a], dtype=float) + float(anchor[c][a])
            aq = np.array(anchor[c]["quat"], dtype=float)
            aq = normalize_quat(aq)
            merged = np.empty_like(rot[c])
            for i in range(n):
                r_base = Rotation.from_quat(aq)
                r_delta = Rotation.from_quat(normalize_quat(rot[c][i]))
                merged[i] = normalize_quat((r_base * r_delta).as_quat())
            out_rot[c] = merged
        return {
            "time": generated["time"],
            "positions": out_pos,
            "rotations": out_rot,
            "segments": generated["segments"],
            "events": generated["events"],
        }


class DirectorApp:
    """customtkinter desktop app with async Generate & Export."""

    def __init__(self, system: BehavioralDirectorSystem) -> None:
        if not GUI_AVAILABLE:
            raise RuntimeError("customtkinter is missing. Install with: python -m pip install customtkinter")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.system = system
        self.queue: queue.Queue[tuple[float, str, dict[str, Any] | None]] = queue.Queue()
        self.target_anchor: dict[str, dict[str, Any]] | None = None
        self.target_anchor_path: Path | None = None
        self.root = ctk.CTk()
        self.root.title("VaM Behavioral AI Director")
        self.root.geometry("820x560")
        self._build()
        self.root.after(100, self._poll)
        self._load_contexts_async()
        self.synthesizer: BehavioralSynthesizer | None = None
        self.bodyaware_synth: BodyAwareSynthesizer | None = None
        self.preset_files: list[Path] = []

    def run(self) -> None:
        self.root.mainloop()

    def _build(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self.root, corner_radius=0)
        head.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(head, text="VaM Behavioral AI Director", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=18, pady=(14, 4))
        ctk.CTkLabel(head, text="True-to-life state-driven biomechanical synthesis").pack(anchor="w", padx=18, pady=(0, 12))

        body = ctk.CTkFrame(self.root)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        body.grid_columnconfigure(1, weight=1)
        self.mode_var = ctk.StringVar(value="Classic")
        self.mode_menu = ctk.CTkOptionMenu(body, values=["Classic", "BodyAware"], variable=self.mode_var, command=self._on_mode_change)
        self._row(body, 0, "Mode", self.mode_menu)
        self.context_var = ctk.StringVar(value="Riding / Cowgirl")
        self.context_menu = ctk.CTkOptionMenu(body, values=["Riding / Cowgirl"], variable=self.context_var)
        self._row(body, 1, "Context Selection", self.context_menu)
        self.duration = self._slider(body, 2, "Duration (sec)", 30, 600, 120)
        self.base_intensity = self._slider(body, 3, "Base Intensity", 0.1, 1.0, 0.65)
        self.playfulness = self._slider(body, 4, "Playfulness", 0.0, 1.0, 0.45)
        self.stamina = self._slider(body, 5, "Stamina", 0.0, 1.0, 0.70)
        self.keyframe_ms = self._slider(body, 6, "Keyframe Abstand (ms)", 100, 1000, 600)

        anchor_box = ctk.CTkFrame(body)
        anchor_box.grid(row=7, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 8))
        anchor_box.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(anchor_box, text="Base Pose Anchor (Optional)").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.preset_var = ctk.StringVar(value="(kein Preset)")
        self.preset_menu = ctk.CTkOptionMenu(anchor_box, values=["(kein Preset)"], variable=self.preset_var, command=self._on_preset_selected)
        self.preset_menu.grid(row=0, column=1, sticky="w", padx=10, pady=8)
        self.anchor_path_var = ctk.StringVar(value="No pose loaded (using fallback anchor)")
        ctk.CTkLabel(anchor_box, textvariable=self.anchor_path_var, anchor="w").grid(row=0, column=2, sticky="ew", padx=10, pady=8)
        self.btn_load_pose = ctk.CTkButton(anchor_box, text="Load VaM Pose (.vap/.json)", command=self._load_pose_anchor)
        self.btn_load_pose.grid(row=0, column=3, sticky="e", padx=10, pady=8)

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.grid(row=8, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 8))
        self.btn_learn = ctk.CTkButton(row, text="Learn Behavior Model", command=self._learn_async)
        self.btn_learn.pack(side="left", padx=(0, 10))
        self.btn_gen = ctk.CTkButton(row, text="Generate & Export", command=self._generate_async, fg_color="green")
        self.btn_gen.pack(side="left")

        self.progress = ctk.CTkProgressBar(body)
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew", padx=14, pady=(8, 6))
        self.progress.set(0.0)
        self.status = ctk.CTkLabel(body, text="Ready")
        self.status.grid(row=10, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 10))
        self._refresh_preset_menu()

    def _on_mode_change(self, _selected: str) -> None:
        self._load_contexts_async()

    def _refresh_preset_menu(self) -> None:
        self.preset_files = sorted(DEFAULT_PRESET_DIR.glob("*.vap"))
        labels = ["(kein Preset)"] + [p.name for p in self.preset_files]
        self.preset_menu.configure(values=labels)
        if any(p.name == "Preset_Cowgirl.vap" for p in self.preset_files):
            self.preset_var.set("Preset_Cowgirl.vap")
            self._on_preset_selected("Preset_Cowgirl.vap")

    def _on_preset_selected(self, selected: str) -> None:
        if selected == "(kein Preset)":
            self.target_anchor = None
            self.target_anchor_path = None
            self.anchor_path_var.set("No pose loaded (using fallback anchor)")
            return
        path = next((p for p in self.preset_files if p.name == selected), None)
        if path is None:
            return
        try:
            self.target_anchor = parse_vam_pose_anchor(path)
            self.target_anchor_path = path
            self.anchor_path_var.set(str(path))
            self.status.configure(text=f"Preset geladen: {path.name}")
        except Exception as exc:
            self.status.configure(text=f"Preset load failed: {exc}")

    def _load_pose_anchor(self) -> None:
        if filedialog is None:
            return
        selected = filedialog.askopenfilename(
            title="Load VaM Pose Anchor JSON",
            filetypes=[("VaM Pose", "*.vap *.json"), ("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            parsed = parse_vam_pose_anchor(Path(selected))
            self.target_anchor = parsed
            self.target_anchor_path = Path(selected)
            self.anchor_path_var.set(str(self.target_anchor_path))
            self.status.configure(text="Pose anchor loaded")
        except Exception as exc:
            if messagebox:
                messagebox.showerror("VaM Director", f"Failed to load pose: {exc}")
            self.status.configure(text=f"Pose load failed: {exc}")

    def _row(self, parent: Any, row: int, label: str, widget: Any) -> None:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=14, pady=10)
        widget.grid(row=row, column=1, sticky="ew", padx=14, pady=10)

    def _slider(self, parent: Any, row: int, label: str, lo: float, hi: float, start: float) -> ctk.CTkSlider:
        var = ctk.DoubleVar(value=start)
        val_label = ctk.CTkLabel(parent, text=f"{start:.2f}")

        def on_change(v: float) -> None:
            val_label.configure(text=f"{float(v):.2f}")

        s = ctk.CTkSlider(parent, from_=lo, to=hi, variable=var, command=on_change)
        s.variable = var  # type: ignore[attr-defined]
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=14, pady=10)
        s.grid(row=row, column=1, sticky="ew", padx=14, pady=10)
        val_label.grid(row=row, column=2, sticky="e", padx=14, pady=10)
        return s

    def _load_contexts_async(self) -> None:
        mode = str(self.mode_var.get())
        self._set_busy(True)
        threading.Thread(target=self._load_contexts_worker, args=(mode,), daemon=True).start()

    def _load_contexts_worker(self, mode: str) -> None:
        try:
            if mode == "BodyAware":
                if not DEFAULT_BODYAWARE_MODEL_PATH.exists():
                    self.queue.put((0.05, "BodyAware scan startet (erstellt Modell)...", None))
                    BodyAwarenessScanner(fps=60.0, chunk_fps=15.0).learn(self.system.mocap_dir, DEFAULT_BODYAWARE_MODEL_PATH)
                if DEFAULT_BODYAWARE_MODEL_PATH.exists():
                    with DEFAULT_BODYAWARE_MODEL_PATH.open("r", encoding="utf-8") as h:
                        bam = json.load(h)
                    contexts = list(bam.get("contexts", {}).keys()) or ["Cowgirl"]
                else:
                    contexts = ["Cowgirl", "Missionary", "Oral", "Grinding"]
                self.queue.put((1.0, "BodyAware model ready", {"contexts": contexts}))
            else:
                model = self.system.load_or_learn(lambda v, m: self.queue.put((v, m, None)))
                self.queue.put((1.0, "Model ready", {"contexts": list(model["contexts"])}))
        except Exception as exc:
            self.queue.put((0.0, f"Model load failed: {exc}", None))

    def _learn_async(self) -> None:
        mode = str(self.mode_var.get())
        self._set_busy(True)
        threading.Thread(target=self._learn_worker, args=(mode,), daemon=True).start()

    def _learn_worker(self, mode: str) -> None:
        try:
            if mode == "BodyAware":
                self.queue.put((0.05, "BodyAware Analyse laeuft (Deep Scan)...", None))
                BodyAwarenessScanner(fps=60.0, chunk_fps=15.0).learn(
                    self.system.mocap_dir,
                    DEFAULT_BODYAWARE_MODEL_PATH,
                )
                if DEFAULT_BODYAWARE_MODEL_PATH.exists():
                    with DEFAULT_BODYAWARE_MODEL_PATH.open("r", encoding="utf-8") as h:
                        bam = json.load(h)
                    contexts = list(bam.get("contexts", {}).keys()) or ["Cowgirl"]
                else:
                    contexts = ["Cowgirl", "Missionary", "Oral", "Grinding"]
                self.queue.put((1.0, "BodyAware model learned", {"contexts": contexts}))
            else:
                model = self.system.learn(lambda v, m: self.queue.put((v, m, None)))
                self.queue.put((1.0, "Behavior model learned", {"contexts": list(model["contexts"])}))
        except Exception as exc:
            self.queue.put((0.0, f"Learning failed: {exc}", None))

    def _generate_async(self) -> None:
        if filedialog is None:
            return
        output = filedialog.asksaveasfilename(
            title="Export VaM Timeline",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="behavioral_generated_scene.json",
        )
        if not output:
            return
        mode = str(self.mode_var.get())
        keyframe_ms = float(self.keyframe_ms.variable.get())  # type: ignore[attr-defined]
        params = DirectorParameters(
            context=self.context_var.get(),
            duration=float(self.duration.variable.get()),  # type: ignore[attr-defined]
            base_intensity=float(self.base_intensity.variable.get()),  # type: ignore[attr-defined]
            playfulness=float(self.playfulness.variable.get()),  # type: ignore[attr-defined]
            stamina=float(self.stamina.variable.get()),  # type: ignore[attr-defined]
            seed=42,
            target_anchor=self.target_anchor,
        )
        self._set_busy(True)
        threading.Thread(
            target=self._generate_worker,
            args=(params, Path(output), mode, keyframe_ms),
            daemon=True,
        ).start()

    def _generate_worker(self, params: DirectorParameters, output: Path, mode: str, keyframe_ms: float) -> None:
        try:
            output_file = str(output)
            if mode == "BodyAware":
                if not DEFAULT_BODYAWARE_MODEL_PATH.exists():
                    BodyAwarenessScanner(fps=60.0, chunk_fps=15.0).learn(self.system.mocap_dir, DEFAULT_BODYAWARE_MODEL_PATH)
                export_fps = 1000.0 / keyframe_ms
                self.bodyaware_synth = BodyAwareSynthesizer(
                    DEFAULT_BODYAWARE_MODEL_PATH,
                    internal_fps=60.0,
                    export_fps=export_fps,
                    seed=params.seed,
                )
                generated = self.bodyaware_synth.synthesize(float(params.duration), params.context)
                resolved_anchor = BehavioralDirectorSystem._resolve_anchor(self.target_anchor)
                generated = self._apply_anchor_to_generated(generated, resolved_anchor)
                self.bodyaware_synth.export_timeline(generated, Path(output_file), atom_type="Person")
            else:
                self.system.load_or_learn(lambda v, m: self.queue.put((v * 0.35, m, None)))
                if self.synthesizer is None:
                    self.synthesizer = BehavioralSynthesizer(str(self.system.model_path))
                duration = float(params.duration)
                intensity = float(params.base_intensity)
                playfulness = float(params.playfulness)
                self.synthesizer.set_keyframe_interval_ms(keyframe_ms)
                self.synthesizer.set_base_anchor(self.target_anchor)
                timeline = self.synthesizer.generate_session(duration, intensity, playfulness, context=params.context)
                self.synthesizer.export_vamtline(timeline, output_file)
            self.queue.put((1.0, f"Erfolg! Datei gespeichert unter: {output_file}", None))
        except Exception as exc:
            self.queue.put((0.0, f"Generation failed: {exc}", None))

    @staticmethod
    def _apply_anchor_to_generated(
        generated: dict[str, Any], anchor: dict[str, dict[str, Any]] | None
    ) -> dict[str, Any]:
        if not anchor:
            return generated
        out = generated
        ctrls = list(out.get("positions", {}).keys())
        for c in ctrls:
            if c not in anchor:
                continue
            bx = float(anchor[c].get("x", 0.0))
            by = float(anchor[c].get("y", 0.0))
            bz = float(anchor[c].get("z", 0.0))
            out["positions"][c]["x"] = np.asarray(out["positions"][c]["x"], dtype=float) + bx
            out["positions"][c]["y"] = np.asarray(out["positions"][c]["y"], dtype=float) + by
            out["positions"][c]["z"] = np.asarray(out["positions"][c]["z"], dtype=float) + bz
            if c in out.get("rotations", {}):
                aq = normalize_quat(np.asarray(anchor[c].get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=float))
                base = Rotation.from_quat(np.tile(aq, (len(out["time"]), 1)))
                delta = Rotation.from_quat(normalize_quat(np.asarray(out["rotations"][c], dtype=float)))
                out["rotations"][c] = normalize_quat((base * delta).as_quat())
        DirectorApp._stabilize_lower_body(out, anchor)
        DirectorApp._solve_leg_rotations_from_chain(out, anchor)
        return out

    @staticmethod
    def _stabilize_lower_body(out: dict[str, Any], anchor: dict[str, dict[str, Any]]) -> None:
        """Clamp only extreme lower-body outliers; do not freeze normal motion."""
        max_dev = {
            "lFootControl": np.array([0.22, 0.20, 0.22], dtype=float),
            "rFootControl": np.array([0.22, 0.20, 0.22], dtype=float),
            "lKneeControl": np.array([0.20, 0.18, 0.20], dtype=float),
            "rKneeControl": np.array([0.20, 0.18, 0.20], dtype=float),
            "lThighControl": np.array([0.18, 0.16, 0.18], dtype=float),
            "rThighControl": np.array([0.18, 0.16, 0.18], dtype=float),
        }
        for c, limit in max_dev.items():
            if c not in out.get("positions", {}) or c not in anchor:
                continue
            ax = float(anchor[c].get("x", 0.0))
            ay = float(anchor[c].get("y", 0.0))
            az = float(anchor[c].get("z", 0.0))
            x = np.asarray(out["positions"][c]["x"], dtype=float)
            y = np.asarray(out["positions"][c]["y"], dtype=float)
            z = np.asarray(out["positions"][c]["z"], dtype=float)
            out["positions"][c]["x"] = ax + np.clip(x - ax, -limit[0], limit[0])
            out["positions"][c]["y"] = ay + np.clip(y - ay, -limit[1], limit[1])
            out["positions"][c]["z"] = az + np.clip(z - az, -limit[2], limit[2])

    @staticmethod
    def _solve_leg_rotations_from_chain(out: dict[str, Any], anchor: dict[str, dict[str, Any]]) -> None:
        """Derive foot swing rotation from knee->foot vector to avoid twisted feet."""
        for side in ("l", "r"):
            knee = f"{side}KneeControl"
            foot = f"{side}FootControl"
            if knee not in out.get("positions", {}) or foot not in out.get("positions", {}):
                continue
            if knee not in anchor or foot not in anchor:
                continue
            base_k = np.array([anchor[knee]["x"], anchor[knee]["y"], anchor[knee]["z"]], dtype=float)
            base_f = np.array([anchor[foot]["x"], anchor[foot]["y"], anchor[foot]["z"]], dtype=float)
            base_vec = base_f - base_k
            bn = float(np.linalg.norm(base_vec))
            if bn <= 1e-8:
                continue
            base_dir = base_vec / bn
            base_q = normalize_quat(np.asarray(anchor[foot].get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=float))
            n = len(out["time"])
            solved = np.empty((n, 4), dtype=float)
            for i in range(n):
                cur_k = np.array(
                    [
                        out["positions"][knee]["x"][i],
                        out["positions"][knee]["y"][i],
                        out["positions"][knee]["z"][i],
                    ],
                    dtype=float,
                )
                cur_f = np.array(
                    [
                        out["positions"][foot]["x"][i],
                        out["positions"][foot]["y"][i],
                        out["positions"][foot]["z"][i],
                    ],
                    dtype=float,
                )
                cur_vec = cur_f - cur_k
                cn = float(np.linalg.norm(cur_vec))
                if cn <= 1e-8:
                    solved[i] = base_q
                    continue
                cur_dir = cur_vec / cn
                cross = np.cross(base_dir, cur_dir)
                dot = float(np.clip(np.dot(base_dir, cur_dir), -1.0, 1.0))
                c_norm = float(np.linalg.norm(cross))
                if c_norm <= 1e-9:
                    if dot < 0.0:
                        # 180° fallback around stable axis.
                        axis = np.array([0.0, 1.0, 0.0], dtype=float)
                        if abs(np.dot(axis, base_dir)) > 0.9:
                            axis = np.array([1.0, 0.0, 0.0], dtype=float)
                        swing = Rotation.from_rotvec(axis * np.pi)
                    else:
                        swing = Rotation.identity()
                else:
                    axis = cross / c_norm
                    angle = math.acos(dot)
                    swing = Rotation.from_rotvec(axis * angle)
                solved[i] = normalize_quat((swing * Rotation.from_quat(base_q)).as_quat())
            out["rotations"][foot] = solved

    def _poll(self) -> None:
        try:
            while True:
                v, msg, payload = self.queue.get_nowait()
                self.progress.set(float(np.clip(v, 0.0, 1.0)))
                self.status.configure(text=msg)
                if payload and "contexts" in payload:
                    self.context_menu.configure(values=payload["contexts"])
                    self.context_var.set(payload["contexts"][0])
                if v >= 1.0 or "failed" in msg.lower():
                    self._set_busy(False)
                    if messagebox and "failed" in msg.lower():
                        messagebox.showerror("VaM Director", msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.btn_learn.configure(state=state)
        self.btn_gen.configure(state=state)
        self.btn_load_pose.configure(state=state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocaps", type=Path, default=DEFAULT_MOCAP_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path(r"G:\VAM_Fresh\behavioral_generated_scene.vamtline"))
    parser.add_argument("--pose", type=Path, default=None, help="Optional VaM pose JSON used as generation anchor")
    parser.add_argument("--context", default="Riding")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--base-intensity", type=float, default=0.65)
    parser.add_argument("--playfulness", type=float, default=0.45)
    parser.add_argument("--stamina", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    system = BehavioralDirectorSystem(args.mocaps, args.model)
    if args.learn:
        system.learn(lambda v, m: LOGGER.info("%3.0f%% %s", v * 100, m))
        return
    if args.generate:
        target_anchor = parse_vam_pose_anchor(args.pose) if args.pose else None
        params = DirectorParameters(
            context=args.context,
            duration=args.duration,
            base_intensity=args.base_intensity,
            playfulness=args.playfulness,
            stamina=args.stamina,
            seed=args.seed,
            target_anchor=target_anchor,
        )
        out = system.generate(params, args.output, lambda v, m: LOGGER.info("%3.0f%% %s", v * 100, m))
        LOGGER.info("Generated %s", out)
        return
    if not GUI_AVAILABLE:
        raise SystemExit("customtkinter missing. Install with: python -m pip install customtkinter")
    DirectorApp(system).run()


if __name__ == "__main__":
    main()
