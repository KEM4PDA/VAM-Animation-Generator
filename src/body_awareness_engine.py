"""Body-aware mocap scanner + synthesizer for VaM Timeline JSON.

Non-domain-specific full-body motion pipeline:
- learns dense motion chunks from raw Timeline mocaps
- computes body-awareness features and anatomical constraints
- synthesizes context-similar animation via motif stitching
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation, Slerp

from motion_analyzer import MotionResampler, TimelineParser

LOGGER = logging.getLogger("body_awareness_engine")
PROJECT_ROOT = Path(r"G:\VAM_Fresh\VAM-Animation-Generator")

AXES = ("x", "y", "z")
CURVE = 3

CTRL = (
    "hipControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lThighControl",
    "rThighControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
)


def _resample(seg: np.ndarray, n: int) -> np.ndarray:
    if seg.ndim == 1:
        seg = seg[:, None]
    if seg.shape[0] <= 1:
        return np.repeat(seg[:1], n, axis=0)
    x0 = np.linspace(0.0, 1.0, seg.shape[0])
    x1 = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, seg.shape[1]), dtype=float)
    for i in range(seg.shape[1]):
        if seg.shape[0] >= 4:
            out[:, i] = CubicSpline(x0, seg[:, i], bc_type="natural")(x1)
        else:
            out[:, i] = np.interp(x1, x0, seg[:, i])
    return out


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        n = np.linalg.norm(q)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float) if n <= 1e-12 else q / n
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n <= 1e-12] = 1.0
    return q / n


def _series_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 4:
        return 0.0
    sa = np.std(a)
    sb = np.std(b)
    if sa <= 1e-9 or sb <= 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class ChunkMeta:
    posture: str
    hip_leg_consistency: float
    hand_head_consistency: float
    movement_energy: float
    axis_x: float
    axis_y: float
    axis_z: float
    circularity_xz: float
    knee_bend: float


class BodyAwarenessScanner:
    def __init__(self, fps: float = 60.0, chunk_fps: float = 15.0) -> None:
        self.fps = fps
        self.chunk_fps = chunk_fps
        self.parser = TimelineParser(CTRL)
        self.resampler = MotionResampler(fps=fps)

    def learn(self, mocap_dir: Path, out_path: Path) -> dict[str, Any]:
        files = sorted(mocap_dir.rglob("*.json"))
        if not files:
            raise RuntimeError(f"No JSON files in {mocap_dir}")
        model: dict[str, Any] = {
            "version": 1,
            "engine": "body_awareness_engine",
            "sample_fps": self.fps,
            "chunk_fps": self.chunk_fps,
            "motion_library": [],
            "constraints": {},
            "contexts": {},
        }

        lengths: dict[str, list[float]] = {}
        chunk_id = 0
        for fp in files:
            for clip in self.parser.parse_file(fp):
                rs = self.resampler.resample(clip)
                if rs is None or "hipControl" not in rs.signals:
                    continue
                sig = {c: rs.signals[c][list(AXES)].to_numpy(dtype=float) for c in rs.signals}
                hip = sig["hipControl"]
                hip0 = hip[0].copy()
                rel_hip = {c: (v - hip) for c, v in sig.items()}
                rel0 = {c: rel_hip[c][0].copy() for c in rel_hip}

                self._acc_lengths(lengths, sig)
                peaks = self._cycle_peaks(hip[:, 1], self.fps)
                for i in range(len(peaks) - 1):
                    s, e = int(peaks[i]), int(peaks[i + 1])
                    if e - s < 6:
                        continue
                    d_s = (e - s) / self.fps
                    n = max(4, int(round(d_s * self.chunk_fps)))
                    motif: dict[str, Any] = {
                        "id": f"{fp.name}:{clip.clip_name}:{chunk_id}",
                        "source_file": fp.name,
                        "duration_s": float(d_s),
                        "posture": "neutral",
                    }
                    chunk_id += 1

                    # hip trajectory is stored as true delta from clip base pose (frame 0).
                    hip_seg = hip[s:e, :] - hip0[None, :]
                    hip_rr = _resample(hip_seg, n)
                    motif["hipControl"] = {
                        "space": "world_delta",
                        "pos_x": hip_rr[:, 0].tolist(),
                        "pos_y": hip_rr[:, 1].tolist(),
                        "pos_z": hip_rr[:, 2].tolist(),
                        "rot_x": [0.0] * n,
                        "rot_y": [0.0] * n,
                        "rot_z": [0.0] * n,
                        "rot_w": [1.0] * n,
                    }

                    for c in CTRL:
                        if c == "hipControl":
                            continue
                        if c not in rel_hip:
                            continue
                        # secondary controllers are stored as local delta vs clip base pose:
                        # delta_local = (ctrl-hip)_t - (ctrl-hip)_frame0
                        loc = rel_hip[c][s:e, :] - rel0[c][None, :]
                        rr = _resample(loc, n)
                        motif[c] = {
                            "space": "hip_local_delta",
                            "pos_x": rr[:, 0].tolist(),
                            "pos_y": rr[:, 1].tolist(),
                            "pos_z": rr[:, 2].tolist(),
                            "rot_x": [0.0] * n,
                            "rot_y": [0.0] * n,
                            "rot_z": [0.0] * n,
                            "rot_w": [1.0] * n,
                        }
                    meta = self._meta_for_segment(sig, s, e)
                    motif["posture"] = meta.posture
                    motif["awareness"] = {
                        "hip_leg_consistency": meta.hip_leg_consistency,
                        "hand_head_consistency": meta.hand_head_consistency,
                        "movement_energy": meta.movement_energy,
                        "axis_x": meta.axis_x,
                        "axis_y": meta.axis_y,
                        "axis_z": meta.axis_z,
                        "circularity_xz": meta.circularity_xz,
                        "knee_bend": meta.knee_bend,
                    }
                    model["motion_library"].append(motif)

        model["constraints"] = self._final_lengths(lengths)
        model["contexts"] = self._build_contexts(model["motion_library"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(model, f, indent=2, ensure_ascii=False)
        return model

    @staticmethod
    def _cycle_peaks(y: np.ndarray, fps: float) -> np.ndarray:
        yy = y - np.mean(y)
        min_dist = max(2, int(round(fps * 0.2)))
        prom = max(1e-5, float(np.std(yy) * 0.2))
        peaks, _ = find_peaks(yy, distance=min_dist, prominence=prom)
        if len(peaks) < 3:
            peaks = np.arange(0, len(y), max(6, int(round(fps * 0.5))))
        if len(peaks) < 2:
            peaks = np.array([0, len(y) - 1], dtype=int)
        return peaks

    @staticmethod
    def _acc_lengths(lengths: dict[str, list[float]], sig: dict[str, np.ndarray]) -> None:
        def add(k: str, a: str, b: str) -> None:
            if a not in sig or b not in sig:
                return
            d = float(np.linalg.norm(sig[a][0] - sig[b][0]))
            lengths.setdefault(k, []).append(d)

        add("hip_chest", "hipControl", "chestControl")
        add("chest_head", "chestControl", "headControl")
        add("l_thigh_knee", "lThighControl", "lKneeControl")
        add("l_knee_foot", "lKneeControl", "lFootControl")
        add("r_thigh_knee", "rThighControl", "rKneeControl")
        add("r_knee_foot", "rKneeControl", "rFootControl")

    @staticmethod
    def _final_lengths(lengths: dict[str, list[float]]) -> dict[str, float]:
        return {k: float(np.median(v)) for k, v in lengths.items() if v}

    def _meta_for_segment(self, sig: dict[str, np.ndarray], s: int, e: int) -> ChunkMeta:
        hip = sig.get("hipControl")
        chest = sig.get("chestControl")
        head = sig.get("headControl")
        lk = sig.get("lKneeControl")
        rk = sig.get("rKneeControl")
        lh = sig.get("lHandControl")
        rh = sig.get("rHandControl")

        posture = "neutral"
        if chest is not None:
            dz = float(np.mean(chest[s:e, 2] - hip[s:e, 2]))
            if dz > 0.05:
                posture = "lean_back"
            elif dz < -0.05:
                posture = "lean_forward"

        hip_y = hip[s:e, 1] if hip is not None else np.zeros(e - s)
        knee_y = np.zeros_like(hip_y)
        if lk is not None and rk is not None:
            knee_y = 0.5 * (lk[s:e, 1] + rk[s:e, 1])
        hl = _series_corr(hip_y, knee_y)

        hhc = 0.0
        if head is not None and lh is not None and rh is not None:
            dl = np.linalg.norm(lh[s:e, :] - head[s:e, :], axis=1)
            dr = np.linalg.norm(rh[s:e, :] - head[s:e, :], axis=1)
            # stable and plausible hand-head relation gets higher score.
            hhc = float(np.clip(1.0 - (np.std(np.hstack([dl, dr])) / 0.25), 0.0, 1.0))

        en = float(np.mean(np.linalg.norm(np.gradient(hip[s:e, :], axis=0) * self.fps, axis=1)))
        dx = float(np.ptp(hip[s:e, 0]))
        dy = float(np.ptp(hip[s:e, 1]))
        dz = float(np.ptp(hip[s:e, 2]))
        circ = float(np.clip(min(dx, dz) / max(max(dx, dz), 1e-6), 0.0, 1.0))
        kb = 0.0
        if lk is not None and rk is not None and hip is not None:
            kb = float(np.mean(0.5 * ((hip[s:e, 1] - lk[s:e, 1]) + (hip[s:e, 1] - rk[s:e, 1]))))
        return ChunkMeta(posture, hl, hhc, en, dx, dy, dz, circ, kb)

    @staticmethod
    def _build_contexts(motion_library: list[dict[str, Any]]) -> dict[str, Any]:
        if not motion_library:
            return {"neutral": {"chunk_ids": [], "count": 0}}
        feats = []
        ids = []
        for m in motion_library:
            aw = m.get("awareness", {})
            feats.append(
                [
                    float(aw.get("axis_x", 0.0)),
                    float(aw.get("axis_y", 0.0)),
                    float(aw.get("axis_z", 0.0)),
                    float(aw.get("circularity_xz", 0.0)),
                    float(aw.get("movement_energy", 0.0)),
                    float(aw.get("knee_bend", 0.0)),
                ]
            )
            ids.append(str(m.get("id")))
        X = np.asarray(feats, dtype=float)
        X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-6)
        k = int(np.clip(round(math.sqrt(len(motion_library) / 40.0)), 3, 8))
        centers, labels = kmeans2(X, k, minit="points")
        contexts: dict[str, dict[str, Any]] = {}
        for ci in range(k):
            chunk_ids = [ids[i] for i in range(len(ids)) if int(labels[i]) == ci]
            c0 = centers[ci]
            name = BodyAwarenessScanner._name_cluster(c0)
            # Ensure unique names.
            base = name
            n = 2
            while name in contexts:
                name = f"{base}_{n}"
                n += 1
            contexts[name] = {"chunk_ids": chunk_ids, "count": len(chunk_ids), "center": c0.tolist()}
        return contexts

    @staticmethod
    def _name_cluster(c: np.ndarray) -> str:
        # c = [axis_x, axis_y, axis_z, circularity_xz, energy, knee_bend]
        ax = c[:3]
        dom = int(np.argmax(np.abs(ax)))
        if c[3] > 0.45:
            motion = "circular_xz"
        elif dom == 1:
            motion = "vertical"
        elif dom == 2:
            motion = "forward_back"
        else:
            motion = "lateral"
        posture = "high_knee" if c[5] > 0 else "low_knee"
        intensity = "intense" if c[4] > 0 else "steady"
        return f"{motion}_{posture}_{intensity}"


class BodyAwareSynthesizer:
    def __init__(self, model_path: Path, internal_fps: float = 60.0, export_fps: float = 15.0, seed: int | None = None) -> None:
        with model_path.open("r", encoding="utf-8") as f:
            self.model = json.load(f)
        self.internal_fps = internal_fps
        self.export_fps = export_fps
        self.rng = np.random.default_rng(seed)
        self.library = list(self.model.get("motion_library", []))
        self.by_id = {m["id"]: m for m in self.library if "id" in m}
        self.constraints = self.model.get("constraints", {})
        self.contexts = self.model.get("contexts", {})

    def synthesize(self, duration: float, context: str | None = None, blend_frames: int = 10) -> dict[str, Any]:
        n = max(2, int(round(duration * self.internal_fps)) + 1)
        t = np.linspace(0.0, duration, n)
        pos = {c: {a: np.zeros(n, dtype=float) for a in AXES} for c in CTRL}
        rot = {c: np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)) for c in CTRL}

        pool = self._context_pool(context)
        cur = 0
        prev: dict[str, np.ndarray] | None = None
        while cur < n:
            motif = self._pick_best(pool, prev)
            seg = self._decode_motif(motif)
            seg_n = seg["hipControl"].shape[0]
            out_n = max(4, int(round(seg_n * (self.internal_fps / max(1e-6, self.model.get("chunk_fps", 15.0))))))
            end = min(n, cur + out_n)
            ln = end - cur
            if ln < 4:
                break
            self._paste(pos, rot, seg, cur, end, ln, blend_frames)
            prev = {c: np.column_stack([pos[c]["x"][max(0, end - blend_frames):end], pos[c]["y"][max(0, end - blend_frames):end], pos[c]["z"][max(0, end - blend_frames):end]]) for c in CTRL}
            cur = end

        self._apply_body_constraints(pos)
        return {"time": t, "positions": pos, "rotations": {k: _norm_quat(v) for k, v in rot.items()}}

    def _context_pool(self, context: str | None) -> list[dict[str, Any]]:
        if not context:
            return self.library
        ctx = self.model.get("contexts", {}).get(context)
        if ctx:
            ids = ctx.get("chunk_ids", [])
            out = [self.by_id[i] for i in ids if i in self.by_id]
            if out:
                return out
        sem = self._semantic_filter(context)
        return sem if sem else self.library

    @staticmethod
    def _norm_text(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    def _semantic_filter(self, context: str) -> list[dict[str, Any]]:
        """Classic-like semantic context filtering over awareness metrics."""
        c = self._norm_text(context)
        out: list[dict[str, Any]] = []
        for m in self.library:
            aw = m.get("awareness", {})
            axx = float(aw.get("axis_x", 0.0))
            axy = float(aw.get("axis_y", 0.0))
            axz = float(aw.get("axis_z", 0.0))
            circ = float(aw.get("circularity_xz", 0.0))
            knee = float(aw.get("knee_bend", 0.0))
            hhc = float(aw.get("hand_head_consistency", 0.0))
            hlc = float(aw.get("hip_leg_consistency", 0.0))
            energy = float(aw.get("movement_energy", 0.0))

            keep = True
            if "riding" in c or "cowgirl" in c:
                keep = (
                    axy >= max(axx, axz) * 0.85
                    and knee > 0.10
                    and hlc > 0.05
                    and hhc > 0.10
                )
            elif "missionary" in c or "thrust" in c:
                keep = axz >= max(axx, axy) * 0.9 and hlc > 0.0
            elif "grinding" in c or "circular" in c:
                keep = circ > 0.35 and energy > 0.01
            elif "teasing" in c or "warmup" in c:
                keep = energy < np.percentile(
                    np.array([float(x.get("awareness", {}).get("movement_energy", 0.0)) for x in self.library], dtype=float),
                    55,
                )
            if keep:
                out.append(m)
        return out

    def _pick_best(self, pool: list[dict[str, Any]], prev: dict[str, np.ndarray] | None) -> dict[str, Any]:
        if prev is None:
            return pool[int(self.rng.integers(0, len(pool)))]
        best = None
        best_score = -1e9
        for _ in range(min(120, len(pool))):
            m = pool[int(self.rng.integers(0, len(pool)))]
            score = self._transition_score(m, prev)
            if score > best_score:
                best_score = score
                best = m
        return best or pool[0]

    def _transition_score(self, m: dict[str, Any], prev: dict[str, np.ndarray]) -> float:
        score = 0.0
        for c in ("hipControl", "chestControl", "headControl"):
            data = m.get(c)
            if not data:
                continue
            first = np.array([data["pos_x"][0], data["pos_y"][0], data["pos_z"][0]], dtype=float)
            # for non-hip controls, compare in reconstructed absolute chunk-start space.
            if c != "hipControl" and "hipControl" in m:
                first += np.array(
                    [m["hipControl"]["pos_x"][0], m["hipControl"]["pos_y"][0], m["hipControl"]["pos_z"][0]],
                    dtype=float,
                )
            p = prev.get(c)
            if p is None or len(p) == 0:
                continue
            d = float(np.linalg.norm(first - p[-1]))
            score -= d * 3.0
        aw = m.get("awareness", {})
        score += float(aw.get("hip_leg_consistency", 0.0)) * 1.5
        score += float(aw.get("hand_head_consistency", 0.0)) * 1.0
        score += float(aw.get("movement_energy", 0.0)) * 0.05
        # Prefer physically coherent chunks.
        score += float(aw.get("circularity_xz", 0.0)) * 0.35
        return score

    @staticmethod
    def _decode_motif(m: dict[str, Any]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        hip = m.get("hipControl")
        if hip:
            hx = np.asarray(hip.get("pos_x", [0.0]), dtype=float)
            hy = np.asarray(hip.get("pos_y", [0.0]), dtype=float)
            hz = np.asarray(hip.get("pos_z", [0.0]), dtype=float)
            out["hipControl"] = np.column_stack([hx, hy, hz])
        else:
            out["hipControl"] = np.zeros((4, 3), dtype=float)

        for c in CTRL:
            if c == "hipControl":
                continue
            d = m.get(c)
            if not d:
                out[c] = np.zeros_like(out["hipControl"])
                continue
            px = np.asarray(d.get("pos_x", [0.0]), dtype=float)
            py = np.asarray(d.get("pos_y", [0.0]), dtype=float)
            pz = np.asarray(d.get("pos_z", [0.0]), dtype=float)
            rel = np.column_stack([px, py, pz])
            # keep local relation to hip; absolute reconstruction happens in paste.
            out[c] = rel
        return out

    def _paste(
        self,
        pos: dict[str, dict[str, np.ndarray]],
        rot: dict[str, np.ndarray],
        seg: dict[str, np.ndarray],
        s: int,
        e: int,
        ln: int,
        blend: int,
    ) -> None:
        b = min(blend, s, ln - 1)
        for c in CTRL:
            if c == "hipControl":
                continue
            # reconstruct absolute trajectory from hip + local controller offset.
            r = _resample(seg[c], ln)
            hip_r = _resample(seg["hipControl"], ln)
            pos[c]["x"][s:e] = hip_r[:, 0] + r[:, 0]
            pos[c]["y"][s:e] = hip_r[:, 1] + r[:, 1]
            pos[c]["z"][s:e] = hip_r[:, 2] + r[:, 2]
        hip_r = _resample(seg["hipControl"], ln)
        pos["hipControl"]["x"][s:e] = hip_r[:, 0]
        pos["hipControl"]["y"][s:e] = hip_r[:, 1]
        pos["hipControl"]["z"][s:e] = hip_r[:, 2]
        if b <= 0:
            return
        w = np.linspace(0.0, 1.0, b)
        for c in CTRL:
            if c == "hipControl":
                r = hip_r
            else:
                r = _resample(seg[c], ln) + hip_r
            for i in range(b):
                dst = s - b + i
                src = i
                for k, a in enumerate(AXES):
                    old = pos[c][a][dst]
                    new = r[src, k]
                    pos[c][a][dst] = (1.0 - w[i]) * old + w[i] * new
            q_old = _norm_quat(rot[c][s - b : s])
            q_new = _norm_quat(np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (b, 1)))
            for i in range(b):
                rr = Rotation.from_quat(np.vstack([q_old[i], q_new[i]]))
                rot[c][s - b + i] = Slerp([0.0, 1.0], rr)([w[i]])[0].as_quat()

    def _apply_body_constraints(self, pos: dict[str, dict[str, np.ndarray]]) -> None:
        # enforce plausible hip-foot reach with simple clamp projection.
        lmax = float(self.constraints.get("l_thigh_knee", 0.2)) + float(self.constraints.get("l_knee_foot", 0.2))
        rmax = float(self.constraints.get("r_thigh_knee", 0.2)) + float(self.constraints.get("r_knee_foot", 0.2))
        n = len(pos["hipControl"]["x"])
        for i in range(n):
            hip = np.array([pos["hipControl"]["x"][i], pos["hipControl"]["y"][i], pos["hipControl"]["z"][i]], dtype=float)
            lf = np.array([pos["lFootControl"]["x"][i], pos["lFootControl"]["y"][i], pos["lFootControl"]["z"][i]], dtype=float)
            rf = np.array([pos["rFootControl"]["x"][i], pos["rFootControl"]["y"][i], pos["rFootControl"]["z"][i]], dtype=float)
            dl = np.linalg.norm(hip - lf)
            dr = np.linalg.norm(hip - rf)
            corr = 0.0
            if dl > lmax > 1e-6:
                corr = min(corr, -(dl - lmax))
            if dr > rmax > 1e-6:
                corr = min(corr, -(dr - rmax))
            if corr != 0.0:
                pos["hipControl"]["y"][i] += corr
            # Keep hands in plausible reach to chest to avoid detached limbs.
            chest = np.array([pos["chestControl"]["x"][i], pos["chestControl"]["y"][i], pos["chestControl"]["z"][i]], dtype=float)
            for hand_name in ("lHandControl", "rHandControl"):
                hand = np.array([pos[hand_name]["x"][i], pos[hand_name]["y"][i], pos[hand_name]["z"][i]], dtype=float)
                d = float(np.linalg.norm(hand - chest))
                max_reach = 0.95
                if d > max_reach and d > 1e-6:
                    hand = chest + (hand - chest) * (max_reach / d)
                    pos[hand_name]["x"][i], pos[hand_name]["y"][i], pos[hand_name]["z"][i] = hand.tolist()
        for c in CTRL:
            for a in AXES:
                pos[c][a] = gaussian_filter1d(pos[c][a], sigma=1.0)

    def export_timeline(self, generated: dict[str, Any], out_path: Path, atom_type: str = "Person") -> None:
        t = generated["time"]
        dur = float(t[-1])
        et = np.linspace(0.0, dur, max(2, int(round(dur * self.export_fps)) + 1))
        ctrls = []
        for c in CTRL:
            x = np.interp(et, t, generated["positions"][c]["x"])
            y = np.interp(et, t, generated["positions"][c]["y"])
            z = np.interp(et, t, generated["positions"][c]["z"])
            q = generated["rotations"][c]
            qrs = _norm_quat(np.column_stack([np.interp(et, t, q[:, i]) for i in range(4)]))
            ctrls.append(
                {
                    "Controller": c,
                    "TargetsPosition": "1",
                    "TargetsRotation": "1",
                    "ControlPosition": "1",
                    "ControlRotation": "1",
                    "X": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, x)],
                    "Y": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, y)],
                    "Z": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, z)],
                    "RotX": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, qrs[:, 0])],
                    "RotY": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, qrs[:, 1])],
                    "RotZ": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, qrs[:, 2])],
                    "RotW": [{"t": self._f(tt), "v": self._f(v), "c": str(CURVE)} for tt, v in zip(et, qrs[:, 3])],
                }
            )
        payload = {
            "SerializeVersion": "283",
            "SerializeMode": "1",
            "AtomType": atom_type,
            "GeneratedBy": "body_awareness_engine.py",
            "Clips": [
                {
                    "AnimationName": "BodyAware Generated",
                    "AnimationLength": self._f(dur),
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
                    "AnimationSegment": "BodyAware",
                    "Controllers": ctrls,
                    "FloatParams": [],
                    "Triggers": [],
                }
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _f(v: float) -> str:
        return f"{float(v):.6f}".rstrip("0").rstrip(".")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("learn")
    l.add_argument("--mocaps", type=Path, required=True)
    l.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "body_awareness_model.json")
    l.add_argument("--fps", type=float, default=60.0)
    l.add_argument("--chunk-fps", type=float, default=15.0)

    s = sub.add_parser("synthesize")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "bodyaware_generated.json")
    s.add_argument("--duration", type=float, default=120.0)
    s.add_argument("--context", default=None)
    s.add_argument("--internal-fps", type=float, default=60.0)
    s.add_argument("--export-fps", type=float, default=15.0)
    s.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.cmd == "learn":
        model = BodyAwarenessScanner(fps=args.fps, chunk_fps=args.chunk_fps).learn(args.mocaps, args.out)
        LOGGER.info("Learned %d chunks -> %s", len(model.get("motion_library", [])), args.out)
        return
    gen = BodyAwareSynthesizer(args.model, args.internal_fps, args.export_fps, args.seed)
    out = gen.synthesize(args.duration, args.context)
    gen.export_timeline(out, args.out, atom_type="Person")
    LOGGER.info("Generated %s", args.out)


if __name__ == "__main__":
    main()
