"""Analyze Virt-a-Mate Timeline mocap clips and extract motion grammar.

The script reads Timeline JSON files, decodes controller keyframes, resamples
them to a fixed-rate signal, extracts biomechanical features, applies a compact
heuristic classifier, and exports:

* mocap_library_index.json - one row per source clip
* motion_grammar.json - aggregate synthesis grammar per recognized context
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.fft import rfft, rfftfreq
    from scipy.interpolate import CubicSpline, interp1d
    from scipy.signal import correlate, correlation_lags, find_peaks

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    CubicSpline = None
    interp1d = None
    find_peaks = None
    correlate = None
    correlation_lags = None
    SCIPY_AVAILABLE = False

    from numpy.fft import rfft, rfftfreq


LOGGER = logging.getLogger("motion_analyzer")


AXES = ("x", "y", "z")
DEFAULT_CONTROLLERS = (
    "hipControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
)
LEAD_CONTROLLERS = ("hipControl", "headControl")
CURVE_KEYS = {"x": "X", "y": "Y", "z": "Z"}


@dataclass(frozen=True)
class Keyframe:
    """One scalar Timeline keyframe."""

    time: float
    value: float
    curve_type: int = 0


@dataclass
class ControllerTrack:
    """Raw 3D position curves for one VaM free controller."""

    name: str
    curves: dict[str, list[Keyframe]] = field(default_factory=dict)

    def has_position(self) -> bool:
        """Return True when all XYZ position curves contain useful data."""

        return all(len(self.curves.get(axis, [])) > 0 for axis in AXES)

    def time_bounds(self) -> tuple[float, float] | None:
        """Return min/max keyframe time across the position curves."""

        times: list[float] = []
        for axis in AXES:
            times.extend(k.time for k in self.curves.get(axis, []))
        if not times:
            return None
        return min(times), max(times)


@dataclass
class ClipData:
    """Parsed Timeline clip containing selected controller tracks."""

    source_file: Path
    clip_name: str
    duration: float
    serialize_version: int
    controllers: dict[str, ControllerTrack]


@dataclass
class ResampledClip:
    """Fixed-rate representation of a parsed clip."""

    clip: ClipData
    fps: float
    time: np.ndarray
    signals: dict[str, pd.DataFrame]


@dataclass
class ControllerRelation:
    """Phase and amplitude relationship to a lead controller."""

    delay_ms: float | None
    amplitude_ratio: float | None
    correlation: float | None


@dataclass
class FeatureSet:
    """Features extracted from a resampled clip."""

    source_file: str
    clip_name: str
    duration_s: float
    context: str
    lead_controller: str
    dominant_axis: str
    axis_variance_x: float
    axis_variance_y: float
    axis_variance_z: float
    bpm: float | None
    amplitude: float | None
    hip_motion_energy: float | None
    head_motion_energy: float | None
    chest_delay_ms: float | None
    chest_amplitude_ratio: float | None
    chest_correlation: float | None
    head_delay_ms: float | None
    head_amplitude_ratio: float | None
    head_correlation: float | None
    hand_delay_ms: float | None
    hand_amplitude_ratio: float | None
    notes: str = ""


class TimelineParser:
    """Universal parser for VaM Timeline JSON clips."""

    def __init__(self, controllers: Iterable[str] = DEFAULT_CONTROLLERS) -> None:
        self.controllers = set(controllers)

    def parse_file(self, path: Path) -> list[ClipData]:
        """Parse one Timeline JSON file and return all usable clips."""

        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Skipping %s: cannot read JSON (%s)", path, exc)
            return []

        version = self._int_value(payload.get("SerializeVersion"), 0)
        clip_nodes = payload.get("Clips")
        if not isinstance(clip_nodes, list):
            LOGGER.warning("Skipping %s: no Clips array found", path)
            return []

        clips: list[ClipData] = []
        for index, clip_json in enumerate(clip_nodes):
            if not isinstance(clip_json, dict):
                continue
            clip = self._parse_clip(path, clip_json, index, version)
            if clip is not None:
                clips.append(clip)
        return clips

    def _parse_clip(
        self, path: Path, clip_json: dict[str, Any], index: int, version: int
    ) -> ClipData | None:
        clip_name = str(clip_json.get("AnimationName") or f"Clip {index + 1}")
        duration = self._float_value(clip_json.get("AnimationLength"), 0.0)

        controller_nodes = clip_json.get("Controllers")
        if not isinstance(controller_nodes, list):
            return None

        tracks: dict[str, ControllerTrack] = {}
        for controller_json in controller_nodes:
            if not isinstance(controller_json, dict):
                continue
            name = str(controller_json.get("Controller") or controller_json.get("id") or "")
            if name not in self.controllers:
                continue
            track = self._parse_controller(name, controller_json, version)
            if track.has_position():
                tracks[name] = track

        if not tracks:
            LOGGER.info("No selected animated controllers in %s / %s", path.name, clip_name)
            return None

        if duration <= 0:
            bounds = [track.time_bounds() for track in tracks.values()]
            valid_bounds = [bound for bound in bounds if bound is not None]
            if valid_bounds:
                duration = max(bound[1] for bound in valid_bounds)

        return ClipData(path, clip_name, duration, version, tracks)

    def _parse_controller(
        self, name: str, controller_json: dict[str, Any], version: int
    ) -> ControllerTrack:
        curves: dict[str, list[Keyframe]] = {}
        for axis, timeline_key in CURVE_KEYS.items():
            curve_node = controller_json.get(timeline_key)
            curves[axis] = self._decode_curve(curve_node, version)
        return ControllerTrack(name=name, curves=curves)

    def _decode_curve(self, curve_node: Any, version: int) -> list[Keyframe]:
        """Decode modern optimized and older Timeline curve encodings."""

        if curve_node is None:
            return []

        if isinstance(curve_node, list):
            return self._decode_curve_array(curve_node, version)

        if isinstance(curve_node, dict):
            keys = curve_node.get("keys")
            if isinstance(keys, list):
                return self._decode_legacy_class(keys)
            if {"t", "v"} & set(curve_node):
                return self._decode_curve_array([curve_node], version)

        if isinstance(curve_node, str):
            return self._decode_legacy_string(curve_node)

        return []

    def _decode_curve_array(self, nodes: list[Any], version: int) -> list[Keyframe]:
        frames: list[Keyframe] = []
        last_t = -1.0
        last_v = 0.0
        last_c = 0

        for node in nodes:
            try:
                if isinstance(node, str) and version >= 230 and node:
                    frame = self._decode_optimized_keyframe(node, last_v, last_c)
                elif isinstance(node, str) and node:
                    frame = self._decode_optimized_keyframe_legacy(node, last_v, last_c)
                elif isinstance(node, dict):
                    frame = Keyframe(
                        time=self._float_value(node.get("t"), -1.0),
                        value=self._float_value(node.get("v"), last_v),
                        curve_type=self._int_value(node.get("c"), last_c),
                    )
                else:
                    continue
            except Exception:
                continue

            if frame.time < 0 or math.isclose(frame.time, last_t, abs_tol=1e-7):
                continue
            frames.append(frame)
            last_t = frame.time
            last_v = frame.value
            last_c = frame.curve_type

        return frames

    def _decode_optimized_keyframe(self, encoded: str, last_v: float, last_c: int) -> Keyframe:
        encoded_value = ord(encoded[0]) - ord("A")
        has_value = (encoded_value & 1) != 0
        has_curve_type = (encoded_value & 2) != 0

        pos = 1
        time = self._decode_float32_hex(encoded[pos : pos + 8])
        pos += 8
        value = last_v
        if has_value:
            value = self._decode_float32_hex(encoded[pos : pos + 8])
            pos += 8
        curve_type = last_c
        if has_curve_type:
            curve_type = int(encoded[pos : pos + 2], 16)
        return Keyframe(time=time, value=value, curve_type=curve_type)

    def _decode_optimized_keyframe_legacy(
        self, encoded: str, last_v: float, last_c: int
    ) -> Keyframe:
        size_char = encoded[0]
        if "0" <= size_char <= "9":
            index = ord(size_char) - ord("0")
        elif "a" <= size_char <= "z":
            index = ord(size_char) - ord("a") + 10
        else:
            index = ord(size_char) - ord("A") + 36

        t_bytes = index // 25
        index %= 25
        v_bytes = index // 5
        has_c = (index % 5) != 0
        pos = 1
        time = self._decode_float32_hex(encoded[pos : pos + t_bytes * 2])
        pos += t_bytes * 2
        value = last_v if v_bytes == 0 else self._decode_float32_hex(encoded[pos : pos + v_bytes * 2])
        pos += v_bytes * 2
        curve_type = last_c if not has_c else int(encoded[pos : pos + 2], 16)
        return Keyframe(time=time, value=value, curve_type=curve_type)

    def _decode_legacy_class(self, nodes: list[Any]) -> list[Keyframe]:
        frames: list[Keyframe] = []
        last_t = -1.0
        for node in nodes:
            if not isinstance(node, dict):
                continue
            time = self._float_value(node.get("time"), -1.0)
            if time < 0 or math.isclose(time, last_t, abs_tol=1e-7):
                continue
            frames.append(Keyframe(time=time, value=self._float_value(node.get("value"), 0.0)))
            last_t = time
        return frames

    def _decode_legacy_string(self, value: str) -> list[Keyframe]:
        frames: list[Keyframe] = []
        last_t = -1.0
        for chunk in value.split(";"):
            if not chunk:
                continue
            parts = chunk.split(",")
            if len(parts) < 2:
                continue
            time = self._float_value(parts[0], -1.0)
            if time < 0 or math.isclose(time, last_t, abs_tol=1e-7):
                continue
            frames.append(
                Keyframe(
                    time=time,
                    value=self._float_value(parts[1], 0.0),
                    curve_type=self._int_value(parts[2], 0) if len(parts) > 2 else 0,
                )
            )
            last_t = time
        return frames

    @staticmethod
    def _decode_float32_hex(encoded: str) -> float:
        if len(encoded) < 8:
            encoded = encoded.ljust(8, "0")
        return struct.unpack("<f", bytes.fromhex(encoded[:8]))[0]

    @staticmethod
    def _float_value(value: Any, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_value(value: Any, default: int = 0) -> int:
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


class MotionResampler:
    """Resample irregular Timeline controller curves to fixed FPS."""

    def __init__(self, fps: float = 60.0, min_duration_s: float = 0.25) -> None:
        self.fps = fps
        self.min_duration_s = min_duration_s

    def resample(self, clip: ClipData) -> ResampledClip | None:
        """Return a fixed-rate clip or None when the clip is too short."""

        duration = max(clip.duration, self._max_track_time(clip))
        if duration < self.min_duration_s:
            return None

        sample_count = max(2, int(math.floor(duration * self.fps)) + 1)
        time = np.linspace(0.0, duration, sample_count)
        signals: dict[str, pd.DataFrame] = {}

        for name, track in clip.controllers.items():
            columns: dict[str, np.ndarray] = {}
            for axis in AXES:
                values = self._resample_curve(track.curves.get(axis, []), time)
                if values is not None:
                    columns[axis] = values
            if len(columns) == 3:
                signals[name] = pd.DataFrame(columns, index=pd.Index(time, name="time"))

        if not signals:
            return None

        return ResampledClip(clip=clip, fps=self.fps, time=time, signals=signals)

    def _resample_curve(self, frames: list[Keyframe], time: np.ndarray) -> np.ndarray | None:
        if not frames:
            return None

        raw_t = np.array([frame.time for frame in frames], dtype=float)
        raw_v = np.array([frame.value for frame in frames], dtype=float)
        order = np.argsort(raw_t)
        raw_t = raw_t[order]
        raw_v = raw_v[order]

        unique_t, unique_idx = np.unique(raw_t, return_index=True)
        raw_t = unique_t
        raw_v = raw_v[unique_idx]

        if len(raw_t) == 1:
            return np.full_like(time, raw_v[0], dtype=float)

        if SCIPY_AVAILABLE and len(raw_t) >= 4 and CubicSpline is not None:
            spline = CubicSpline(raw_t, raw_v, bc_type="natural", extrapolate=True)
            return np.asarray(spline(time), dtype=float)

        if SCIPY_AVAILABLE and interp1d is not None:
            interpolator = interp1d(raw_t, raw_v, kind="linear", fill_value="extrapolate")
            return np.asarray(interpolator(time), dtype=float)

        return np.interp(time, raw_t, raw_v, left=raw_v[0], right=raw_v[-1])

    @staticmethod
    def _max_track_time(clip: ClipData) -> float:
        bounds = [track.time_bounds() for track in clip.controllers.values()]
        valid = [bound[1] for bound in bounds if bound is not None]
        return max(valid) if valid else 0.0


class FeatureExtractor:
    """Extract biomechanics-inspired motion features from resampled clips."""

    def __init__(self, min_peak_distance_s: float = 0.20) -> None:
        self.min_peak_distance_s = min_peak_distance_s

    def extract(self, resampled: ResampledClip) -> dict[str, Any]:
        """Compute scalar feature dictionary used by the classifier and exporter."""

        signals = resampled.signals
        hip = signals.get("hipControl")
        head = signals.get("headControl")

        lead = self._select_lead_controller(signals)
        lead_df = signals[lead]
        variances = self._axis_variances(lead_df)
        dominant_axis = max(variances, key=variances.get)
        dominant_signal = self._detrend(lead_df[dominant_axis].to_numpy(dtype=float))

        bpm = self._estimate_bpm(dominant_signal, resampled.fps)
        amplitude = self._estimate_amplitude(dominant_signal, resampled.fps)

        hip_energy = self._motion_energy(hip) if hip is not None else None
        head_energy = self._motion_energy(head) if head is not None else None

        relations: dict[str, ControllerRelation] = {}
        for follower in ("chestControl", "headControl", "lHandControl", "rHandControl"):
            if follower not in signals or follower == lead:
                continue
            relation_axis = dominant_axis
            if follower.startswith(("lHand", "rHand")) and lead == "headControl":
                relation_axis = self._dominant_axis(signals[follower])
            relations[follower] = self._relation(
                lead_df[dominant_axis].to_numpy(dtype=float),
                signals[follower][relation_axis].to_numpy(dtype=float),
                resampled.fps,
            )

        hand_relations = [
            rel for name, rel in relations.items() if name in {"lHandControl", "rHandControl"}
        ]
        hand_delay = self._nanmean([rel.delay_ms for rel in hand_relations])
        hand_ratio = self._nanmean([rel.amplitude_ratio for rel in hand_relations])

        return {
            "lead_controller": lead,
            "dominant_axis": dominant_axis,
            "axis_variance_x": variances["x"],
            "axis_variance_y": variances["y"],
            "axis_variance_z": variances["z"],
            "bpm": bpm,
            "amplitude": amplitude,
            "hip_motion_energy": hip_energy,
            "head_motion_energy": head_energy,
            "chest_relation": relations.get("chestControl"),
            "head_relation": relations.get("headControl"),
            "hand_delay_ms": hand_delay,
            "hand_amplitude_ratio": hand_ratio,
        }

    def _select_lead_controller(self, signals: dict[str, pd.DataFrame]) -> str:
        energies = {
            name: self._motion_energy(df)
            for name, df in signals.items()
            if name in LEAD_CONTROLLERS
        }
        if not energies:
            return next(iter(signals))
        hip_energy = energies.get("hipControl", 0.0)
        head_energy = energies.get("headControl", 0.0)
        if head_energy > hip_energy * 3.0 and hip_energy < 0.035:
            return "headControl"
        return "hipControl" if "hipControl" in energies else max(energies, key=energies.get)

    @staticmethod
    def _axis_variances(df: pd.DataFrame) -> dict[str, float]:
        return {axis: float(np.nanvar(df[axis].to_numpy(dtype=float))) for axis in AXES}

    def _dominant_axis(self, df: pd.DataFrame) -> str:
        variances = self._axis_variances(df)
        return max(variances, key=variances.get)

    @staticmethod
    def _motion_energy(df: pd.DataFrame | None) -> float:
        if df is None or df.empty:
            return 0.0
        centered = df[list(AXES)].to_numpy(dtype=float)
        centered = centered - np.nanmean(centered, axis=0)
        return float(np.sqrt(np.nanmean(np.sum(centered * centered, axis=1))))

    @staticmethod
    def _detrend(signal: np.ndarray) -> np.ndarray:
        if signal.size == 0:
            return signal
        return signal - np.nanmean(signal)

    def _estimate_bpm(self, signal: np.ndarray, fps: float) -> float | None:
        signal = np.asarray(signal, dtype=float)
        if signal.size < max(8, int(fps)):
            return None
        if float(np.nanstd(signal)) <= 1e-8:
            return None

        peak_bpm = self._estimate_bpm_peaks(signal, fps)
        fft_bpm = self._estimate_bpm_fft(signal, fps)
        if peak_bpm is not None and 20.0 <= peak_bpm <= 360.0:
            return peak_bpm
        return fft_bpm

    def _estimate_bpm_peaks(self, signal: np.ndarray, fps: float) -> float | None:
        distance = max(1, int(self.min_peak_distance_s * fps))
        prominence = max(float(np.nanstd(signal)) * 0.25, 1e-6)
        if SCIPY_AVAILABLE and find_peaks is not None:
            peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
        else:
            peaks = self._fallback_find_peaks(signal, distance)
        if len(peaks) < 2:
            return None
        intervals = np.diff(peaks) / fps
        intervals = intervals[intervals > 1e-6]
        if len(intervals) == 0:
            return None
        return float(60.0 / np.median(intervals))

    @staticmethod
    def _fallback_find_peaks(signal: np.ndarray, distance: int) -> np.ndarray:
        candidates = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]))[0] + 1
        if len(candidates) == 0:
            return candidates
        selected = [int(candidates[0])]
        for candidate in candidates[1:]:
            if candidate - selected[-1] >= distance:
                selected.append(int(candidate))
            elif signal[candidate] > signal[selected[-1]]:
                selected[-1] = int(candidate)
        return np.asarray(selected, dtype=int)

    @staticmethod
    def _estimate_bpm_fft(signal: np.ndarray, fps: float) -> float | None:
        windowed = signal * np.hanning(signal.size)
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(signal.size, d=1.0 / fps)
        mask = (freqs >= 0.25) & (freqs <= 6.0)
        if not np.any(mask):
            return None
        idx = int(np.argmax(spectrum[mask]))
        freq = freqs[mask][idx]
        if freq <= 0:
            return None
        return float(freq * 60.0)

    def _estimate_amplitude(self, signal: np.ndarray, fps: float) -> float | None:
        if signal.size < 3:
            return None
        distance = max(1, int(self.min_peak_distance_s * fps))
        if SCIPY_AVAILABLE and find_peaks is not None:
            maxima, _ = find_peaks(signal, distance=distance)
            minima, _ = find_peaks(-signal, distance=distance)
        else:
            maxima = self._fallback_find_peaks(signal, distance)
            minima = self._fallback_find_peaks(-signal, distance)

        if len(maxima) > 0 and len(minima) > 0:
            high = float(np.nanmedian(signal[maxima]))
            low = float(np.nanmedian(signal[minima]))
            return abs(high - low)
        return float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))

    def _relation(self, lead: np.ndarray, follower: np.ndarray, fps: float) -> ControllerRelation:
        lead = self._detrend(lead)
        follower = self._detrend(follower)
        lead_amp = self._estimate_amplitude(lead, fps)
        follower_amp = self._estimate_amplitude(follower, fps)
        ratio = None
        if lead_amp is not None and lead_amp > 1e-8 and follower_amp is not None:
            ratio = float(follower_amp / lead_amp)

        if lead.size != follower.size or lead.size < 4:
            return ControllerRelation(None, ratio, None)
        if float(np.nanstd(lead)) <= 1e-8 or float(np.nanstd(follower)) <= 1e-8:
            return ControllerRelation(None, ratio, None)

        if SCIPY_AVAILABLE and correlate is not None and correlation_lags is not None:
            corr = correlate(follower, lead, mode="full", method="auto")
            lags = correlation_lags(follower.size, lead.size, mode="full")
        else:
            corr = np.correlate(follower, lead, mode="full")
            lags = np.arange(-lead.size + 1, follower.size)

        best = int(np.argmax(np.abs(corr)))
        lag_samples = int(lags[best])
        delay_ms = float(lag_samples / fps * 1000.0)
        denom = np.linalg.norm(lead) * np.linalg.norm(follower)
        corr_coeff = float(corr[best] / denom) if denom > 1e-12 else None
        return ControllerRelation(delay_ms, ratio, corr_coeff)

    @staticmethod
    def _nanmean(values: Iterable[float | None]) -> float | None:
        valid = [value for value in values if value is not None and np.isfinite(value)]
        return float(np.mean(valid)) if valid else None


class MotionClassifier:
    """Heuristic context classifier based on transform-only features."""

    def classify(self, features: dict[str, Any]) -> str:
        """Assign one motion context label from extracted features."""

        lead = features["lead_controller"]
        axis = features["dominant_axis"]
        vx = features["axis_variance_x"]
        vy = features["axis_variance_y"]
        vz = features["axis_variance_z"]
        hip_energy = features.get("hip_motion_energy") or 0.0
        head_energy = features.get("head_motion_energy") or 0.0
        amplitude = features.get("amplitude") or 0.0

        total_var = max(vx + vy + vz, 1e-12)
        x_share = vx / total_var
        y_share = vy / total_var
        z_share = vz / total_var

        if hip_energy < 0.01 and head_energy < 0.01:
            return "Stationary / Pose"

        if lead == "headControl" and hip_energy < 0.035 and head_energy > max(hip_energy * 2.5, 0.035):
            return "Blowjob / Oral"

        if x_share > 0.24 and z_share > 0.24 and (x_share + z_share) > y_share * 1.4:
            return "Grinding / Circular"

        if axis == "y" and y_share >= 0.45:
            return "Riding / Cowgirl"

        if axis == "z" and z_share >= 0.42:
            return "Missionary / Thrusting"

        if axis == "x" and x_share >= 0.45:
            return "Side-to-Side / Lateral"

        if amplitude < 0.025:
            return "Subtle / Idle Motion"

        return "Mixed / Ambiguous"


class MotionGrammarExporter:
    """Build library and aggregate grammar exports."""

    def build_feature_set(
        self, resampled: ResampledClip, features: dict[str, Any], context: str
    ) -> FeatureSet:
        """Convert raw feature dict into a typed row object."""

        chest = features.get("chest_relation") or ControllerRelation(None, None, None)
        head = features.get("head_relation") or ControllerRelation(None, None, None)
        clip = resampled.clip
        return FeatureSet(
            source_file=str(clip.source_file),
            clip_name=clip.clip_name,
            duration_s=float(resampled.time[-1]) if len(resampled.time) else clip.duration,
            context=context,
            lead_controller=features["lead_controller"],
            dominant_axis=features["dominant_axis"],
            axis_variance_x=features["axis_variance_x"],
            axis_variance_y=features["axis_variance_y"],
            axis_variance_z=features["axis_variance_z"],
            bpm=features["bpm"],
            amplitude=features["amplitude"],
            hip_motion_energy=features["hip_motion_energy"],
            head_motion_energy=features["head_motion_energy"],
            chest_delay_ms=chest.delay_ms,
            chest_amplitude_ratio=chest.amplitude_ratio,
            chest_correlation=chest.correlation,
            head_delay_ms=head.delay_ms,
            head_amplitude_ratio=head.amplitude_ratio,
            head_correlation=head.correlation,
            hand_delay_ms=features.get("hand_delay_ms"),
            hand_amplitude_ratio=features.get("hand_amplitude_ratio"),
            notes="" if SCIPY_AVAILABLE else "SciPy unavailable: linear interpolation/fallback peaks used",
        )

    def to_dataframe(self, rows: list[FeatureSet]) -> pd.DataFrame:
        """Return the internal pandas feature table."""

        return pd.DataFrame([row.__dict__ for row in rows])

    def export_library(self, df: pd.DataFrame, output_path: Path) -> None:
        """Write per-file classification and metrics JSON."""

        records = self._records_for_json(df)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)

    def export_grammar(self, df: pd.DataFrame, output_path: Path) -> None:
        """Write aggregate context grammar JSON."""

        grammar: dict[str, Any] = {}
        if df.empty:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(grammar, handle, indent=2)
            return

        for context, group in df.groupby("context", dropna=False):
            bpm = self._range(group["bpm"])
            amplitude = self._range(group["amplitude"])
            dominant_axis = self._mode(group["dominant_axis"])
            lead_controller = self._mode(group["lead_controller"])

            grammar[str(context)] = {
                "sample_count": int(len(group)),
                "lead_controller": lead_controller,
                "dominant_axis": dominant_axis,
                "frequency_bpm": {"min": bpm[0], "max": bpm[1], "mean": self._mean(group["bpm"])},
                "amplitude": {
                    "min": amplitude[0],
                    "max": amplitude[1],
                    "mean": self._mean(group["amplitude"]),
                },
                "axis_variance_share": self._axis_share(group),
                "sub_controllers": {
                    "chestControl": {
                        "delay_ms_mean": self._mean(group["chest_delay_ms"]),
                        "delay_ms_range": self._range_dict(group["chest_delay_ms"]),
                        "amplitude_ratio_mean": self._mean(group["chest_amplitude_ratio"]),
                        "correlation_mean": self._mean(group["chest_correlation"]),
                    },
                    "headControl": {
                        "delay_ms_mean": self._mean(group["head_delay_ms"]),
                        "delay_ms_range": self._range_dict(group["head_delay_ms"]),
                        "amplitude_ratio_mean": self._mean(group["head_amplitude_ratio"]),
                        "correlation_mean": self._mean(group["head_correlation"]),
                    },
                    "hands": {
                        "delay_ms_mean": self._mean(group["hand_delay_ms"]),
                        "delay_ms_range": self._range_dict(group["hand_delay_ms"]),
                        "amplitude_ratio_mean": self._mean(group["hand_amplitude_ratio"]),
                    },
                },
                "source_files": sorted(group["source_file"].astype(str).unique().tolist()),
            }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(grammar, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
        records = df.replace({np.nan: None}).to_dict(orient="records")
        return records

    @staticmethod
    def _mean(series: pd.Series) -> float | None:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return float(numeric.mean()) if not numeric.empty else None

    @staticmethod
    def _range(series: pd.Series) -> tuple[float | None, float | None]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return None, None
        return float(numeric.min()), float(numeric.max())

    def _range_dict(self, series: pd.Series) -> dict[str, float | None]:
        low, high = self._range(series)
        return {"min": low, "max": high}

    @staticmethod
    def _mode(series: pd.Series) -> str | None:
        clean = series.dropna()
        if clean.empty:
            return None
        return str(clean.mode().iloc[0])

    @staticmethod
    def _axis_share(group: pd.DataFrame) -> dict[str, float | None]:
        sums = {
            axis: float(pd.to_numeric(group[f"axis_variance_{axis}"], errors="coerce").sum())
            for axis in AXES
        }
        total = sum(sums.values())
        if total <= 1e-12:
            return {axis: None for axis in AXES}
        return {axis: value / total for axis, value in sums.items()}


class MotionAnalyzer:
    """End-to-end pipeline orchestrator."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        fps: float = 60.0,
        controllers: Iterable[str] = DEFAULT_CONTROLLERS,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.parser = TimelineParser(controllers)
        self.resampler = MotionResampler(fps=fps)
        self.extractor = FeatureExtractor()
        self.classifier = MotionClassifier()
        self.exporter = MotionGrammarExporter()

    def run(self) -> pd.DataFrame:
        """Run the complete analysis and write JSON exports."""

        rows: list[FeatureSet] = []
        files = sorted(self.input_dir.rglob("*.json"))
        LOGGER.info("Found %d JSON files in %s", len(files), self.input_dir)

        for path in files:
            for clip in self.parser.parse_file(path):
                try:
                    resampled = self.resampler.resample(clip)
                    if resampled is None:
                        continue
                    features = self.extractor.extract(resampled)
                    context = self.classifier.classify(features)
                    rows.append(self.exporter.build_feature_set(resampled, features, context))
                except Exception as exc:
                    LOGGER.warning("Failed to analyze %s / %s: %s", path.name, clip.clip_name, exc)

        df = self.exporter.to_dataframe(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exporter.export_library(df, self.output_dir / "mocap_library_index.json")
        self.exporter.export_grammar(df, self.output_dir / "motion_grammar.json")
        df.to_csv(self.output_dir / "mocap_feature_table.csv", index=False)
        return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("SavedMocaps"),
        help="Directory containing VaM Timeline JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Directory for mocap_library_index.json and motion_grammar.json.",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Resampling rate in Hz.")
    parser.add_argument(
        "--controllers",
        nargs="*",
        default=list(DEFAULT_CONTROLLERS),
        help="Controller names to parse. hip/chest/head are recommended.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not SCIPY_AVAILABLE:
        LOGGER.warning(
            "SciPy is not installed; using linear interpolation, numpy FFT, and fallback peak detection."
        )

    analyzer = MotionAnalyzer(
        input_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        controllers=args.controllers,
    )
    df = analyzer.run()
    LOGGER.info("Analyzed %d clip(s).", len(df))
    if not df.empty:
        LOGGER.info("Context counts:\n%s", df["context"].value_counts().to_string())


if __name__ == "__main__":
    main()
