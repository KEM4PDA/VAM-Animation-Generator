import numpy as np
import json
from typing import Any, Dict
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


class MocapSampler:
    """Lädt die rohen Mocap-Deltas aus der Motif-Datenbank."""

    def __init__(self, model_data: Dict[str, Any], context_name: str | None = None):
        self.model = model_data
        self.context_name = context_name
        self.motifs = model_data.get("motion_library", [])
        if not self.motifs:
            self.motifs = self._build_from_motion_dictionary(context_name)
        elif context_name:
            self.motifs = self._filter_motion_library_by_context(self.motifs, context_name)

    def _build_from_motion_dictionary(self, context_name: str | None):
        motifs = []
        contexts = self.model.get("contexts", {})
        if not isinstance(contexts, dict):
            return motifs
        if context_name:
            selected = self._resolve_context(context_name, contexts)
            ctx_values = [contexts[selected]] if selected else []
        else:
            ctx_values = list(contexts.values())
        for ctx in ctx_values:
            chunks = ctx.get("motion_dictionary", {}).get("chunks", [])
            motifs.extend(self._chunks_to_motifs(chunks))
        return motifs

    @staticmethod
    def _norm(s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    def _resolve_context(self, context_name: str, contexts: Dict[str, Any]) -> str | None:
        if context_name in contexts:
            return context_name
        n = self._norm(context_name)
        for key in contexts.keys():
            nk = self._norm(key)
            if n == nk or n in nk or nk in n:
                return key
        return None

    def _filter_motion_library_by_context(self, motifs: list[Dict[str, Any]], context_name: str) -> list[Dict[str, Any]]:
        n = self._norm(context_name)
        out = []
        for m in motifs:
            mc = str(m.get("context") or m.get("source_context") or "")
            if not mc:
                continue
            nm = self._norm(mc)
            if n == nm or n in nm or nm in n:
                out.append(m)
        return out if out else motifs

    def _chunks_to_motifs(self, chunks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        motifs: list[Dict[str, Any]] = []
        for ch in chunks:
            motif: Dict[str, Any] = {}
            hip = ch.get("hip_delta", {})
            hx = hip.get("x", [])
            hy = hip.get("y", [])
            hz = hip.get("z", [])
            if min(len(hx), len(hy), len(hz)) > 0:
                motif["hipControl"] = {
                    "pos_x": hx,
                    "pos_y": hy,
                    "pos_z": hz,
                    "rot_x": [0.0] * len(hx),
                    "rot_y": [0.0] * len(hx),
                    "rot_z": [0.0] * len(hx),
                    "rot_w": [1.0] * len(hx),
                }
            ctrls = ch.get("controllers", {})
            if isinstance(ctrls, dict):
                for ctrl, d in ctrls.items():
                    if not isinstance(d, dict):
                        continue
                    px = d.get("x", [])
                    py = d.get("y", [])
                    pz = d.get("z", [])
                    if min(len(px), len(py), len(pz)) <= 0:
                        continue
                    qx = d.get("qx", [0.0] * len(px))
                    qy = d.get("qy", [0.0] * len(px))
                    qz = d.get("qz", [0.0] * len(px))
                    qw = d.get("qw", [1.0] * len(px))
                    motif[ctrl] = {
                        "pos_x": px,
                        "pos_y": py,
                        "pos_z": pz,
                        "rot_x": qx,
                        "rot_y": qy,
                        "rot_z": qz,
                        "rot_w": qw,
                    }
            if motif:
                motifs.append(motif)
        return motifs

    def get_random_chunk(self, controller: str):
        if not self.motifs:
            return None
        valid_motifs = [m for m in self.motifs if controller in m]
        if not valid_motifs:
            return None
        motif = np.random.choice(valid_motifs)
        return motif[controller]


class BehavioralSynthesizer:
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)
        self.sampler = MocapSampler(self.model_data, context_name=None)

        self.keyframe_interval = 0.6
        self.base_anchor: Dict[str, Dict[str, float]] = {}

    def set_keyframe_interval_ms(self, interval_ms: float) -> None:
        ms = float(np.clip(interval_ms, 100.0, 1000.0))
        self.keyframe_interval = ms / 1000.0

    def set_base_anchor(self, anchor: Dict[str, Any] | None) -> None:
        self.base_anchor = {}
        if not anchor:
            return
        for ctrl, data in anchor.items():
            if not isinstance(data, dict):
                continue
            self.base_anchor[ctrl] = {
                "x": float(data.get("x", 0.0)),
                "y": float(data.get("y", 0.0)),
                "z": float(data.get("z", 0.0)),
            }

    def generate_session(self, duration: float, intensity: float, playfulness: float, context: str | None = None) -> Dict[str, Any]:
        self.sampler = MocapSampler(self.model_data, context_name=context)
        timeline = {}
        controllers = [
            "hipControl",
            "chestControl",
            "headControl",
            "lHandControl",
            "rHandControl",
            "lKneeControl",
            "rKneeControl",
            "lFootControl",
            "rFootControl",
        ]

        num_anchors = int(duration / self.keyframe_interval) + 1
        export_times = np.linspace(0, duration, num_anchors)

        print("\n[SYNTHESIZER] Generiere extrem ausgedünnte Timeline...")
        print(f"[SYNTHESIZER] Dauer: {duration}s | Abstand: {self.keyframe_interval}s | Frames gesamt: {num_anchors}")
        print(f"[SYNTHESIZER] Verfügbare Motifs: {len(self.sampler.motifs)}")

        for ctrl in controllers:
            pos_list = []
            rot_list = []

            current_pos = np.zeros(3)
            current_rot = [0, 0, 0, 1]

            for _ in range(num_anchors):
                chunk = self.sampler.get_random_chunk(ctrl)
                if chunk and len(chunk.get("pos_x", [])) > 0:
                    idx = int(np.random.randint(0, len(chunk["pos_x"])))
                    target_pos = np.array([chunk["pos_x"][idx], chunk["pos_y"][idx], chunk["pos_z"][idx]])
                    target_rot = np.array(
                        [chunk["rot_x"][idx], chunk["rot_y"][idx], chunk["rot_z"][idx], chunk["rot_w"][idx]]
                    )
                else:
                    target_pos = current_pos
                    target_rot = current_rot

                pos_list.append(target_pos)
                rot_list.append(target_rot)
                current_pos = target_pos
                current_rot = target_rot

            pos_arr = np.array(pos_list)
            rot_arr = np.array(rot_list)

            # Glätten der Peaks
            for axis in range(3):
                pos_arr[:, axis] = gaussian_filter1d(pos_arr[:, axis], sigma=1.5)

            base = self.base_anchor.get(ctrl, {"x": 0.0, "y": 0.0, "z": 0.0})
            pos_arr[:, 0] += float(base["x"])
            pos_arr[:, 1] += float(base["y"])
            pos_arr[:, 2] += float(base["z"])

            timeline[ctrl] = {"times": export_times.tolist(), "pos": pos_arr.tolist(), "rot": rot_arr.tolist()}

        return timeline

    def export_vamtline(self, timeline: Dict[str, Any], output_path: str):
        """Schreibt die Sparse-Timeline als VaM Timeline JSON (AtomType=Person)."""
        payload = {
            "SerializeVersion": "283",
            "SerializeMode": "1",
            "AtomType": "Person",
            "GeneratedBy": "motion_synthesizer.py",
            "Clips": [
                {
                    "AnimationName": "Behavioral Generated",
                    "AnimationLength": "0",
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
                    "AnimationSegment": "Generated",
                    "Controllers": [],
                    "FloatParams": [],
                    "Triggers": [],
                }
            ],
        }

        frame_count = len(timeline["hipControl"]["times"])
        print(f"[SYNTHESIZER] SCHREIBE DATEI: Erzwungene {frame_count} Keyframes in {output_path}")
        duration = float(timeline["hipControl"]["times"][-1]) if frame_count > 0 else 0.0
        payload["Clips"][0]["AnimationLength"] = f"{duration:.6f}".rstrip("0").rstrip(".")

        for ctrl, data in timeline.items():
            ctrl_entry = {
                "Controller": ctrl,
                "TargetsPosition": "1",
                "TargetsRotation": "1",
                "ControlPosition": "1",
                "ControlRotation": "1",
                "X": [],
                "Y": [],
                "Z": [],
                "RotX": [],
                "RotY": [],
                "RotZ": [],
                "RotW": [],
            }

            for i in range(len(data["times"])):
                p = data["pos"][i]
                r = np.asarray(data["rot"][i], dtype=float)

                norm = np.linalg.norm(r)
                if norm > 1e-6:
                    r = r / norm
                else:
                    r = np.array([0, 0, 0, 1], dtype=float)

                tt = round(float(data["times"][i]), 6)
                ctrl_entry["X"].append({"t": tt, "v": round(float(p[0]), 6), "c": 3})
                ctrl_entry["Y"].append({"t": tt, "v": round(float(p[1]), 6), "c": 3})
                ctrl_entry["Z"].append({"t": tt, "v": round(float(p[2]), 6), "c": 3})
                ctrl_entry["RotX"].append({"t": tt, "v": round(float(r[0]), 6), "c": 3})
                ctrl_entry["RotY"].append({"t": tt, "v": round(float(r[1]), 6), "c": 3})
                ctrl_entry["RotZ"].append({"t": tt, "v": round(float(r[2]), 6), "c": 3})
                ctrl_entry["RotW"].append({"t": tt, "v": round(float(r[3]), 6), "c": 3})

            payload["Clips"][0]["Controllers"].append(ctrl_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("[SYNTHESIZER] Export abgeschlossen (Timeline JSON / AtomType Person).")


PROJECT_ROOT = Path(r"G:\VAM_Fresh\VAM-Animation-Generator")
