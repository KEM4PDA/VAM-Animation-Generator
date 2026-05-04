# VAM-Animation-Generator

## Struktur

- `src/` Python-Quellcode
  - `vam_behavioral_director.py`
  - `body_awareness_engine.py`
  - `motion_analyzer.py`
  - `motion_synthesizer.py`
- `models/` gelernte Modelle/Features
- `outputs/` generierte JSON-Animationen
- `outputs/tmp/` Test- und Zwischenstände
- `presets/` Base-Pose Presets (`.vap`)

## Schnellstart

```powershell
cd G:\VAM_Fresh\VAM-Animation-Generator\src
python vam_behavioral_director.py
```

BodyAware CLI:

```powershell
python body_awareness_engine.py learn --mocaps G:\VAM_Fresh\SavedMocaps
python body_awareness_engine.py synthesize --model G:\VAM_Fresh\VAM-Animation-Generator\models\body_awareness_model.json
```
