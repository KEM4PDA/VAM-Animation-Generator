# Models Folder

This folder stores learned/generated model artifacts and can become very large.

Ignored by default via `.gitignore`:
- `*.json`
- `*.csv`

Generate locally:

```powershell
cd G:\VAM_Fresh\VAM-Animation-Generator\src
python body_awareness_engine.py learn --mocaps G:\VAM_Fresh\SavedMocaps --out ..\models\body_awareness_model.json
```

