# Project Guidelines

## Code Style
- Follow existing Python style in this repo: type hints where already used, concise docstrings, and minimal comments.
- Preserve mixed Vietnamese/English wording used in existing modules, especially for GUI labels and user-facing messages.
- Keep changes scoped and avoid broad refactors unless explicitly requested.

## Architecture
- Entry points:
  - GUI app startup: `run_gui.py`
  - CLI processing: `main.py`
- Core processing pipeline lives in `process/video.py`:
  - model inference -> tracker update -> violation detection -> optional BEV render -> optional async video write.
- Model backend abstraction lives in `models/`:
  - Use `models.loader.load_model(...)` instead of importing backend handlers directly unless backend-specific behavior is required.
- Tracking code is isolated in `tracking/bytetrack.py`; violation logic in `violations/detector.py`; lane/BEV logic in `lane_mapping/`.
- GUI orchestration lives in `gui/main_window.py`, while parameter/state editing is in `gui/config_panel.py`.

## Build And Test
- Environment setup:
  - `pip install -r requirements.txt`
- Run GUI:
  - `python run_gui.py`
- Run CLI on video:
  - `python main.py --source video --input <video_path> --display`
- Run tests:
  - `python -m pytest tests`
  - Avoid bare `pytest` from repo root, because packaged `release/app/_internal/**` can be collected and cause unrelated import errors.
- Build executable:
  - `python scripts/build.py --clean`
- Build + stage + installer:
  - `python scripts/build.py --clean --installer --set-version <MAJOR.MINOR.PATCH>`

## Conventions And Pitfalls
- PyInstaller spec intentionally excludes `onnx` and `onnx.reference` in `vision_app.spec`; do not remove this without validating Windows build stability.
- For frozen Windows runs, DLL path setup in `run_gui.py` is critical for ONNX Runtime/Torch startup; keep `os.add_dll_directory` handling and long-lived handles intact.
- Keep release pipeline flow consistent: build `dist/vision_app` -> stage to `release/app` via `scripts/release.py` -> optional Inno Setup installer.
- In Inno Setup flow, prefer passing version via `ISCC /DMyAppVersion=...` (as done by release script) rather than parsing raw version text inside installer script logic.

## References
- Project overview and usage examples: `README.md`
- Build automation details: `scripts/build.py`
- Release/installer staging: `scripts/release.py`
- Packaging spec: `vision_app.spec`