# Highway Detection System

Ứng dụng phát hiện vi phạm giao thông trên đường cao tốc sử dụng YOLO và Computer Vision.

## 🚀 Quick Start

### Development

```bash
# Clone repository
git clone <repository-url>
cd highway_detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python run_gui.py
```

### Build Windows App

```bash
# Using build script
scripts\build_windows.bat

# Or manually
pip install pyinstaller
pyinstaller highway_detection.spec --clean
```

## 📦 CI/CD Pipeline

Pipeline tự động chạy khi:
- **Push/PR** vào `main` hoặc `develop`: Chạy lint + tests
- **Tag `v*`**: Build Windows app + tạo GitHub Release

### Tạo Release mới

```bash
# Update version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Pipeline Jobs

| Job | Trigger | Description |
|-----|---------|-------------|
| `lint` | All pushes | Code quality check (flake8, black, isort) |
| `test` | All pushes | Unit tests trên matrix (OS x Python) |
| `build-windows` | main/tags | Build Windows executable |
| `release` | v* tags | Tạo GitHub Release với artifacts |

## 🛠️ Project Structure

```
highway_detection/
├── .github/workflows/    # CI/CD pipelines
├── gui/                  # PyQt5 GUI components
├── lane_detection/       # Lane & zone detection
├── models/               # YOLO model handlers
├── process/              # Video processing
├── tracking/             # Object tracking
├── violations/           # Violation detection
├── scripts/              # Build scripts
├── tests/                # Unit tests
├── run_gui.py           # Entry point
├── requirements.txt      # Dependencies
└── highway_detection.spec # PyInstaller config
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html
```

## 📋 Requirements

- Python 3.9+
- CUDA (optional, for GPU acceleration)
- Windows 10/11 (for packaged app)

## 📝 License

MIT License
