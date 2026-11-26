# Contributing to Sky Observer

Contributions, bug reports, dataset integrations, and meteorological accuracy improvements are welcome.

## Development Setup

### Linux / macOS (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
cd frontend && npm ci && npm run build && cd ..
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[dev]"
cd frontend; npm ci; npm run build; cd ..
```

## Verification Before Opening a Pull Request

```bash
# Backend lint & tests
ruff check backend scripts
PYTHONPATH=backend pytest backend/tests -q

# Frontend typecheck & build
cd frontend
npx tsc --noEmit --noUnusedLocals --noUnusedParameters
npm run build
```

## Core Engineering Guidelines

1. **Offline-First**: All new network features must be strictly optional and disabled when `server.offline_mode = true`.
2. **Low-End Hardware Compatibility**: Keep the default CPU-only path (`radiometric-physics-v1`, 4 GB RAM) functional without requiring a GPU.
3. **Honest Metrics**: Never invent or hard-code accuracy numbers. Measure claims via `scripts/evaluate.py` or `scripts/evaluate_nowcast.py`.
4. **No Weight Files or Secrets in Git**: Never commit `.pt`, `.pth`, `.ckpt`, API keys, or local `config/config.yaml` credentials.
5. **Licensing**: Code must be compatible with the repository's **MIT** license. Do not copy GPL/unlicensed third-party code; cite papers/methods instead.
6. **Commit Messages**: Use Conventional Commits (`feat(...)`, `fix(...)`, `docs(...)`, `chore(...)`, `test(...)`).
