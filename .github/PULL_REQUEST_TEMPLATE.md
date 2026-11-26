## Summary
Describe the changes and motivation concisely.

## Verification
- [ ] `ruff check backend scripts` passes
- [ ] `PYTHONPATH=backend pytest backend/tests -q` passes
- [ ] `cd frontend && npx tsc --noEmit --noUnusedLocals --noUnusedParameters && npm run build` passes
- [ ] No model weight files (`.pt`/`.pth`) or secrets/credentials committed
- [ ] Offline mode and CPU-only Low-End compatibility preserved
