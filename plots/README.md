# Final graphs

Run from the repository root:

```bash
python -m plots.generate
```

The command reads bundled measurements and writes `plots/final/` PNG/PDF figures for detector accuracy, macro-F1, false alarms and NMRI recall; search target GPU cost; fixed-compute validation scores; and anytime quality. No training or data download is required.

Original paper figures remain in `docs/latex/figures/` and study-specific result directories. Their original builders are in `scripts/build_*_assets.py`, `scripts/archive_free_analyze.py` and `scripts/efficiency_analyze.py`.
