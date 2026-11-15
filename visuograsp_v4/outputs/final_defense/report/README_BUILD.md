# Building the VisuoGrasp V4 final report

* **Source:** `main.tex` (IEEEtran conference class, two columns) with bibliography `references.bib` (BibTeX, IEEEtran
  style). All figures are in `figures/` and referenced by relative paths.
* **Compiled output:** `VisuoGrasp_V4_Final_Report.pdf`, 8 pages including references and the appendix.

## Local build (Windows, MiKTeX, PowerShell)

```powershell
cd "C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\report"
$env:PATH = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;$env:PATH"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
Copy-Item main.pdf VisuoGrasp_V4_Final_Report.pdf -Force
```

MiKTeX needed two one-time steps on this laptop:
* `miktex packages check-update`: a fresh MiKTeX refuses to run until updates have been checked once.
* `initexmf --set-config-value=[MPM]AutoInstall=1`: lets MiKTeX install missing packages automatically.

The message "major issue: So far, you have not checked for MiKTeX updates" can still appear. It is only a warning.

## Overleaf

1. Upload `main.tex`, `references.bib` and the `figures/` folder, keeping the folder name.
2. Choose the pdfLaTeX compiler.

The packages used are standard: amsmath, amssymb, graphicx, booktabs, multirow, xcolor, cite, url, bm, array,
placeins, hyperref.

## Regenerating the figures (optional)

Figures are produced from frozen result files by scripts in `../build/`. Run them from the project root
(`C:\Users\user\Desktop\AMRL\Project_2_2`):

```powershell
python visuograsp_v4/outputs/final_defense/build/make_figures.py
python visuograsp_v4/outputs/final_defense/build/make_media.py
```

| Script | What it writes | Reads |
|---|---|---|
| `make_figures.py` | Statistical plots, architecture diagram | `rl/wave5/final_test/final/analysis/*` and `episodes.csv` |
| `make_media.py` | Image panels, GIFs | Existing videos and the re-rendered clips |
| `record_defense_clips.py` (already run) | Re-rendered clips | Replays 17 already-evaluated episodes with the frozen policies; checks each outcome against the frozen record |

`record_defense_clips.py` performs no training and no new evaluation.

**`figures/fig_e6_sequence.jpg`** is a spare figure; neither `main.tex` nor the slides reference it. Its content
(approach, contact, lift, hold) is shown in the bottom row of Fig. 6.

## Sources of every number

See `../evidence/VERIFIED_RESULTS.md` and `../evidence/SOURCE_MANIFEST.md`.

All learned results in the report use **PRIVILEGED** simulator state (object pose and GT contacts). None is vision-based.
