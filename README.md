# Who's to blame? Actual causation and congestion pricing for macro-placement failures

A single Kaggle notebook that runs the whole study end to end — no internet, no external
datasets, no EDA-tool install:

```
whos_to_blame_macro_placement.ipynb     the notebook (run top to bottom)
cells/                                  its source, one file per cell
build_notebook.py                       cells/ -> .ipynb
run_cells.py                            execute cells/ as a plain script (for local dev)
outputs/                                everything the notebook produces
```

**The claim.** Saliency answers *what did the model look at?* A designer asks the courtroom
question: *but for macro M3 being here, would this DRC hotspot exist?* The notebook computes
Halpern–Pearl probability of necessity / sufficiency, Chockler–Halpern degree of responsibility
and blame, a Pigou/Vickrey congestion price per macro, and Shapley values — then validates all of
them by re-running place and route on the edits each one recommends, against Grad-CAM,
integrated gradients and a random control.

**The oracle.** Kaggle has no OpenROAD, so the notebook ships a small but real flow: an
ePlace-style analytical placer that re-places the standard cells in response to the macros,
then a PathFinder-style negotiated-congestion global router whose overflow map is the
DRC-marker proxy. One call is ~50–180 ms, so a few hundred genuine interventions per design are
affordable. Everything downstream touches it through one method, `evaluate(macro_xy) -> Result`;
`outputs/data/orfs_adapter.py` is the drop-in OpenROAD/ORFS replacement.

**Scale switch.** `QUICK = True` (the default) runs in ~4 minutes and produces every artifact.
Set it to `False` for the reported numbers (~30–60 min on a Kaggle CPU kernel).

**Outputs.** `outputs/html/report.html` (self-contained), `figures/*.png|pdf`, `images/*.png`
(bitmap die shots), `svg/*.svg` (verdict cards, causal model), `anim/repair_loop.gif`,
`tables/*.csv|tex`, `data/interventions.csv`.
