#!/usr/bin/env python3
"""Assemble cells/*.{md,py} into a single self-contained Kaggle notebook."""
import json, glob, os, re

INTRO = {
 "02": "## 1. Designs\n\nThree blocks in the shape of small ORFS designs: hard macros (SRAMs), "
       "standard-cell clusters grouped into modules, a clustered hypergraph netlist and fixed IO "
       "pads. Each carries two reference floorplans — the **as-given** one we are asked to explain, "
       "and a canonical **peripheral** one that serves as the default `off` setting for every "
       "counterfactual.",
 "03": "## 2. The interventional oracle\n\nAn ePlace-style analytical placer that re-places the "
       "standard cells in response to the macros, followed by a PathFinder-style "
       "negotiated-congestion global router. `evaluate(macro_xy) -> Result` is the only interface "
       "the causal machinery ever touches — swap in the OpenROAD adapter at the end of the "
       "notebook and every experiment below reruns unchanged.",
 "04": "## 3. Visual toolkit\n\nA matplotlib style, a bitmap die-shot renderer (PIL) and a small "
       "SVG writer. Figures are produced in four media on purpose: plots for the statistics, "
       "bitmaps for the layouts, vector documents for the verdicts, and an animation for the "
       "repair loop.",
 "05": "## 4. The interventional dataset\n\nA few hundred genuine `do()` operations per design: "
       "park a macro at its default site, relocate it inside an ECO radius, swap a pair, jitter a "
       "coalition. This is the ground truth, and the surrogates' training set.",
 "06": "## 5. Surrogate A — U-Net on the congestion map\n\nPredicts the DRC-marker map from the "
       "floorplan alone. It is also the vehicle the saliency baselines need: without a "
       "differentiable predictor there is nothing for Grad-CAM to be a gradient of.",
 "07": "## 6. Surrogate B — GNN on the macro graph\n\nAmortises the set function "
       "v̂(configuration), which is what necessity, responsibility and Shapley all query "
       "thousands of times. Fourier position features matter here: a plain MLP is too smooth to "
       "feel an ECO-sized move, and the DRC response to one is not smooth at all.",
 "08": "## 7. Halpern–Pearl actual causation\n\nProbability of necessity, probability of "
       "sufficiency, degree of responsibility (Chockler–Halpern) and blame — computed against the "
       "designer's real action set, because a cause you cannot act on is a cause you cannot use.",
 "09": "## 8. Congestion pricing\n\nThe Pigouvian externality of each macro, measured exactly by "
       "intervention and estimated cheaply from the router's negotiated-congestion history, which "
       "is an estimate of the Lagrange multiplier on each GCell's capacity constraint.",
 "10": "## 9. Shapley values\n\nThe cooperative-game reading: the unique attribution whose "
       "per-macro numbers sum to the block's total violation count.",
 "11": "## 10. Baselines: gradient saliency\n\nWhat 'explainable ML for placement' usually means, "
       "implemented properly on the same surrogate so the comparison is fair.",
 "12": "## 11. Ground truth and faithfulness\n\nGround truth is the tool. Every macro's but-for "
       "effect and its best available ECO move are measured by real re-runs, and every attribution "
       "is scored against them.",
 "13": "## 12. Closed-loop validation — the table this is all for\n\nGiven a budget of macro "
       "moves, which macros should you move? Each method nominates; the tool grades.",
 "14": "## 13. Figures",
 "15": "## 14. Die shots, verdict cards, animation, interactive appendix",
 "16": "## 15. The report",
 "17": "## 16. Reproducing against OpenROAD, and the summary",
}

cells = []
def _id(): return f"cell{len(cells):03d}"
def md(t): cells.append({"id": _id(), "cell_type": "markdown", "metadata": {},
                         "source": t.splitlines(True)})
def code(t):
    cells.append({"id": _id(), "cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t.rstrip("\n").splitlines(True)})

for path in sorted(glob.glob("cells/*")):
    n = os.path.basename(path)[:2]
    if path.endswith(".md"):
        md(open(path).read())
    elif path.endswith(".py"):
        if n in INTRO: md(INTRO[n])
        code(open(path).read())

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "kaggle": {"accelerator": "none", "dataSources": [],
                              "isInternetEnabled": False, "language": "python",
                              "sourceType": "notebook"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "whos_to_blame_macro_placement.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print(f"{out}: {len(cells)} cells, {os.path.getsize(out)/1024:.0f} KB")
