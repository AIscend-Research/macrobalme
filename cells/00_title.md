# Who's to Blame? Actual Causation and Congestion Pricing for Explaining Macro-Placement Failures

**End-to-end reproduction notebook.** Runs top-to-bottom on a Kaggle CPU or GPU kernel with no
internet access and no external datasets. Everything below — designs, the interventional oracle,
the surrogates, the attributions, the closed-loop validation, and every figure — is produced here.

---

### The argument

Saliency asks *"what did the model look at?"* — a correlational question about a predictor.
A designer asks the courtroom question: **"but for macro M3 being here, would this DRC hotspot exist?"**
That is a counterfactual about the *design*, not about the network, and the Halpern–Pearl theory of
actual causation gives it a precise form. We compute, per macro:

| quantity | reading |
| --- | --- |
| **PN** — probability of necessity | move it away: does the violation vanish? |
| **PS** — probability of sufficiency | put it back into a clean floorplan: does the violation appear? |
| **PNS** | necessary *and* sufficient |
| **HP responsibility** (Chockler–Halpern) | how small a coalition of other changes it needs to become pivotal — $1/(1+|W|)$ |
| **Blame** | responsibility averaged over the designer's uncertainty about the context |
| **Congestion price** (Vickrey/Pigou) | the routing cost this macro imposes *on everybody else* — a shadow price in cost units |
| **Shapley value** | the game-theoretic split of the total DRC count across macros |

The trick that makes this cheap and defensible: **we own a perfect interventional oracle** — the
place-and-route engine itself. We run a few hundred *real* interventions (move one macro, swap two,
jitter a coalition), fit a surrogate (U-Net on congestion maps + a GNN on the macro graph) to
amortize the $2^{n}$ counterfactual queries, and then **verify the top attributions with real
oracle reruns**. That loop is the contribution.

### What the notebook produces

1. An interventional dataset over several designs (`interventions.csv`).
2. Two trained surrogates with held-out fidelity numbers.
3. Six attribution methods on the same designs: HP-responsibility, blame, Pigouvian price,
   Shapley, Grad-CAM, input-gradient saliency.
4. **The paper table**: responsibility-ranked edits vs. saliency-ranked edits, DRC reduction measured
   by re-running the oracle. Plus deletion/insertion faithfulness curves and rank correlation to
   exhaustive ground-truth necessity.
5. A figure pack: matplotlib PDFs, PIL-rendered die shots, hand-written SVG vector figures, an
   animated GIF of the repair loop, interactive Plotly HTML, and a self-contained `report.html`.

### A note on the oracle

Kaggle kernels have no OpenROAD/ORFS install and no network. So the notebook ships a
**self-contained mini-EDA oracle**: an ePlace-style analytical global placer for the standard-cell
clusters that *reacts* to macro positions, followed by a PathFinder/FastRoute-style
negotiated-congestion global router with rip-up-and-reroute on a GCell grid, whose overflow map is
the DRC-hotspot proxy. It is a real placer and a real router, just small. Every experiment here is
oracle-agnostic: `Oracle.evaluate(placement) -> Result` is the only interface the causal machinery
touches, and the final section ships a drop-in `ORFSOracle` adapter that shells out to `openroad`
on a machine that has it, so the identical notebook reproduces against ibex/aes on Nangate45
outside Kaggle.
