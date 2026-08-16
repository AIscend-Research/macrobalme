# =====================================================================================
# 12. Ground truth and faithfulness
# =====================================================================================
# Ground truth is not a label file: it is the tool. For every macro we already ran the real
# but-for experiment (move it to the default site, re-place, re-route), so we know the true
# effect of each macro on the total violation count and on each named hotspot.

GT = {}
for name in DESIGNS:
    d = DESIGNS[name]; v = DATA[name]
    idx = {int(r.changed): i for i, r in v["df"].iterrows() if r.kind == "single_off"}
    dall = np.array([BASE[name].drc - v["df"].drc.values[idx[m]] for m in range(d.NM)])
    dreg = np.array([[float(BASE[name].hotspot[r["mask"]].sum()
                            - v["hot"][idx[m]].astype(np.float32)[r["mask"]].sum())
                      for r in REGIONS[name]] for m in range(d.NM)])
    GT[name] = dict(delta_all=dall, delta_region=dreg)
    print(f"{name:11s} true effect of removing each macro (DRC markers): " +
          "  ".join(f"{d.macro_names[m]} {dall[m]:+.0f}" for m in np.argsort(-dall)))

# The ECO ground truth: for every macro, the best *local* move the tool actually rewards.
# One real place-and-route run per candidate site -- expensive, and the only honest referee.
t0 = time.time()
for name in DESIGNS:
    d, o = DESIGNS[name], ORACLES[name]
    best = np.zeros(d.NM); bsite = np.zeros((d.NM, 2))
    for m in range(d.NM):
        vals = []
        for s in LOCAL_SITES[name][m]:
            xy = d.base_sites.copy(); xy[m] = s
            vals.append(o.evaluate(xy).drc)
        j = int(np.argmin(vals))
        best[m] = BASE[name].drc - vals[j]; bsite[m] = LOCAL_SITES[name][m][j]
    GT[name]["delta_local"] = best; GT[name]["best_site"] = bsite
    print(f"{name:11s} best single ECO move: {d.macro_names[int(np.argmax(best))]} "
          f"-> {100*best.max()/BASE[name].drc:.1f}% fewer DRC markers "
          f"(worst choice: {100*best.min()/BASE[name].drc:+.1f}%)")
print(f"ECO ground truth: {time.time()-t0:.0f}s\n")

_rng = np.random.default_rng(CFG.seed + 31)
METHOD_SPEC = [
    # label,                group,       f(name) -> per-macro score,                 oracle calls at query time
    ("HP responsibility",  "causal",    lambda n: HP[n]["resp_score"][:, 0],         0),
    ("HP blame",           "causal",    lambda n: HP[n]["blame"][:, 0],              0),
    ("Prob. of necessity", "causal",    lambda n: HP[n]["PN"][:, 0],                 0),
    ("Shapley value",      "causal",    lambda n: SHAP[n],                           0),
    ("Pigou externality",  "economic",  lambda n: PRICE[n]["pigou"],                 lambda n: DESIGNS[n].NM),
    ("Shadow price",       "economic",  lambda n: PRICE[n]["dual"],                  0),
    ("Grad-CAM",           "saliency",  lambda n: SAL[n]["gradcam"],                 0),
    ("Integrated grads",   "saliency",  lambda n: SAL[n]["ig"],                      0),
    ("Input x gradient",   "saliency",  lambda n: SAL[n]["inputgrad"],               0),
    ("Proximity heuristic","heuristic", lambda n: SAL[n]["proximity"],               0),
    ("Random",             "control",   lambda n: _rng.random(DESIGNS[n].NM),        0),
]
METHODS = {m[0]: m[2] for m in METHOD_SPEC}
GROUP = {m[0]: m[1] for m in METHOD_SPEC}
GROUP_COLOR = {"causal": SERIES[0], "economic": SERIES[1], "saliency": SERIES[3],
               "heuristic": SERIES[4], "control": INK3}
METHOD_COLOR.update({m[0]: GROUP_COLOR[m[1]] for m in METHOD_SPEC})
SCORES = {n: {k: np.asarray(f(n), float) for k, f in METHODS.items()} for n in DESIGNS}

# ---- (i) rank agreement with the true but-for effects ---------------------------------
rows = []
for name in DESIGNS:
    for k in METHODS:
        rows.append(dict(design=name, method=k, group=GROUP[k],
                         rho_total=spearman(SCORES[name][k], GT[name]["delta_all"]),
                         rho_local=spearman(SCORES[name][k], GT[name]["delta_local"]),
                         rho_region=float(np.mean([spearman(SCORES[name][k],
                                                            GT[name]["delta_region"][:, r])
                                                   for r in range(len(REGIONS[name]))]))))
RANK_DF = pd.DataFrame(rows)
piv = RANK_DF.pivot_table(index="method", values=["rho_total", "rho_local", "rho_region"],
                          aggfunc="mean")
piv["group"] = [GROUP[m] for m in piv.index]
print("\nrank correlation with the tool's own but-for effects (mean over designs)")
print(piv.sort_values("rho_total", ascending=False).round(3).to_string())

# ---- (ii) deletion / insertion curves, measured with real P&R runs ---------------------
KDEL = min(5, min(d.NM for d in DESIGNS.values()))


def deletion_curve(name, order, insert=False):
    d, o = DESIGNS[name], ORACLES[name]
    out = []
    for k in range(0, KDEL + 1):
        on = np.ones(d.NM, bool) if not insert else np.zeros(d.NM, bool)
        on[list(order[:k])] = insert
        out.append(o.evaluate(config_from_mask(name, on)).drc)
    return np.array(out)


t0 = time.time()
DEL, INS = {}, {}
for name in DESIGNS:
    DEL[name], INS[name] = {}, {}
    for k in METHODS:
        order = np.argsort(-SCORES[name][k])
        DEL[name][k] = deletion_curve(name, order)
        INS[name][k] = deletion_curve(name, order, insert=True)
    # the best achievable curve given the same budget, by brute force over the true effects
    DEL[name]["_oracle_best"] = deletion_curve(name, np.argsort(-GT[name]["delta_all"]))
print(f"\ndeletion/insertion curves: {sum(o.calls for o in ORACLES.values())} cumulative oracle "
      f"calls, {time.time()-t0:.0f}s for this stage")

rows = []
for name in DESIGNS:
    b = BASE[name].drc
    for k in METHODS:
        c = DEL[name][k] / b
        rows.append(dict(design=name, method=k, group=GROUP[k],
                         drop_at_1=float(1 - c[1]), drop_at_3=float(1 - c[min(3, KDEL)]),
                         del_auc=float(1 - c[1:].mean()),
                         ins_auc=float((INS[name][k][1:] / b).mean())))
FAITH_DF = pd.DataFrame(rows)
FAITH = FAITH_DF.groupby(["method", "group"], as_index=False).mean(numeric_only=True)
print("\nfaithfulness (real re-runs): fraction of DRC removed by nulling the top-k macros")
print(FAITH.sort_values("del_auc", ascending=False).round(3).to_string(index=False))
RANK_DF.to_csv(P("tables", "rank_correlation.csv"), index=False)
FAITH_DF.to_csv(P("tables", "faithfulness.csv"), index=False)
