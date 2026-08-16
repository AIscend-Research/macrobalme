# =====================================================================================
# 13. Closed-loop validation: does the attribution actually fix the block?
# =====================================================================================
# The table the paper is for. A designer has a budget of k macro moves. Each method nominates
# which macros to move; the edit is applied; the real tool re-places and re-routes; we count
# the DRC markers that survive. Nobody grades their own homework -- the oracle does.
#
#   Protocol A (headline): every method uses the SAME site-selection rule, so the only thing
#     that varies is *which* macros were blamed. This isolates attribution quality.
#   Protocol B: each method also chooses the destination in its own idiom -- the congestion
#     price has a gradient, saliency only has "move it away from the bright pixels".

def _sites_for(name, xy, m, n=None):
    """One fixed ECO-sized candidate set per macro, shared by every method, so no method wins
    by being handed better destinations."""
    S = LOCAL_SITES[name][m]
    ok = np.array([s for s in S if all(not overlaps(np.concatenate([xy[:m], s[None], xy[m + 1:]]),
                                                    DESIGNS[name].macro_wh, m, j)
                                       for j in range(DESIGNS[name].NM) if j != m)])
    if not len(ok): ok = S[:1]
    return ok if n is None else ok[:n]


def choose_site_surrogate(name, xy, m):
    """Common rule: among legal sites, take the one the surrogate says is best."""
    S = _sites_for(name, xy, m)
    cfgs = np.stack([np.concatenate([xy[:m], s[None], xy[m + 1:]]) for s in S])
    return S[int(np.argmin(v_hat_batch(name, cfgs)[:, 0]))]


def choose_site_price(name, xy, m, res):
    """The price-native edit: the same ECO candidate set, ranked by first-order social cost."""
    S = _sites_for(name, xy, m)
    o = ORACLES[name]
    ext = np.array([float((res.lam * o.macro_cover_one(
        np.concatenate([xy[:m], s[None], xy[m + 1:]]), m)).sum()) for s in S]) * o.base_cap * 1.84
    anc = _net_anchor(name, res, m)
    mid = (S[:, None, :] + anc[None, :, :]) / 2
    gi = np.clip((mid / o.gs).astype(int), 0, o.G - 1)
    priv = ((np.abs(S[:, None, :] - anc[None, :, :]).sum(-1) / o.gs)
            * (1.0 + res.lam[gi[..., 1], gi[..., 0]])).sum(1)
    return S[int(np.argmin(ext + priv))]


def choose_site_lowfield(name, xy, m, field):
    """The saliency-native edit: move the macro to where the map is coolest."""
    o = ORACLES[name]
    S = _sites_for(name, xy, m)
    load = [float((field * o.macro_cover_one(np.concatenate([xy[:m], s[None], xy[m + 1:]]), m)).sum())
            for s in S]
    return S[int(np.argmin(load))]


def repair(name, order, k, mode="surrogate", field=None):
    d, o = DESIGNS[name], ORACLES[name]
    xy = d.base_sites.copy()
    trace = [(None, o.evaluate(xy))]
    for m in order[:k]:
        m = int(m)
        if mode == "surrogate":  s = choose_site_surrogate(name, xy, m)
        elif mode == "price":    s = choose_site_price(name, xy, m, trace[-1][1])
        elif mode == "field":    s = choose_site_lowfield(name, xy, m, field)
        else:                    s = _sites_for(name, xy, m, 1)[0]
        xy = xy.copy(); xy[m] = s
        trace.append((m, o.evaluate(xy)))
    return xy, trace


NATIVE_MODE = {"Pigou externality": "price", "Shadow price": "price",
               "Grad-CAM": "field", "Integrated grads": "field", "Input x gradient": "field",
               "Proximity heuristic": "field", "Random": "random"}
NATIVE_FIELD = {"Grad-CAM": "cam_map", "Integrated grads": "ig_map"}

rows, TRACES = [], {}
t0 = time.time()
for name in DESIGNS:
    b_ = BASE[name]
    ranked = {k: [np.argsort(-SCORES[name][k])] for k in METHODS}
    # the control gets several draws, so it is not judged on one lucky order
    ranked["Random"] = [np.random.default_rng(CFG.seed + 500 + i).permutation(DESIGNS[name].NM)
                        for i in range(5)]
    ranked["Oracle upper bound"] = [np.argsort(-GT[name]["delta_local"])]
    for meth, orders in ranked.items():
        for rep, order in enumerate(orders):
            xyA, trA = repair(name, order, CFG.repair_topk, "surrogate")
            mode = NATIVE_MODE.get(meth, "surrogate")
            fld = (SAL[name][NATIVE_FIELD[meth]] if meth in NATIVE_FIELD else
                   (BASE[name].hotspot if mode == "field" else None))
            xyB, trB = repair(name, order, CFG.repair_topk, mode, fld)
            if rep == 0: TRACES[(name, meth)] = trA
            for k in range(1, CFG.repair_topk + 1):
                rA, rB = trA[k][1], trB[k][1]
                rows.append(dict(design=name, method=meth, group=GROUP.get(meth, "bound"),
                                 k=k, rep=rep,
                                 moved=",".join(DESIGNS[name].macro_names[int(m)] for m in order[:k]),
                                 drc_base=b_.drc, drc_A=rA.drc, drc_B=rB.drc,
                                 red_A=100 * (b_.drc - rA.drc) / b_.drc,
                                 red_B=100 * (b_.drc - rB.drc) / b_.drc,
                                 wl_A=100 * (rA.wl - b_.wl) / b_.wl,
                                 wl_B=100 * (rB.wl - b_.wl) / b_.wl))
REPAIR_DF = pd.DataFrame(rows)
_per_case = REPAIR_DF.groupby(["method", "group", "design", "k"], as_index=False).mean(numeric_only=True)
REPAIR = _per_case.groupby(["method", "group"], as_index=False).agg(
    red_A=("red_A", "mean"), red_B=("red_B", "mean"), wl_A=("wl_A", "mean"),
    wl_B=("wl_B", "mean"), worst=("red_A", "min"),
    win_rate=("red_A", lambda c: 100.0 * (np.asarray(c) > 0).mean())
).sort_values("red_A", ascending=False)
print(f"closed-loop repair: up to {CFG.repair_topk} macro moves per design, "
      f"{time.time()-t0:.0f}s, {sum(o.calls for o in ORACLES.values())} cumulative oracle calls\n")
print("DRC reduction after re-running place & route (%, mean over designs; higher is better)")
print(REPAIR.round(1).to_string(index=False))
REPAIR_DF.to_csv(P("tables", "repair_raw.csv"), index=False)
REPAIR.to_csv(P("tables", "repair_summary.csv"), index=False)
