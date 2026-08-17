# =====================================================================================
# 14. Figure pack (static): matplotlib panels, PIL die shots, hand-written SVG
# =====================================================================================
MSHORT = {"HP responsibility": "HP resp.", "Prob. of necessity": "PN", "HP blame": "blame",
          "Shapley value": "Shapley", "Counterfactual dDRC": "cf-dDRC",
          "Pigou externality": "Pigou", "Shadow price": "shadow price",
          "Integrated grads": "int. grads", "Input x gradient": "input x grad",
          "Proximity heuristic": "proximity", "Grad-CAM": "Grad-CAM", "Random": "random"}
ORDER = [m[0] for m in METHOD_SPEC]


def _norm(v):
    v = np.asarray(v, float); v = v - v.min()
    return v / max(v.max(), 1e-12)


# ---- F1: what the oracle produces ------------------------------------------------------
fig, axes = plt.subplots(len(DESIGNS), 3, figsize=(11.5, 3.6 * len(DESIGNS)))
for r, (name, res) in enumerate(BASE.items()):
    d, o = DESIGNS[name], ORACLES[name]
    ax = axes[r, 0]
    ax.scatter(res.cell_xy[:, 0], res.cell_xy[:, 1], s=.6, c="#c9d7ea", lw=0)
    for m in range(d.NM):
        w, h = d.macro_wh[m]
        ax.add_patch(plt.Rectangle(d.base_sites[m] - [w / 2, h / 2], w, h, fc="white",
                                   ec=INK, lw=1.1))
        ax.text(*d.base_sites[m], d.macro_names[m], ha="center", va="center", fontsize=7)
    ax.set(xlim=(0, d.die), ylim=(0, d.die), title=f"{name}: floorplan + placement",
           xticks=[], yticks=[]); ax.grid(False); ax.set_aspect(1)
    ax = axes[r, 1]
    im = ax.imshow(res.usage / res.cap, origin="lower", cmap=CMAP_CONG, vmin=0,
                   vmax=np.percentile(res.usage / res.cap, 99.5))
    plt.colorbar(im, ax=ax, fraction=.046).set_label("utilisation", color=INK2)
    ax.set(title="routing utilisation", xticks=[], yticks=[]); ax.grid(False)
    ax = axes[r, 2]
    im = ax.imshow(res.hotspot, origin="lower", cmap=CMAP_CONG, vmin=0)
    for reg in REGIONS[name]:
        ys, xs = np.where(reg["core"])
        ax.add_patch(plt.Rectangle((xs.min() - .5, ys.min() - .5), np.ptp(xs) + 1, np.ptp(ys) + 1,
                                   fill=False, ec=SERIES[1], lw=1.4))
        ax.text(xs.mean(), ys.max() + 1.5, f"R{reg['rid']}", color=SERIES[1], fontsize=8,
                ha="center")
    plt.colorbar(im, ax=ax, fraction=.046).set_label("DRC markers", color=INK2)
    ax.set(title=f"DRC markers ({res.drc:.0f} total)", xticks=[], yticks=[]); ax.grid(False)
fig.suptitle("The oracle: place, route, count violations, name the hotspots", y=1.005,
             fontsize=12, ha="center")
ART["fig1"] = save_fig(fig, "fig1_oracle")

# ---- F2: surrogate fidelity ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
ax = axes[0]
for i, (lab, h, c) in enumerate([("U-Net (map)", UHIST[:, 1] if len(UHIST) > 1 else [], SERIES[0]),
                                 ("GNN", GHIST[:, 1] if len(GHIST) > 1 else [], SERIES[2])]):
    if len(h): ax.plot(np.asarray(h) / max(np.asarray(h)[0], 1e-9), color=c, lw=2, label=lab)
ax.set(xlabel="epoch", ylabel="held-out loss (normalised)", title="surrogate training")
ax.legend()
for ax, (fid, lab, c) in zip(axes[1:], [(UNET_FID, "U-Net", SERIES[0]), (GNN_FID, "GNN", SERIES[2])]):
    for i, name in enumerate(DESIGNS):
        idx = [j for j in VAL_IDX if D_ALL[j] == name]
        loc = [list(np.where(D_ALL == name)[0]).index(j) for j in idx]
        true = Y_ALL[idx].sum((1, 2))
        pred = (np.array([unet_predict(name, DATA[name]["xy"][j])[1] for j in loc]) if lab == "U-Net"
                else v_hat_batch(name, DATA[name]["xy"][loc])[:, 0])
        ax.scatter(true, pred, s=14, color=SERIES[i], lw=0, alpha=.8, label=name)
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=INK3, lw=1, ls="--")
    rho = np.mean([f["spearman"] for f in fid])
    ax.set(xlabel="true DRC markers (oracle)", ylabel=f"{lab} prediction",
           title=f"{lab}: held-out, mean rho = {rho:+.2f}")
    ax.legend(fontsize=7)
ART["fig2"] = save_fig(fig, "fig2_surrogates")

# ---- F3: the attributions disagree -----------------------------------------------------
fig, axes = plt.subplots(1, len(DESIGNS), figsize=(4.6 * len(DESIGNS), 4.6))
for ax, name in zip(np.atleast_1d(axes), DESIGNS):
    d = DESIGNS[name]
    M = np.stack([_norm(SCORES[name][k]) for k in ORDER])
    im = ax.imshow(M, cmap=CMAP_CONG, vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(d.NM), xticklabels=d.macro_names, yticks=range(len(ORDER)),
           yticklabels=[MSHORT[k] for k in ORDER], title=f"{name}: who is to blame?")
    for i, k in enumerate(ORDER):
        ax.get_yticklabels()[i].set_color(METHOD_COLOR[k])
    ax.grid(False)
    j = int(np.argmax(GT[name]["delta_local"]))
    ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(ORDER), fill=False, ec=SERIES[1], lw=2.2))
    ax.text(j, len(ORDER) - .3, "true best fix", color=SERIES[1], ha="center", fontsize=8)
plt.colorbar(im, ax=np.atleast_1d(axes)[-1], fraction=.04).set_label("normalised score", color=INK2)
ART["fig3"] = save_fig(fig, "fig3_attribution_matrix")

# ---- F4: rank agreement with the tool's own counterfactuals ----------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.2))
piv2 = RANK_DF.pivot_table(index="method", values=["rho_total", "rho_local", "rho_region"],
                           aggfunc="mean").reindex(ORDER)
x = np.arange(len(ORDER)); w = .27
targets = [("rho_total", "vs. total DRC effect", 1.0), ("rho_local", "vs. best ECO move", .72),
           ("rho_region", "vs. per-hotspot effect", .45)]
for i, (col, lab, alpha) in enumerate(targets):
    ax.bar(x + (i - 1) * w, piv2[col], w * .92,
           color=[METHOD_COLOR[m] for m in ORDER], alpha=alpha,
           edgecolor=SURFACE, linewidth=1.2)
# colour carries the method group, opacity carries the target -- so say so in neutral ink
handles = [plt.Rectangle((0, 0), 1, 1, fc=INK2, alpha=a, ec=SURFACE) for _, _, a in targets]
ax.axhline(0, color=INK2, lw=1)
ax.set(xticks=x, ylabel="Spearman rank correlation", title="Does the attribution agree with the tool?")
ax.set_xticklabels([MSHORT[m] for m in ORDER], rotation=32, ha="right")
for t, m in zip(ax.get_xticklabels(), ORDER): t.set_color(METHOD_COLOR[m])
ax.legend(handles, [t[1] for t in targets], ncol=3, fontsize=8)
ART["fig4"] = save_fig(fig, "fig4_rank_correlation")

# ---- F5: deletion curves, measured with real re-runs -----------------------------------
fig, axes = plt.subplots(1, len(DESIGNS) + 1, figsize=(3.6 * (len(DESIGNS) + 1), 3.4))
for ax, name in zip(axes, DESIGNS):
    for k in ORDER:
        ax.plot(DEL[name][k] / BASE[name].drc, color=METHOD_COLOR[k], lw=1.8,
                alpha=.95 if GROUP[k] in ("causal", "economic") else .55,
                ls="-" if GROUP[k] != "control" else ":")
    ax.plot(DEL[name]["_oracle_best"] / BASE[name].drc, color=INK, lw=1.6, ls="--")
    ax.set(title=name, xlabel="macros parked at default site", ylim=(0, 1.15),
           ylabel="DRC markers / base" if name == list(DESIGNS)[0] else None)
ax = axes[-1]
for k in ORDER:
    ax.plot(np.mean([DEL[n][k] / BASE[n].drc for n in DESIGNS], 0), color=METHOD_COLOR[k],
            lw=2.2, label=MSHORT[k], ls="-" if GROUP[k] != "control" else ":")
ax.plot(np.mean([DEL[n]["_oracle_best"] / BASE[n].drc for n in DESIGNS], 0), color=INK, lw=1.8,
        ls="--", label="oracle best")
ax.set(title="mean over designs", xlabel="macros parked at default site", ylim=(0, 1.15))
ax.legend(fontsize=6.5, ncol=2)
fig.suptitle("Park the accused: real place & route after removing the top-k blamed macros",
             y=1.02, fontsize=11)
ART["fig5"] = save_fig(fig, "fig5_deletion_curves")

# ---- F6: THE table, as a figure ---------------------------------------------------------
sub = REPAIR.set_index("method")
order6 = [m for m in ["Oracle-ranked (ground truth)"] + ORDER if m in sub.index]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
vals = sub.loc[order6, "red_A"].values
cols = [INK if m == "Oracle-ranked (ground truth)" else METHOD_COLOR[m] for m in order6]
ax.bar(np.arange(len(order6)), vals, .68, color=cols, edgecolor=SURFACE, linewidth=1.2)
per = REPAIR_DF.groupby(["method", "design"], as_index=False).red_A.mean()
for i, m in enumerate(order6):
    v = per[per.method == m].red_A.values
    ax.scatter(np.full(len(v), i) + np.linspace(-.16, .16, len(v)), v, s=16,
               facecolor=SURFACE, edgecolor=INK2, lw=.9, zorder=3)
rnd = float(sub.loc["Random", "red_A"])
ax.axhline(rnd, color=INK3, lw=1.2, ls=":")
ax.text(len(order6) - .4, rnd + .6, "random control", color=INK3, fontsize=8, ha="right")
ax.axhline(0, color=INK2, lw=1)
ax.set(xticks=np.arange(len(order6)), ylabel="DRC markers removed (%)",
       title=f"Closed-loop repair: {CFG.repair_topk} ECO macro moves, verified by re-running the tool")
ax.set_xticklabels([MSHORT.get(m, "oracle-ranked\n(ground truth)") for m in order6],
                   rotation=32, ha="right")
for t, m in zip(ax.get_xticklabels(), order6):
    t.set_color(INK if m == "Oracle-ranked (ground truth)" else METHOD_COLOR[m])
ART["fig6"] = save_fig(fig, "fig6_repair_table")

# ---- F7: the money picture -- saliency looks elsewhere ----------------------------------
name = max(DESIGNS, key=lambda n: BASE[n].drc)
d = DESIGNS[name]
sal_top = int(np.argmax(SCORES[name]["Grad-CAM"]))
resp_top = int(np.argmax(SCORES[name]["HP responsibility"]))
gt_top = int(np.argmax(GT[name]["delta_local"]))
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (fld, title) in zip(axes, [(BASE[name].hotspot, "DRC markers (what failed)"),
                                   (SAL[name]["cam_map"], "Grad-CAM (what the model looked at)"),
                                   (BASE[name].lam, "congestion price lambda (what it costs)")]):
    ax.imshow(fld, origin="lower", cmap=CMAP_CONG)
    for m in range(d.NM):
        w, h = d.macro_wh[m] / ORACLES[name].gs
        c = d.base_sites[m] / ORACLES[name].gs
        hot = m in (sal_top, resp_top, gt_top)
        ax.add_patch(plt.Rectangle(c - [w / 2, h / 2], w, h, fill=False,
                                   ec=SERIES[1] if hot else "white", lw=2 if hot else .8))
        ax.text(*c, d.macro_names[m], color=SERIES[1] if hot else "white", fontsize=7,
                ha="center", va="center")
    ax.set(title=title, xticks=[], yticks=[]); ax.grid(False)
fig.suptitle(f"{name}: Grad-CAM blames {d.macro_names[sal_top]}, HP responsibility blames "
             f"{d.macro_names[resp_top]}, the tool says {d.macro_names[gt_top]}", y=1.03, fontsize=11)
ART["fig7"] = save_fig(fig, "fig7_saliency_vs_cause")

# ---- F8: the congestion price field and the move it recommends --------------------------
m = max(range(d.NM), key=lambda i: PRICE[name]["dual"][i])
S, ext, priv = price_field(name, m)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
ax = axes[0]
ax.imshow(BASE[name].lam, origin="lower", cmap=CMAP_CONG,
          extent=[0, d.die, 0, d.die])
sc = ax.scatter(S[:, 0], S[:, 1], c=ext + priv, cmap="cividis", s=44, edgecolor="white", lw=.5)
j = int(np.argmin(ext + priv))
ax.annotate("", xy=S[j], xytext=d.base_sites[m],
            arrowprops=dict(arrowstyle="-|>", color=SERIES[1], lw=2.4))
ax.set(title=f"price field for {d.macro_names[m]} over its ECO sites", xticks=[], yticks=[])
ax.grid(False); plt.colorbar(sc, ax=ax, fraction=.046).set_label("private + external cost", color=INK2)
ax = axes[1]
o = np.argsort(ext + priv)
ax.plot(ext[o], color=SERIES[1], lw=2, label="externality (what it costs others)")
ax.plot(priv[o] * (ext.max() / max(priv.max(), 1e-9)), color=SERIES[0], lw=2,
        label="own routing cost (rescaled)")
ax.plot((ext + priv)[o] * (ext.max() / max((ext + priv).max(), 1e-9)), color=INK, lw=1.4, ls="--",
        label="social cost (rescaled)")
ax.set(xlabel="candidate site (sorted by social cost)", ylabel="first-order price",
       title="charge the externality, let the macro choose")
ax.legend(fontsize=8)
ART["fig8"] = save_fig(fig, "fig8_price_field")
print("static figures:", [k for k in ART if k.startswith("fig")])
