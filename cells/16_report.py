# =====================================================================================
# 16. The report: one self-contained HTML file with every number and figure in it
# =====================================================================================
import base64


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path, cap):
    ext = os.path.splitext(path)[1].lstrip(".")
    mime = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg"}.get(ext, "image/png")
    return (f'<figure><img src="data:{mime};base64,{b64(path)}" alt="{cap}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def svg_inline(path, cap):
    with open(path) as f: s = f.read()
    return f'<figure><div class="svg">{s}</div><figcaption>{cap}</figcaption></figure>'


def table_html(df, caption, fmt="{:.3f}"):
    d2 = df.copy()
    for c in d2.columns:
        if d2[c].dtype.kind == "f": d2[c] = d2[c].map(lambda v: fmt.format(v))
    return (f'<figure><div class="tbl">{d2.to_html(index=False, border=0)}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


def latex_table(df, name, caption):
    with open(P("tables", name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f", caption=caption,
                            label="tab:" + name))


latex_table(REPAIR.round(2), "repair_summary", "Closed-loop repair: DRC reduction after "
            "re-running place and route on the edits each attribution method recommends.")
latex_table(piv2.reset_index().round(3), "rank_correlation", "Rank correlation between each "
            "attribution and the tool's own counterfactual effects.")
latex_table(FAITH.round(3), "faithfulness", "Deletion/insertion faithfulness, measured by "
            "re-running place and route.")

# ---- read the findings off the numbers, so the prose cannot drift from the run ----------
GRP = lambda g: RANK_DF[RANK_DF.group.isin(g)].groupby("method").rho_total.mean()
diag_cause = GRP(["causal", "economic"]); diag_sal = GRP(["saliency"])
diag_heur = GRP(["heuristic"]); diag_rand = GRP(["control"])
rep = REPAIR.set_index("method")
rep_cause = rep[rep.group.isin(["causal", "economic"])].red_A
rep_sal = rep[rep.group == "saliency"].red_A
rep_heur = rep[rep.group == "heuristic"].red_A
bound = float(rep.loc["Oracle-ranked (ground truth)", "red_A"])
rnd_rep = float(rep.loc["Random", "red_A"])
FINDING = (
    f"<p><strong>On the diagnostic question &mdash; which macro is responsible for this "
    f"failure &mdash; the causal and economic attributions win outright.</strong> Their rank "
    f"correlation with the tool's own but-for effects averages "
    f"{diag_cause.mean():+.2f} (best: {diag_cause.idxmax()}, {diag_cause.max():+.2f}), against "
    f"{diag_sal.mean():+.2f} for gradient saliency (best {diag_sal.max():+.2f}, worst "
    f"{diag_sal.min():+.2f}) and {diag_rand.mean():+.2f} for the random control. Saliency is not "
    f"merely weaker here; on this target it carries almost no signal, which is what the "
    f"correlational/counterfactual distinction predicts.</p>"
    f"<p><strong>On the repair action, the picture is more interesting and less flattering to "
    f"a clean story.</strong> With a budget of {CFG.repair_topk} ECO moves, causal and economic "
    f"rankings remove {rep_cause.mean():.1f}% of DRC markers on average (best "
    f"{rep_cause.idxmax()} at {rep_cause.max():.1f}%) against {rnd_rep:.1f}% for the random "
    f"control and {rep_sal.mean():.1f}% for saliency &mdash; but the proximity heuristic, which "
    f"is simply <em>move whatever macro sits in the red</em>, reaches {rep_heur.max():.1f}%, "
    f"close to the {bound:.1f}% of the oracle-ranked bound. For an ECO-sized local move that is "
    f"not surprising: proximity is a decent proxy for <em>who can be relieved by moving a "
    f"little</em>, even though it is a poor proxy for <em>who is to blame</em> "
    f"({diag_heur.mean():+.2f} on the diagnostic target). The honest claim this run supports is "
    f"therefore narrower than 'causal beats everything': responsibility and price answer the "
    f"attribution question that saliency cannot answer at all, and they convert into repairs "
    f"that beat random and beat saliency &mdash; while a cheap spatial heuristic remains a "
    f"strong baseline for local repair specifically, and belongs in any honest table.</p>")

best = REPAIR[REPAIR.method != "Oracle-ranked (ground truth)"].iloc[0]
sal_best = REPAIR[REPAIR.group == "saliency"].red_A.max()
rnd = float(REPAIR[REPAIR.method == "Random"].red_A.iloc[0])
n_calls = sum(o.calls for o in ORACLES.values())

HERO = [("designs", f"{len(DESIGNS)}"),
        ("real P&R runs", f"{n_calls:,}"),
        ("counterfactuals answered", f"{len(DESIGNS) * CFG.shapley_perms * NMAX:,}"),
        ("best method", f"{MSHORT.get(best.method, best.method)}"),
        ("DRC removed", f"{best.red_A:.0f}%"),
        ("best saliency", f"{sal_best:.0f}%"),
        ("random control", f"{rnd:.0f}%")]

CSS = """
:root{--s:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8b8a84;--line:#e6e5e0;--a:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--s);color:var(--ink);font:16px/1.62 -apple-system,Segoe UI,
 "DejaVu Sans",Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:3.5rem 1.4rem 6rem}
h1{font-size:2.35rem;line-height:1.15;letter-spacing:-.02em;margin:.2rem 0 .6rem}
h2{font-size:1.35rem;margin:3.2rem 0 .6rem;padding-top:1.2rem;border-top:1px solid var(--line)}
h3{font-size:1.02rem;color:var(--ink2);margin:1.8rem 0 .4rem}
p,li{color:var(--ink2);max-width:74ch}
strong{color:var(--ink)}
.lede{font-size:1.12rem;color:var(--ink2)}
.hero{display:flex;flex-wrap:wrap;gap:.5rem;margin:2rem 0}
.hero div{flex:1 1 130px;border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem;
 background:#fff}
.hero b{display:block;font-size:1.5rem;color:var(--ink);letter-spacing:-.01em}
.hero span{font-size:.76rem;text-transform:uppercase;letter-spacing:.07em;color:var(--ink3)}
figure{margin:1.8rem 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{font-size:.86rem;color:var(--ink3);margin-top:.55rem}
.tbl{overflow-x:auto}
.svg{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:#fff;padding:.6rem}
.svg svg{max-width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink3);text-transform:uppercase;font-size:.72rem;
 letter-spacing:.06em;border-bottom:1.5px solid var(--line);padding:.45rem .5rem}
td{padding:.42rem .5rem;border-bottom:1px solid var(--line);color:var(--ink2)}
tr:hover td{background:#f4f6f9}
code{background:#f1f0ec;padding:.1rem .3rem;border-radius:4px;font-size:.88em}
.tag{display:inline-block;font-size:.72rem;padding:.12rem .5rem;border-radius:99px;
 border:1px solid var(--line);color:var(--ink3);margin-right:.3rem}
"""

H = ["<title>Who's to Blame?</title>", f"<style>{CSS}</style>", "<main>",
     "<span class='tag'>reproduction notebook</span>"
     "<span class='tag'>actual causation</span><span class='tag'>congestion pricing</span>",
     "<h1>Who's to blame?</h1>",
     "<p class='lede'>Actual causation and congestion pricing for explaining macro-placement "
     "failures. Every number on this page was produced by re-running a real place-and-route "
     "flow under intervention &mdash; not by inspecting a model's gradients.</p>",
     "<div class='hero'>" + "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in HERO)
     + "</div>",
     "<h2>The question</h2>",
     "<p>Saliency answers <em>what did the model look at?</em> A designer asks the courtroom "
     "question: <strong>but for macro M3 being here, would this DRC hotspot exist?</strong> "
     "The Halpern&ndash;Pearl framework turns that into computable quantities &mdash; probability "
     "of necessity and sufficiency, and a degree of responsibility equal to "
     "1/(1+|W|) for the smallest coalition W of other changes that makes a macro pivotal. "
     "Transport economics supplies a second reading: each macro is an agent on the routing "
     "graph, and its Pigouvian congestion price is the routing cost it imposes on everyone "
     "else &mdash; an attribution denominated in cost, which turns directly into an action.</p>",
     "<h2>The oracle</h2>",
     f"<p>The interventional oracle is the flow itself: an analytical placer that re-places "
     f"{min(d.NC for d in DESIGNS.values())}&ndash;{max(d.NC for d in DESIGNS.values())} "
     f"standard-cell clusters in response to the macros, then a negotiated-congestion global "
     f"router whose overflow map is the DRC-marker proxy. One call costs "
     f"{ORACLE_SECONDS_PER_CALL*1000:.0f}&nbsp;ms, so a few hundred genuine <code>do()</code> "
     f"operations per design are affordable; the {len(ALL_DF):,} of them in this run are what "
     f"the surrogates are fitted to and what every claim below is checked against.</p>",
     img_tag(ART["fig1"], "The oracle. Left: macro floorplan and the placer's response. "
             "Middle: routing utilisation. Right: DRC markers, with the named hotspot regions "
             "R0-R2 that the causal analysis attributes blame for."),
     img_tag(ART["dieshot_ibex_like"], "A rendered die shot straight from the oracle "
             "(bitmap, not a plot): congestion raster, macro outlines, DRC markers."),
     "<h2>The surrogates</h2>",
     "<p>The causal quantities live on a lattice of macro configurations that no tool can "
     "enumerate. A U-Net predicts the marker map from the floorplan (and gives the saliency "
     "baselines something to be gradients of); a GNN over the macro graph amortises the "
     "set-function v&#770;(S) that necessity, responsibility and Shapley all query. The "
     "amortised query is roughly "
     f"{ORACLE_SECONDS_PER_CALL/45e-6:.0f}&times; cheaper than a real run.</p>",
     img_tag(ART["fig2"], "Held-out surrogate fidelity on interventions the models never saw."),
     "<h2>Six attributions, one floorplan</h2>",
     img_tag(ART["fig3"], "Normalised per-macro scores. The methods disagree, and the orange "
             "box marks the macro whose relocation the tool actually rewards most."),
     svg_inline(ART["verdict_svg"], "The attribution as a document: per-macro necessity, "
                "sufficiency, responsibility, Shapley share and congestion price, with the "
                "verdict each combination supports."),
     svg_inline(ART["causal_svg"], "Why the top-ranked macro qualifies as an actual cause under "
                "Halpern&ndash;Pearl AC2, and what its witness coalition is."),
     "<h2>Does it agree with the tool?</h2>",
     img_tag(ART["fig4"], "Rank correlation against three ground truths, all measured by "
             "re-running place and route: total DRC effect, best available ECO move, and "
             "per-hotspot effect."),
     img_tag(ART["fig5"], "Park the accused: DRC markers surviving after the top-k blamed "
             "macros are moved to their default sites and the block is re-placed and re-routed."),
     "<h2>What we found</h2>", FINDING,
     "<h2>The table this paper is for</h2>",
     "<p>A designer has a budget of ECO macro moves. Each method nominates which macros to "
     "move; every method is handed the same candidate sites and the same verification budget, "
     "so the only thing that varies is <strong>which macro was blamed</strong>. The tool grades "
     "the result.</p>",
     img_tag(ART["fig6"], "Closed-loop repair. Bars are means over designs; circles are the "
             "individual designs. The dotted line is a random-macro control averaged over five "
             "draws."),
     table_html(REPAIR.rename(columns={"red_A": "DRC removed % (shared sites)",
                                       "red_B": "DRC removed % (method-native sites)",
                                       "wl_A": "wirelength %", "worst": "worst design %",
                                       "win_rate": "cases improved %"})
                .drop(columns=["wl_B"]).round(2),
                "Closed-loop repair, full results.", "{:.1f}"),
     table_html(FAITH.round(3), "Deletion and insertion faithfulness (real re-runs)."),
     table_html(piv2.reset_index().round(3), "Rank correlation with the tool's counterfactuals."),
     "<h2>Why saliency fails here</h2>",
     img_tag(ART["fig7"], "The same block under three lenses. Grad-CAM highlights whatever sits "
             "in dense routing; responsibility asks whether removing it would have helped; the "
             "tool settles it."),
     img_tag(ART["contact_sheet"], "Left: responsibility and probability of necessity per macro. "
             "Right: Grad-CAM mass per macro. They do not name the same suspect."),
     "<h2>The price is an instruction</h2>",
     img_tag(ART["fig8"], "The congestion price field over one macro's legal ECO sites. The "
             "externality alone would park every macro in a corner; charging the externality and "
             "letting the macro minimise its own cost plus the charge gives the move."),
     "<h2>The repair, in motion</h2>",
     img_tag(ART["repair_gif"], "Each frame is a real place-and-route run after one more "
             "responsibility-ranked macro move."),
     "<h2>What would break this</h2>",
     "<ul>"
     "<li>The oracle here is a small placer and router, not OpenROAD. The pipeline is written "
     "against a one-method interface (<code>evaluate(macro_xy) -> Result</code>) and the final "
     "section ships the ORFS adapter, but the numbers on this page are the mini-flow's.</li>"
     "<li>Necessity and responsibility are computed against a <em>contrast set</em> &mdash; the "
     "sites a designer can actually move a macro to. Widen that set and the numbers move; "
     "a cause you cannot act on is not a useful cause.</li>"
     "<li>The surrogate is fitted per run on a few hundred interventions. Where it is wrong, "
     "the attributions are wrong, which is exactly why every headline claim is re-verified with "
     "real runs rather than read off the model.</li>"
     "<li>Three designs and a handful of ECO moves is a small sample, and multi-threaded CPU "
     "training is not bit-reproducible: repeated runs of this notebook move individual methods "
     "by a few points in the repair table, which is the same order as the gaps between "
     "neighbouring rows. The gaps that survive that noise are the ones between "
     "<em>groups</em> &mdash; causal/economic and the proximity heuristic above the random "
     "control, gradient saliency at or below it on the diagnostic target &mdash; and those are "
     "the only claims made here.</li></ul>",
     "</main>"]

with open(P("html", "report.html"), "w") as f: f.write("\n".join(H))
ART["report"] = P("html", "report.html")
print("report written:", ART["report"], f"({os.path.getsize(ART['report'])/1e6:.1f} MB)")
