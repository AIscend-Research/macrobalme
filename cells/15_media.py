# =====================================================================================
# 15. Media pack: die shots, a legal-style verdict card in SVG, an animated repair, Plotly
# =====================================================================================
# Not everything belongs in a matplotlib axis. A floorplan wants to be a bitmap; an argument
# about who is to blame wants to be a document; a repair wants to be a before/after.

# ---- (a) the verdict: PIL die shots, annotated with the attribution ---------------------
name = max(DESIGNS, key=lambda n: BASE[n].drc)
d, o = DESIGNS[name], ORACLES[name]
resp = HP[name]["resp_score"][:, 0]; pn = HP[name]["PN"][:, 0]
annot = {m: f"resp {HP[name]['resp'][m,0]:.2f}\nPN {pn[m]:.2f}" for m in range(d.NM)}
im = render_die(d, BASE[name], f"{name}  -  causal responsibility for signoff failure",
                "annotation: Halpern-Pearl degree of responsibility and probability of necessity",
                annot=annot, highlight=[int(np.argmax(resp))])
im.save(P("images", f"dieshot_{name}_responsibility.png"))
ART["dieshot_resp"] = P("images", f"dieshot_{name}_responsibility.png")

sal_top = int(np.argmax(SCORES[name]["Grad-CAM"]))
im2 = render_die(d, BASE[name], f"{name}  -  what gradient saliency looked at",
                 "annotation: Grad-CAM mass over each macro footprint (normalised)",
                 annot={m: f"cam {_norm(SCORES[name]['Grad-CAM'])[m]:.2f}" for m in range(d.NM)},
                 highlight=[sal_top], field="hotspot")
im2.save(P("images", f"dieshot_{name}_saliency.png"))
ART["dieshot_sal"] = P("images", f"dieshot_{name}_saliency.png")

# side-by-side contact sheet
sheet = Image.new("RGB", (im.width + im2.width, im.height), SURFACE)
sheet.paste(im, (0, 0)); sheet.paste(im2, (im.width, 0))
sheet.thumbnail((2400, 2400))
sheet.save(P("images", "contact_sheet_cause_vs_saliency.png"))
ART["contact_sheet"] = P("images", "contact_sheet_cause_vs_saliency.png")

# ---- (b) the repair, as an animation ---------------------------------------------------
best_method = REPAIR.iloc[0]["method"] if REPAIR.iloc[0]["method"] != "Oracle-ranked (ground truth)" \
    else REPAIR.iloc[1]["method"]
tr = TRACES[(name, best_method)]
frames = []
for i, (m, res) in enumerate(tr):
    hl = [int(x[0]) for x in tr[1:i + 1] if x[0] is not None]
    f = render_die(d, res, f"{name}  -  repair step {i}/{len(tr)-1}   ({best_method})",
                   ("as given" if m is None else
                    f"moved {d.macro_names[int(m)]}   ->   DRC {res.drc:.0f} "
                    f"({100*(res.drc-tr[0][1].drc)/tr[0][1].drc:+.1f}%)"),
                   highlight=hl, px=900)
    frames.append(f.convert("P", palette=Image.ADAPTIVE))
frames = frames + [frames[-1]] * 2
frames[0].save(P("anim", "repair_loop.gif"), save_all=True, append_images=frames[1:],
               duration=1100, loop=0)
ART["repair_gif"] = P("anim", "repair_loop.gif")

# ---- (c) SVG: the verdict card ----------------------------------------------------------
def verdict_card(name, outcome=0, path="verdict_card"):
    d = DESIGNS[name]
    R = HP[name]["resp"][:, outcome]; RS = HP[name]["resp_score"][:, outcome]
    PNv, PSv = HP[name]["PN"][:, outcome], HP[name]["PS"][:, outcome]
    pr = _norm(PRICE[name]["pigou"]); sh = SHAP[name]
    order = np.argsort(-RS)
    W, H = 980, 150 + 46 * d.NM
    s = svg_open(W, H)
    lbl = "signoff failure (total DRC)" if outcome == 0 else f"hotspot R{outcome-1}"
    s.append(f'<text x="34" y="52" font-size="27" font-weight="bold" fill="{INK}">'
             f'Attribution of responsibility &#8212; {name}</text>')
    s.append(f'<text x="34" y="78" font-size="15" fill="{INK2}">outcome under consideration: '
             f'{lbl}, {BASE[name].drc:.0f} markers in {BASE[name].n_hotspot} GCells</text>')
    cols = [("macro", 34), ("HP responsibility", 150), ("PN", 400), ("PS", 470),
            ("Shapley", 545), ("congestion price", 660), ("verdict", 800)]
    for t, x in cols:
        s.append(f'<text x="{x}" y="112" font-size="12.5" fill="{INK3}" '
                 f'letter-spacing="0.06em">{t.upper()}</text>')
    s.append(f'<line x1="34" y1="120" x2="{W-34}" y2="120" stroke="{GRIDC}" stroke-width="1.5"/>')
    for i, m in enumerate(order):
        y = 150 + 46 * i
        verdict = ("necessary cause" if PNv[m] > .6 else
                   "contributing cause" if RS[m] > .3 else
                   "sufficient, not necessary" if PSv[m] > .5 else "not a cause")
        vc = SERIES[7] if verdict == "necessary cause" else (
            SERIES[1] if verdict == "contributing cause" else INK3)
        s.append(f'<text x="34" y="{y+5}" font-size="17" font-weight="bold" fill="{INK}">'
                 f'{d.macro_names[m]}</text>')
        s.append(f'<rect x="150" y="{y-12}" width="{220*max(RS[m],0.004):.1f}" height="17" '
                 f'rx="4" fill="{SERIES[0]}"/>')
        s.append(f'<rect x="150" y="{y-12}" width="220" height="17" rx="4" fill="none" '
                 f'stroke="{GRIDC}"/>')
        s.append(f'<text x="378" y="{y+2}" font-size="12" fill="{INK2}" text-anchor="end">'
                 f'{R[m]:.2f}</text>')
        for x, v in ((400, PNv[m]), (470, PSv[m])):
            s.append(f'<text x="{x}" y="{y+2}" font-size="13" fill="{INK}">{v:.2f}</text>')
        s.append(f'<text x="545" y="{y+2}" font-size="13" fill="{INK}">{sh[m]:+.1f}</text>')
        s.append(f'<rect x="660" y="{y-10}" width="{110*pr[m]:.1f}" height="13" rx="3" '
                 f'fill="{SERIES[1]}" opacity="0.85"/>')
        s.append(f'<text x="800" y="{y+2}" font-size="13.5" fill="{vc}">{verdict}</text>')
        s.append(f'<line x1="34" y1="{y+22}" x2="{W-34}" y2="{y+22}" stroke="{GRIDC}"/>')
    s.append(f'<text x="34" y="{H-16}" font-size="11.5" fill="{INK3}">'
             f'PN = probability of necessity, PS = probability of sufficiency, both over the '
             f'designer\'s action set; responsibility = 1/(1+|W|) for the smallest witness '
             f'coalition W; price in units of routing cost imposed on other nets.</text>')
    return svg_save(s, path)


ART["verdict_svg"] = verdict_card(name)
for n_ in DESIGNS: verdict_card(n_, 0, f"verdict_card_{n_}")

# ---- (d) SVG: the causal model, drawn ---------------------------------------------------
def causal_diagram(name, m, path="causal_model"):
    d = DESIGNS[name]
    wit = HP[name]["witness"][m][0]
    W, H = 900, 400
    s = svg_open(W, H)
    s.append(f'<text x="30" y="44" font-size="23" font-weight="bold" fill="{INK}">'
             f'Why {d.macro_names[m]} is an actual cause</text>')
    s.append(f'<text x="30" y="70" font-size="14" fill="{INK2}">Halpern-Pearl AC2 with witness '
             f'set W = {{{", ".join(d.macro_names[j] for j in (wit[0] if wit else [])) or "empty"}}}'
             f'&#8194;&#8226;&#8194;degree of responsibility = '
             f'{HP[name]["resp"][m,0]:.2f}</text>')
    def node(x, y, t, fill, r=34):
        s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{INK}" '
                 f'stroke-width="1.6"/>')
        s.append(f'<text x="{x}" y="{y+5}" font-size="14" font-weight="bold" text-anchor="middle" '
                 f'fill="{"white" if fill != SURFACE else INK}">{t}</text>')
    def arrow(x1, y1, x2, y2, c=INK2, dash=""):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2" '
                 f'marker-end="url(#a)" {dash}/>')
    s.append(f'<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
             f'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="{INK2}"/></marker></defs>')
    node(110, 200, d.macro_names[m], SERIES[0])
    ys = np.linspace(130, 300, max(len(wit[0]) if wit else 0, 1))
    for j, yy in zip((wit[0] if wit else []), ys):
        node(110, yy + 90, d.macro_names[j], SERIES[3], 26)
        arrow(140, yy + 90, 320, 250)
    node(360, 200, "placer", SURFACE); node(560, 200, "router", SURFACE)
    node(770, 200, "DRC", SERIES[7])
    arrow(145, 200, 322, 200); arrow(396, 200, 522, 200); arrow(596, 200, 733, 200)
    s.append(f'<text x="30" y="360" font-size="13.5" fill="{INK2}">AC2(a): with W held at its '
             f'witness setting and {d.macro_names[m]} where it is, the violation still occurs. '
             f'AC2(b): move {d.macro_names[m]} alone and it does not. '
             f'Everything downstream &#8212; the standard cells, the routes &#8212; is recomputed '
             f'by the tool, which is what makes this a do() and not a perturbation.</text>')
    return svg_save(s, path)


ART["causal_svg"] = causal_diagram(name, int(np.argmax(HP[name]["resp_score"][:, 0])))

# ---- (e) Plotly: interactive congestion + method comparison -----------------------------
if HAS_PLOTLY:
    figs = []
    z = BASE[name].usage / BASE[name].cap
    f1 = go.Figure(go.Surface(z=z, colorscale="Blues", showscale=True,
                              colorbar=dict(title="utilisation")))
    f1.update_layout(title=f"{name}: routing utilisation surface (drag to rotate)",
                     scene=dict(zaxis_title="usage / capacity", xaxis_title="GCell x",
                                yaxis_title="GCell y"), height=620,
                     paper_bgcolor=SURFACE, font=dict(color=INK))
    figs.append(("congestion surface", f1))

    f2 = go.Figure()
    for k in ORDER:
        f2.add_trace(go.Bar(name=MSHORT[k], x=list(DESIGNS),
                            y=[REPAIR_DF[(REPAIR_DF.method == k) & (REPAIR_DF.design == n_)].red_A.mean()
                               for n_ in DESIGNS], marker_color=METHOD_COLOR[k]))
    f2.update_layout(barmode="group", title="DRC reduction after real re-runs, by design",
                     yaxis_title="% markers removed", height=520, paper_bgcolor=SURFACE,
                     plot_bgcolor=SURFACE, font=dict(color=INK))
    figs.append(("closed-loop repair", f2))

    dims = []
    for k in ORDER:
        col = RANK_DF[RANK_DF.method == k]
        dims.append(dict(label=MSHORT[k], values=[col.rho_total.mean(), col.rho_local.mean(),
                                                  col.rho_region.mean()]))
    f3 = go.Figure(go.Parcoords(line=dict(color=[0, 1, 2], colorscale="Blues"), dimensions=dims))
    f3.update_layout(title="rank correlation with ground truth (three targets)", height=460,
                     paper_bgcolor=SURFACE, font=dict(color=INK))
    figs.append(("attribution agreement", f3))

    parts = ['<meta charset="utf-8"><style>body{background:%s;color:%s;font-family:'
             'DejaVu Sans,Helvetica,Arial,sans-serif;max-width:1100px;margin:2rem auto}</style>'
             % (SURFACE, INK), "<h1>Interactive appendix</h1>"]
    for i, (t, f) in enumerate(figs):
        parts.append(f"<h2>{t}</h2>")
        parts.append(pio.to_html(f, include_plotlyjs="inline" if i == 0 else False,
                                 full_html=False))
    with open(P("html", "interactive.html"), "w") as fh: fh.write("\n".join(parts))
    ART["interactive"] = P("html", "interactive.html")
print("media:", {k: os.path.basename(v) for k, v in ART.items() if k not in ()})
