# =====================================================================================
# 4. Visual toolkit: matplotlib style, a PIL die-shot renderer, SVG writer
# =====================================================================================
# Color assignment follows the job of the data: one hue, light->dark for magnitude
# (congestion); two hues + neutral midpoint for polarity (DRC deltas); a fixed
# categorical order for the attribution methods, never cycled.

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8b8a84"
SURFACE, GRIDC = "#fcfcfb", "#e6e5e0"
METHOD_COLOR = {}   # filled in once the method list exists

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
})

CMAP_CONG = mcolors.LinearSegmentedColormap.from_list(
    "cong", ["#f7f9fc", "#cfe0f5", "#93bdec", "#4f92dd", "#2a78d6", "#1b4f8f", "#0d2745"])
CMAP_DIV = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#1b4f8f", "#7fb0e4", "#eceae4", "#f0a184", "#c0392b"])


def save_fig(fig, name, also_pdf=True):
    fig.savefig(P("figures", name + ".png"), dpi=200)
    if also_pdf: fig.savefig(P("figures", name + ".pdf"))
    plt.close(fig)
    return P("figures", name + ".png")


# ---- fonts for the PIL renderer -------------------------------------------------------
def _font(sz, bold=False):
    base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for f in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try: return ImageFont.truetype(os.path.join(base, f), sz)
        except Exception: pass
    return ImageFont.load_default()


def _ramp_rgb(a, cmap, lo=None, hi=None):
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    n = (a - lo) / max(hi - lo, 1e-9)
    return (np.asarray(cmap(np.clip(n, 0, 1)))[..., :3] * 255).astype(np.uint8)


def render_die(design: Design, res: Result, title: str, subtitle: str = "",
               field: str = "util", annot: Optional[Dict[int, str]] = None,
               highlight: Optional[Sequence[int]] = None, px: int = 1400) -> Image.Image:
    """A 'die shot': congestion raster + macro floorplan + DRC markers, drawn directly to a
    bitmap (no matplotlib), which is what a physical-design tool's GUI actually shows you."""
    pad_l, pad_t, pad_b = 60, 96, 74
    W = px + 2 * pad_l
    H = px + pad_t + pad_b
    img = Image.new("RGB", (W, H), SURFACE)
    dr = ImageDraw.Draw(img, "RGBA")
    G = res.usage.shape[0]

    fld = {"util": res.usage / np.maximum(res.cap, 1e-9), "hotspot": res.hotspot,
           "lam": res.lam, "ovfl": res.ovfl}[field]
    hi = float(np.percentile(fld, 99.5)) or 1.0
    heat = Image.fromarray(_ramp_rgb(fld[::-1], CMAP_CONG, 0.0, hi)).resize((px, px), Image.NEAREST)
    img.paste(heat, (pad_l, pad_t))

    sc = px / design.die
    fx = lambda x: pad_l + x * sc
    fy = lambda y: pad_t + px - y * sc

    # standard cells as a faint pin-density stipple
    step = max(1, len(res.cell_xy) // 2600)
    for x, y in res.cell_xy[::step]:
        dr.point((fx(x), fy(y)), fill=(255, 255, 255, 70))

    # DRC markers
    gs = design.die / G
    ys, xs = np.where(res.hotspot >= 1.0)
    for gy, gx in zip(ys, xs):
        cx, cy = fx((gx + .5) * gs), fy((gy + .5) * gs)
        r = 3.5
        dr.line([(cx - r, cy - r), (cx + r, cy + r)], fill="#e34948", width=2)
        dr.line([(cx - r, cy + r), (cx + r, cy - r)], fill="#e34948", width=2)

    # macros
    hl = set(highlight or [])
    for i in range(design.NM):
        w, h = design.macro_wh[i] * sc
        cx, cy = fx(res.macro_xy[i, 0]), fy(res.macro_xy[i, 1])
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        acc = "#e34948" if i in hl else "#0b0b0b"
        dr.rectangle(box, fill=(255, 255, 255, 205), outline=acc, width=5 if i in hl else 2)
        for k in range(1, 4):                                  # SRAM-ish hatch
            dr.line([box[0] + 4, box[1] + k * (h / 4), box[2] - 4, box[1] + k * (h / 4)],
                    fill=(11, 11, 11, 40), width=1)
        lab = design.macro_names[i] + (f"\n{annot[i]}" if annot and i in annot else "")
        f = _font(20, bold=True)
        dr.multiline_text((cx, cy), lab, font=f, fill=acc, anchor="mm", align="center")

    dr.rectangle([pad_l, pad_t, pad_l + px, pad_t + px], outline="#0b0b0b", width=2)
    dr.text((pad_l, 26), title, font=_font(34, True), fill=INK)
    dr.text((pad_l, 66), subtitle, font=_font(19), fill=INK2)

    # colorbar
    bx0, by0, bw, bh = pad_l, pad_t + px + 22, px * 0.45, 14
    ramp = Image.fromarray(_ramp_rgb(np.linspace(0, 1, 256)[None, :].repeat(8, 0), CMAP_CONG, 0, 1))
    img.paste(ramp.resize((int(bw), bh)), (int(bx0), int(by0)))
    dr.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=GRIDC)
    dr.text((bx0, by0 + bh + 6), {"util": "routing utilisation  0", "hotspot": "DRC markers  0",
                                  "lam": "congestion price  0", "ovfl": "overflow  0"}[field],
            font=_font(16), fill=INK2)
    dr.text((bx0 + bw, by0 + bh + 6), f"{hi:.2f}", font=_font(16), fill=INK2, anchor="ra")
    dr.line([(bx0 + bw + 60, by0 + 7 - 5), (bx0 + bw + 60 + 10, by0 + 7 + 5)], fill="#e34948", width=2)
    dr.line([(bx0 + bw + 60, by0 + 7 + 5), (bx0 + bw + 60 + 10, by0 + 7 - 5)], fill="#e34948", width=2)
    dr.text((bx0 + bw + 80, by0 + 2), "DRC marker", font=_font(16), fill=INK2)
    return img


def svg_open(w, h, bg=SURFACE):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="{bg}"/>']


def svg_save(parts, name):
    parts.append("</svg>")
    path = P("svg", name + ".svg")
    with open(path, "w") as f: f.write("\n".join(parts))
    return path


ART: Dict[str, str] = {}      # registry of every artefact this notebook writes

for k, d in DESIGNS.items():
    im = render_die(d, BASE[k], f"{d.name}  -  as-given floorplan",
                    f"DRC markers {BASE[k].drc:.0f} in {BASE[k].n_hotspot} GCells   |   "
                    f"WL {BASE[k].wl:.0f}   |   {d.NM} macros, {d.NC} cell clusters, {d.NNET} nets")
    p = P("images", f"dieshot_{k}_base.png"); im.save(p); ART[f"dieshot_{k}"] = p
print("die shots:", [os.path.basename(v) for v in ART.values()])
