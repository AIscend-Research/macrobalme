# =====================================================================================
# 5. The interventional dataset: a few hundred real do() operations per design
# =====================================================================================
# Each row is one *actual* place-and-route run under an intervention on the macro
# placement. This is the ground truth everything else is measured against, and the
# training set for the surrogates that amortise the 2^NM counterfactual queries.

try:
    from scipy import ndimage as ndi
    def _label(mask): return ndi.label(mask, structure=np.ones((3, 3)))
except Exception:                                              # pragma: no cover
    def _label(mask):
        lab = np.zeros(mask.shape, int); cur = 0
        for sy in range(mask.shape[0]):
            for sx in range(mask.shape[1]):
                if mask[sy, sx] and not lab[sy, sx]:
                    cur += 1; st = [(sy, sx)]; lab[sy, sx] = cur
                    while st:
                        y, x = st.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                        and mask[ny, nx] and not lab[ny, nx]):
                                    lab[ny, nx] = cur; st.append((ny, nx))
        return lab, cur


def _dilate(mask, k=2):
    m = mask.copy()
    for _ in range(k):
        m = (m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1))
    return m


def overlaps(xy, wh, i, j, gap=8.0):
    return (abs(xy[i, 0] - xy[j, 0]) < (wh[i, 0] + wh[j, 0]) / 2 + gap and
            abs(xy[i, 1] - xy[j, 1]) < (wh[i, 1] + wh[j, 1]) / 2 + gap)


def candidate_sites(d: Design, xy: np.ndarray, m: int, k: int = 24, rng=None, lattice=7):
    """Legal alternative sites for macro m, holding every other macro where it is."""
    rng = rng or np.random.default_rng(0)
    w, h = d.macro_wh[m]
    gx = np.linspace(w / 2 + 10, d.die - w / 2 - 10, lattice)
    gy = np.linspace(h / 2 + 10, d.die - h / 2 - 10, lattice)
    cand = np.array([(a, b) for a in gx for b in gy])
    cand = cand + rng.uniform(-8, 8, cand.shape)
    ok = []
    for c in cand:
        t = xy.copy(); t[m] = c
        if all(not overlaps(t, d.macro_wh, m, j) for j in range(d.NM) if j != m):
            ok.append(c)
    ok = np.array(ok) if ok else d.null_sites[m:m + 1]
    if len(ok) > k: ok = ok[rng.choice(len(ok), k, replace=False)]
    return ok


def _jitter(d, xy, m, rng, scale=0.10):
    for _ in range(30):
        c = xy[m] + rng.normal(0, scale * d.die, 2)
        c = np.clip(c, d.macro_wh[m] / 2 + 6, d.die - d.macro_wh[m] / 2 - 6)
        t = xy.copy(); t[m] = c
        if all(not overlaps(t, d.macro_wh, m, j) for j in range(d.NM) if j != m):
            return c
    return xy[m]


def build_interventions(name: str, n: int, seed: int):
    d, o = DESIGNS[name], ORACLES[name]
    rng = np.random.default_rng(seed)
    base, null = d.base_sites, d.null_sites
    rows, maps, XY = [], [], []

    def run(xy, kind, changed, offmask):
        r = o.evaluate(xy)
        rows.append(dict(design=name, kind=kind, drc=r.drc, n_hotspot=r.n_hotspot, wl=r.wl,
                         changed=",".join(map(str, changed)),
                         offmask="".join("1" if b else "0" for b in offmask)))
        maps.append(r.hotspot.astype(np.float16)); XY.append(xy.copy())
        return r

    run(base, "base", [], np.zeros(d.NM, bool))
    run(null, "all_off", list(range(d.NM)), np.ones(d.NM, bool))
    # (a) exhaustive single-macro "off": the but-for test, run for real on every macro
    for m in range(d.NM):
        xy = base.copy(); xy[m] = null[m]
        off = np.zeros(d.NM, bool); off[m] = True
        run(xy, "single_off", [m], off)
    # (b) exhaustive single-macro relocation to a lattice of legal sites
    for m in range(d.NM):
        for c in candidate_sites(d, base, m, k=4, rng=rng):
            xy = base.copy(); xy[m] = c
            run(xy, "single_move", [m], np.zeros(d.NM, bool))
    while len(rows) < n:
        u = rng.random()
        off = np.zeros(d.NM, bool)
        if u < 0.42:                                  # (c) coalition "off" -- the Shapley lattice
            k = int(rng.integers(1, d.NM + 1))
            S = rng.choice(d.NM, k, replace=False)
            off[S] = True
            xy = np.where(off[:, None], null, base)
            run(xy, "coalition_off", list(S), off)
        elif u < 0.72:                                # (d) coalition jitter
            k = int(rng.integers(1, max(2, d.NM // 2) + 1))
            S = rng.choice(d.NM, k, replace=False)
            xy = base.copy()
            for m in S: xy[m] = _jitter(d, xy, m, rng)
            run(xy, "coalition_jitter", list(S), off)
        elif u < 0.88:                                # (e) pairwise swap
            i, j = rng.choice(d.NM, 2, replace=False)
            xy = base.copy(); xy[[i, j]] = xy[[j, i]]
            if any(overlaps(xy, d.macro_wh, i, t) for t in range(d.NM) if t != i): continue
            run(xy, "swap", [int(i), int(j)], off)
        else:                                         # (f) mixed: some off, some relocated
            k = int(rng.integers(1, d.NM + 1))
            S = rng.choice(d.NM, k, replace=False)
            off[S[: max(1, len(S) // 2)]] = True
            xy = np.where(off[:, None], null, base)
            for m in S[max(1, len(S) // 2):]:
                xy[m] = candidate_sites(d, xy, m, k=1, rng=rng)[0]
            run(xy, "mixed", list(S), off)
    return pd.DataFrame(rows), np.stack(maps), np.stack(XY)


DATA: Dict[str, dict] = {}
t0 = time.time()
for name in DESIGNS:
    df, mp, xy = build_interventions(name, CFG.n_interventions, CFG.seed + 91 * len(DATA))
    DATA[name] = dict(df=df, hot=mp, xy=xy)
    print(f"{name:11s} {len(df):5d} real P&R interventions   DRC "
          f"[{df.drc.min():.0f}, {df.drc.max():.0f}]  mean {df.drc.mean():.0f}   "
          f"({time.time() - t0:.0f}s elapsed)")

ALL_DF = pd.concat([v["df"] for v in DATA.values()], ignore_index=True)
ALL_DF.to_csv(P("data", "interventions.csv"), index=False)
ART["interventions_csv"] = P("data", "interventions.csv")
np.savez_compressed(P("data", "intervention_maps.npz"),
                    **{f"{k}_hot": v["hot"] for k, v in DATA.items()},
                    **{f"{k}_xy": v["xy"] for k, v in DATA.items()})
print(f"\ntotal oracle calls: {sum(o.calls for o in ORACLES.values())}   "
       f"wall clock: {time.time() - t0:.0f}s")

# ---- DRC hotspot regions: the specific 'violations' we will attribute blame for --------
REGIONS: Dict[str, List[dict]] = {}
for name, o in ORACLES.items():
    lab, ncc = _label(BASE[name].hotspot >= 1.0)
    mass = [(int((lab == i).sum()), float(BASE[name].hotspot[lab == i].sum()), i)
            for i in range(1, ncc + 1)]
    mass.sort(reverse=True)
    # The outcome we attribute blame for is "a violation still sits *here*", so the region is
    # dilated by two GCells: a hotspot that shifts by one GCell is the same hotspot.
    REGIONS[name] = [dict(rid=j, label=i, cells=c, drc=float(BASE[name].hotspot[_dilate(lab == i)].sum()),
                          core=(lab == i), mask=_dilate(lab == i))
                     for j, (c, m, i) in enumerate(mass[:3])]
    print(f"{name:11s} hotspot regions -> " +
          "  ".join(f"R{r['rid']}: {r['cells']} GCells / {r['drc']:.0f} markers"
                    for r in REGIONS[name]))
