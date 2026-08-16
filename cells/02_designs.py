# =====================================================================================
# 2. Designs: synthetic-but-structured netlists in the shape of small ORFS blocks
# =====================================================================================
# A design carries: hard macros (SRAMs), standard-cell clusters grouped into modules,
# a clustered hypergraph netlist, and fixed IO pads. Node index space:
#     [0, NC)                cell clusters      (movable, placed by the oracle's placer)
#     [NC, NC+NM)            macros             (fixed by us -- the decision variables)
#     [NC+NM, NC+NM+NIO)     IO pads            (fixed on the die boundary)

@dataclass
class Design:
    name: str
    die: float
    macro_wh: np.ndarray        # (NM, 2)  macro widths/heights
    macro_names: List[str]
    base_sites: np.ndarray      # (NM, 2)  centres of the *actual* (as-given) floorplan
    null_sites: np.ndarray      # (NM, 2)  centres of the canonical peripheral "off" floorplan
    cell_area: np.ndarray       # (NC,)
    cell_module: np.ndarray     # (NC,)
    macro_module: np.ndarray    # (NM,)
    net_ptr: np.ndarray         # (NNET+1,) CSR pointer into net_nodes
    net_nodes: np.ndarray       # flat node ids
    io_xy: np.ndarray           # (NIO, 2)
    n_modules: int

    @property
    def NM(self): return len(self.macro_wh)
    @property
    def NC(self): return len(self.cell_area)
    @property
    def NIO(self): return len(self.io_xy)
    @property
    def NNET(self): return len(self.net_ptr) - 1


def _pack_periphery(macro_wh: np.ndarray, die: float, pad: float = 12.0) -> np.ndarray:
    """Canonical 'off' floorplan: macros hug the die boundary, largest first, walking the
    perimeter. This is what a human does with SRAMs when they are not fighting for area, and
    it is the reference (do(M = off)) setting for every counterfactual below."""
    order = np.argsort(-macro_wh.max(axis=1))
    sites = np.zeros_like(macro_wh)
    cursor, side = pad, 0                       # side: 0 bottom, 1 right, 2 top, 3 left
    for i in order:
        w, h = macro_wh[i]
        span = w if side % 2 == 0 else h
        depth = h if side % 2 == 0 else w
        if cursor + span + pad > die:           # move to the next edge
            side, cursor = (side + 1) % 4, pad
        c = cursor + span / 2
        d = pad + depth / 2
        sites[i] = {0: (c, d), 1: (die - d, c), 2: (die - c, die - d), 3: (d, die - c)}[side]
        cursor += span + pad
    return sites


def _random_legal_sites(rng, macro_wh, die, margin=25.0, tries=4000) -> np.ndarray:
    """A plausible but imperfect as-given floorplan: macros dropped in the core with a
    non-overlap constraint and a mild bias toward the centre, which is exactly the habit that
    manufactures congestion hotspots."""
    sites = np.zeros_like(macro_wh)
    placed = []
    for i in np.argsort(-macro_wh.prod(axis=1)):
        w, h = macro_wh[i]
        for _ in range(tries):
            cx = np.clip(rng.normal(die / 2, die / 5.5), margin + w / 2, die - margin - w / 2)
            cy = np.clip(rng.normal(die / 2, die / 5.5), margin + h / 2, die - margin - h / 2)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if all(not (box[0] < b[2] + 8 and b[0] - 8 < box[2] and
                        box[1] < b[3] + 8 and b[1] - 8 < box[3]) for b in placed):
                placed.append(box); sites[i] = (cx, cy); break
        else:                                    # give up: park it at the periphery
            sites[i] = _pack_periphery(macro_wh, die)[i]
            placed.append((sites[i, 0] - w / 2, sites[i, 1] - h / 2,
                           sites[i, 0] + w / 2, sites[i, 1] + h / 2))
    return sites


def make_design(name: str, n_macros: int, n_cells: int, n_nets: int, n_modules: int,
                die: float, seed: int) -> Design:
    rng = np.random.default_rng(seed)
    macro_wh = rng.uniform(0.075, 0.155, size=(n_macros, 2)) * die
    macro_wh[:, 1] *= rng.uniform(0.7, 1.4, size=n_macros)          # SRAMs are not square
    macro_wh = np.clip(macro_wh, 0.05 * die, 0.20 * die)
    macro_names = [f"M{i}" for i in range(n_macros)]

    cell_module = rng.integers(0, n_modules, size=n_cells)
    macro_module = rng.integers(0, n_modules, size=n_macros)        # each SRAM serves a module
    cell_area = rng.lognormal(0.0, 0.5, size=n_cells) * (0.45 * die * die / n_cells)

    NC, NM = n_cells, n_macros
    n_io = max(16, n_modules * 4)
    t = np.linspace(0, 4, n_io, endpoint=False)
    side, frac = np.floor(t).astype(int), t - np.floor(t)
    io_xy = np.stack([np.where(side == 0, frac * die, np.where(side == 1, die,
                      np.where(side == 2, (1 - frac) * die, 0.0))),
                      np.where(side == 0, 0.0, np.where(side == 1, frac * die,
                      np.where(side == 2, die, (1 - frac) * die)))], axis=1)

    # ---- clustered hypergraph: mostly intra-module, macros bound to their module ---------
    by_mod = [np.where(cell_module == m)[0] for m in range(n_modules)]
    macros_of = [np.where(macro_module == m)[0] for m in range(n_modules)]
    ptr, nodes = [0], []
    for _ in range(n_nets):
        m = int(rng.integers(0, n_modules))
        deg = int(np.clip(rng.lognormal(0.85, 0.55), 2, 7))
        pool = by_mod[m] if len(by_mod[m]) >= deg else np.arange(NC)
        pins = list(rng.choice(pool, size=min(deg, len(pool)), replace=False))
        if rng.random() < 0.14:                                     # cross-module net
            m2 = int(rng.integers(0, n_modules))
            if len(by_mod[m2]): pins.append(int(rng.choice(by_mod[m2])))
        if len(macros_of[m]) and rng.random() < 0.34:               # memory access net
            pins.append(NC + int(rng.choice(macros_of[m])))
        elif rng.random() < 0.05:
            pins.append(NC + int(rng.integers(0, NM)))
        if rng.random() < 0.08:                                     # primary IO
            pins.append(NC + NM + int(rng.integers(0, n_io)))
        pins = list(dict.fromkeys(pins))
        if len(pins) < 2: continue
        nodes.extend(pins); ptr.append(len(nodes))

    # standard-cell cluster area tracks its pin count (as it does in a real netlist); this is what
    # makes uniform-density placement also mean uniform *pin* density, hence smooth routing demand
    nodes_arr = np.asarray(nodes, dtype=np.int64)
    pin_cnt = np.bincount(nodes_arr[nodes_arr < NC], minlength=NC).astype(float)
    cell_area = cell_area * (0.4 + pin_cnt) / (0.4 + pin_cnt).mean()

    return Design(name=name, die=die, macro_wh=macro_wh, macro_names=macro_names,
                  base_sites=_random_legal_sites(rng, macro_wh, die),
                  null_sites=_pack_periphery(macro_wh, die),
                  cell_area=cell_area, cell_module=cell_module, macro_module=macro_module,
                  net_ptr=np.asarray(ptr), net_nodes=np.asarray(nodes, dtype=np.int64),
                  io_xy=io_xy, n_modules=n_modules)


# Sized after the small ORFS blocks the method is meant for (ibex / aes / jpeg on Nangate45).
_SPECS = [
    ("ibex_like",  8, 3000, 4600, 7, 1000.0),
    ("aes_like",   6, 2400, 3800, 5,  860.0),
    ("jpeg_like", 11, 3800, 5800, 9, 1180.0),
]
if QUICK:
    _SPECS = [(n, m, int(c * .55), int(nn * .55), k, d) for (n, m, c, nn, k, d) in _SPECS]

DESIGNS: Dict[str, Design] = {
    s[0]: make_design(s[0], s[1], s[2], s[3], s[4], s[5], CFG.seed + 17 * i)
    for i, s in enumerate(_SPECS)
}
for d in DESIGNS.values():
    print(f"{d.name:11s} macros={d.NM:3d} cell-clusters={d.NC:5d} nets={d.NNET:5d} "
          f"pins={len(d.net_nodes):6d} modules={d.n_modules}  die={d.die:.0f}um")
