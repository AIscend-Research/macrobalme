# =====================================================================================
# 3. The interventional oracle: an ePlace-style placer + a PathFinder-style global router
# =====================================================================================
# Oracle.evaluate(macro_xy) -> Result is the ONLY interface the causal machinery uses.
# Swap in ORFSOracle (final section) to run the identical experiments against OpenROAD.

@dataclass
class Result:
    drc: float                  # DRC-hotspot proxy = total positive routing overflow
    n_hotspot: int              # GCells whose overflow crosses the violation threshold
    wl: float                   # routed wirelength (GCell units)
    hotspot: np.ndarray         # (G,G) per-GCell violation map  -- the "DRC marker" layer
    ovfl: np.ndarray            # (G,G) continuous overflow (h+v)
    usage: np.ndarray           # (G,G) total usage
    cap: np.ndarray             # (G,G) total capacity
    lam: np.ndarray             # (G,G) PathFinder history cost == estimated capacity duals
    cell_xy: np.ndarray         # (NC,2) placer output -- macros move, cells follow
    net_cost: np.ndarray        # (NNET,) per-net routed cost, for the externality accounting
    macro_xy: np.ndarray


def _box_blur(a, r):
    if r <= 0: return a
    k = 2 * r + 1
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), axis=0)
    a = (c[k:] - c[:-k]) / k
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def _gauss(a, r):
    return _box_blur(_box_blur(_box_blur(a, r), r), r)


class Oracle:
    """Deterministic function of the macro placement. Everything downstream -- the standard-cell
    placement, the routing topology, the congestion -- is recomputed from scratch, which is what
    makes an intervention here a genuine do() on the design rather than a perturbation of an input
    tensor."""

    def __init__(self, design: Design, cfg: Config):
        self.d, self.cfg = design, cfg
        self.G = cfg.grid
        self.gs = design.die / self.G                     # GCell size
        rng = np.random.default_rng(CFG.seed ^ hash(design.name) % (2**31))
        # deterministic starting point for the placer: identical for every intervention, so any
        # difference in the final placement is *caused* by the macro move
        self.cell_init = rng.uniform(0.12, 0.88, size=(design.NC, 2)) * design.die
        self.macro_pin_off = np.array([[.5, 0], [0, .5], [-.5, 0], [0, -.5]])
        self.calls = 0
        self._precompute_nets()
        self._calibrate()

    def _calibrate(self):
        """Track capacity is a technology constant, not a function of the placement, so we fix it
        once: route the canonical peripheral floorplan with unlimited capacity and set the per-GCell
        supply from a high quantile of that demand. Every intervention then sees the same supply."""
        self.base_cap = 1e9
        r = self.route(self.place(self.d.null_sites), self.d.null_sites, seed=0)
        u = np.concatenate([r["uh"].ravel(), r["uv"].ravel()])
        self.base_cap = float(max(4.0, self.cfg.cap_k * np.percentile(u, self.cfg.cap_pct)))

    # -- netlist bookkeeping -----------------------------------------------------------
    def _precompute_nets(self):
        d = self.d
        ptr, nodes = d.net_ptr, d.net_nodes
        self.pin_net = np.repeat(np.arange(d.NNET), np.diff(ptr))
        self.pin_node = nodes
        self.pin_is_cell = nodes < d.NC
        self.pin_is_macro = (nodes >= d.NC) & (nodes < d.NC + d.NM)
        self.pin_is_io = nodes >= d.NC + d.NM
        self.pin_macro_id = np.where(self.pin_is_macro, nodes - d.NC, 0)
        self.pin_io_id = np.where(self.pin_is_io, nodes - d.NC - d.NM, 0)
        self.pin_cell_id = np.where(self.pin_is_cell, nodes, 0)
        self.pin_macro_face = np.arange(len(nodes)) % 4
        self.net_deg = np.diff(ptr)
        # nets touching each macro -- needed for the "cost to everyone else" accounting
        self.nets_of_macro = [np.unique(self.pin_net[self.pin_is_macro & (self.pin_macro_id == m)])
                              for m in range(d.NM)]
        # cell -> nets incidence for the wirelength force
        self.cell_pin_idx = np.where(self.pin_is_cell)[0]

    # -- macro geometry ----------------------------------------------------------------
    def macro_density(self, macro_xy):
        """Fractional GCell coverage by hard macros (exact box/GCell overlap area)."""
        d, G, gs = self.d, self.G, self.gs
        cov = np.zeros((G, G))
        edges = np.arange(G + 1) * gs
        for i in range(d.NM):
            w, h = d.macro_wh[i]
            x0, x1 = macro_xy[i, 0] - w / 2, macro_xy[i, 0] + w / 2
            y0, y1 = macro_xy[i, 1] - h / 2, macro_xy[i, 1] + h / 2
            ox = np.clip(np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0), 0, None)
            oy = np.clip(np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0), 0, None)
            cov += np.outer(oy, ox) / (gs * gs)
        return np.clip(cov, 0, 1)

    def macro_cover_one(self, macro_xy, m):
        d, G, gs = self.d, self.G, self.gs
        edges = np.arange(G + 1) * gs
        w, h = d.macro_wh[m]
        x0, x1 = macro_xy[m, 0] - w / 2, macro_xy[m, 0] + w / 2
        y0, y1 = macro_xy[m, 1] - h / 2, macro_xy[m, 1] + h / 2
        ox = np.clip(np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0), 0, None)
        oy = np.clip(np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0), 0, None)
        return np.clip(np.outer(oy, ox) / (gs * gs), 0, 1)

    # -- stage 1: analytical placement of the standard-cell clusters --------------------
    def _shift(self, xy, area, mcov, axis, nb=14, blend=0.55):
        """Bin-based cell shifting (Gordian-L / Kraftwerk style): inside each slice, remap the
        cells monotonically so that they fill the *free* area uniformly. Macro coverage removes
        free area, so cells flow around hard blockages instead of piling on top of them."""
        d, G = self.d, self.G
        die = d.die
        oth = 1 - axis
        edges = np.linspace(0, die, G + 1)
        occ_free = 1.0 - (mcov if axis == 0 else mcov.T)      # rows indexed by the other axis
        sl = np.clip((xy[:, oth] / die * nb).astype(int), 0, nb - 1)
        rows_per = G / nb
        new = xy[:, axis].copy()
        for j in range(nb):
            idx = np.where(sl == j)[0]
            if len(idx) < 4: continue
            r0, r1 = int(j * rows_per), max(int(j * rows_per) + 1, int((j + 1) * rows_per))
            free = occ_free[r0:r1].sum(axis=0)                # (G,) free width per bin
            cfree = np.concatenate([[0.0], np.cumsum(np.maximum(free, 1e-3))])
            order = idx[np.argsort(xy[idx, axis])]
            ca = np.cumsum(area[order]); ca = (ca - 0.5 * area[order]) / ca[-1]
            new[order] = np.interp(ca * cfree[-1], cfree, edges)
        xy[:, axis] = (1 - blend) * xy[:, axis] + blend * new
        return xy

    def place(self, macro_xy):
        """ePlace-lite: star-model wirelength force + a density force, with periodic bin-based
        shifting to keep the standard-cell density uniform over the free area. Hard macros enter
        as fixed charge, so moving a macro genuinely re-places the whole block."""
        d, cfg, G, gs = self.d, self.cfg, self.G, self.gs
        xy = self.cell_init.copy()
        area = d.cell_area / d.cell_area.mean()
        mcov = self.macro_density(macro_xy)
        io = d.io_xy
        pn, pnode = self.pin_net, self.pin_node
        v = np.zeros_like(xy)
        for it in range(cfg.place_iters):
            # ---- wirelength: pull every pin toward its net centroid (star model) --------
            pos = np.empty((len(pnode), 2))
            pos[self.pin_is_cell] = xy[self.pin_cell_id[self.pin_is_cell]]
            pos[self.pin_is_macro] = (macro_xy[self.pin_macro_id[self.pin_is_macro]] +
                                      self.macro_pin_off[self.pin_macro_face[self.pin_is_macro]] *
                                      d.macro_wh[self.pin_macro_id[self.pin_is_macro]])
            pos[self.pin_is_io] = io[self.pin_io_id[self.pin_is_io]]
            cen = np.zeros((d.NNET, 2)); np.add.at(cen, pn, pos)
            cen /= self.net_deg[:, None]
            gwl = np.zeros_like(xy)
            ci = self.cell_pin_idx
            np.add.at(gwl, self.pin_cell_id[ci], pos[ci] - cen[pn[ci]])
            # ---- density: push down the gradient of the blurred occupancy field ---------
            gx = np.clip((xy[:, 0] / gs).astype(int), 0, G - 1)
            gy = np.clip((xy[:, 1] / gs).astype(int), 0, G - 1)
            dens = np.zeros((G, G)); np.add.at(dens, (gy, gx), area)
            dens = dens / max(dens.mean(), 1e-9) + 8.0 * mcov     # macros = large fixed charge
            phi = _gauss(dens, max(1, G // 12))
            fy, fx = np.gradient(phi)
            g = 0.05 * gwl + 0.9 * gs * np.stack([fx[gy, gx], fy[gy, gx]], 1)
            v = 0.7 * v - 0.5 * g
            xy = np.clip(xy + v, 2.0, d.die - 2.0)
            if it % 5 == 4 and it < cfg.place_iters - 2:          # keep density uniform
                xy = self._shift(xy, area, mcov, it // 5 % 2)
                xy = np.clip(xy, 2.0, d.die - 2.0)
        # ---- hard-blockage cleanup: no standard cell may sit inside a macro -------------
        for i in range(d.NM):
            w, h = d.macro_wh[i]; cx, cy = macro_xy[i]
            dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
            inside = (np.abs(dx) < w / 2) & (np.abs(dy) < h / 2)
            if inside.any():
                px = (w / 2 - np.abs(dx[inside])); py = (h / 2 - np.abs(dy[inside]))
                usex = px < py
                ii = np.where(inside)[0]
                xy[ii[usex], 0] = cx + np.sign(dx[inside][usex] + 1e-9) * (w / 2 + 1.5)
                xy[ii[~usex], 1] = cy + np.sign(dy[inside][~usex] + 1e-9) * (h / 2 + 1.5)
        return np.clip(xy, 1.0, d.die - 1.0)

    # -- stage 2: negotiated-congestion global routing ---------------------------------
    def _segments(self, cell_xy, macro_xy):
        """Pins -> GCell coords -> 2-pin segments by chaining each net's pins in x-major order
        (a cheap single-trunk RSMT that FastRoute-class routers use as their starting topology)."""
        d, G, gs = self.d, self.G, self.gs
        pos = np.empty((len(self.pin_node), 2))
        pos[self.pin_is_cell] = cell_xy[self.pin_cell_id[self.pin_is_cell]]
        pos[self.pin_is_macro] = (macro_xy[self.pin_macro_id[self.pin_is_macro]] +
                                  self.macro_pin_off[self.pin_macro_face[self.pin_is_macro]] *
                                  d.macro_wh[self.pin_macro_id[self.pin_is_macro]])
        pos[self.pin_is_io] = d.io_xy[self.pin_io_id[self.pin_is_io]]
        gx = np.clip((pos[:, 0] / gs).astype(np.int64), 0, G - 1)
        gy = np.clip((pos[:, 1] / gs).astype(np.int64), 0, G - 1)
        order = np.lexsort((gy, gx, self.pin_net))
        n_, x_, y_ = self.pin_net[order], gx[order], gy[order]
        keep = n_[1:] == n_[:-1]
        return (n_[:-1][keep], x_[:-1][keep], y_[:-1][keep], x_[1:][keep], y_[1:][keep])

    def route(self, cell_xy, macro_xy, seed=0):
        d, cfg, G = self.d, self.cfg, self.G
        rng = np.random.default_rng(seed)
        snet, x1, y1, x2, y2 = self._segments(cell_xy, macro_xy)
        S = len(x1)
        # ---- capacity: base tracks, minus macro blockage, minus local pin obstruction ----
        mcov = self.macro_density(macro_xy)
        pin = np.zeros((G, G))
        cgx = np.clip((cell_xy[:, 0] / self.gs).astype(int), 0, G - 1)
        cgy = np.clip((cell_xy[:, 1] / self.gs).astype(int), 0, G - 1)
        np.add.at(pin, (cgy, cgx), 1.0)
        pin = pin / max(pin.mean(), 1e-9)
        cap_h = self.base_cap * np.clip(1.0 - 0.92 * mcov, 0.02, 1.0) * np.clip(1.0 - 0.06 * pin, 0.35, 1.0)
        cap_v = (self.base_cap * np.clip(1.0 - 0.92 * mcov, 0.02, 1.0)
                 * np.clip(1.0 - 0.06 * pin, 0.35, 1.0))
        hist_h = np.zeros((G, G)); hist_v = np.zeros((G, G))
        uh = np.zeros((G, G)); uv = np.zeros((G, G))
        best = None

        def prefix(ch, cv):
            return (np.pad(np.cumsum(ch, axis=1), ((0, 0), (1, 0))),
                    np.pad(np.cumsum(cv, axis=0), ((1, 0), (0, 0))))

        lo_x, hi_x = np.minimum(x1, x2), np.maximum(x1, x2)
        lo_y, hi_y = np.minimum(y1, y2), np.maximum(y1, y2)
        xm = x2.copy(); ym = y2.copy(); useA = np.ones(S, bool)

        for it in range(cfg.route_iters):
            ch = (1.0 + hist_h) * (1.0 + 9.0 * np.clip(uh - cap_h, 0, None) / cap_h)
            cv = (1.0 + hist_v) * (1.0 + 9.0 * np.clip(uv - cap_v, 0, None) / cap_v)
            Sx, Sy = prefix(ch, cv)
            det = 1 + 2 * it                      # detour budget grows as negotiation proceeds
            # Two route families, both single-trunk:
            #   A: horizontal-vertical-horizontal, trunk column xm   (xm = x1 or x2 -> an L)
            #   B: vertical-horizontal-vertical,  trunk row    ym
            XM = np.stack([x1, x2, (x1 + x2) // 2,
                           rng.integers(np.maximum(lo_x - det, 0), np.minimum(hi_x + det, G - 1) + 1),
                           np.clip(lo_x - rng.integers(1, det + 2, S), 0, G - 1),
                           np.clip(hi_x + rng.integers(1, det + 2, S), 0, G - 1)], 1)
            YM = np.stack([y1, y2, (y1 + y2) // 2,
                           rng.integers(np.maximum(lo_y - det, 0), np.minimum(hi_y + det, G - 1) + 1),
                           np.clip(lo_y - rng.integers(1, det + 2, S), 0, G - 1),
                           np.clip(hi_y + rng.integers(1, det + 2, S), 0, G - 1)], 1)
            def hseg(row, a_, b_):                # cost of a horizontal run, prefix-sum lookup
                l, h_ = np.minimum(a_, b_), np.maximum(a_, b_)
                return Sx[row, h_ + 1] - Sx[row, l]
            def vseg(col, a_, b_):
                l, h_ = np.minimum(a_, b_), np.maximum(a_, b_)
                return Sy[h_ + 1, col] - Sy[l, col]
            cA = (hseg(y1[:, None], x1[:, None], XM) + hseg(y2[:, None], XM, x2[:, None]) +
                  vseg(XM, y1[:, None], y2[:, None]))
            cB = (vseg(x1[:, None], y1[:, None], YM) + vseg(x2[:, None], YM, y2[:, None]) +
                  hseg(YM, x1[:, None], x2[:, None]))
            iA, iB = np.argmin(cA, 1), np.argmin(cB, 1)
            xm, ym = XM[np.arange(S), iA], YM[np.arange(S), iB]
            useA = cA[np.arange(S), iA] <= cB[np.arange(S), iB]
            # ---- rip up everything, lay the new routes down (difference-array scatter) ---
            dh = np.zeros((G, G + 1)); dv = np.zeros((G + 1, G))
            def addh(row, a_, b_):
                l, h_ = np.minimum(a_, b_), np.maximum(a_, b_)
                np.add.at(dh, (row, l), 1.0); np.add.at(dh, (row, h_ + 1), -1.0)
            def addv(col, a_, b_):
                l, h_ = np.minimum(a_, b_), np.maximum(a_, b_)
                np.add.at(dv, (l, col), 1.0); np.add.at(dv, (h_ + 1, col), -1.0)
            A, B = useA, ~useA
            addh(y1[A], x1[A], xm[A]); addh(y2[A], xm[A], x2[A]); addv(xm[A], y1[A], y2[A])
            addv(x1[B], y1[B], ym[B]); addv(x2[B], ym[B], y2[B]); addh(ym[B], x1[B], x2[B])
            uh = np.cumsum(dh, axis=1)[:, :G]
            uv = np.cumsum(dv, axis=0)[:G, :]
            # ---- PathFinder history update: hist_g is the running Lagrange multiplier on
            #      GCell g's capacity constraint -- reused below as the congestion shadow price.
            hist_h += 0.45 * np.clip(uh - cap_h, 0, None) / cap_h
            hist_v += 0.45 * np.clip(uv - cap_v, 0, None) / cap_v
            tot_of = float(np.clip(uh - cap_h, 0, None).sum() + np.clip(uv - cap_v, 0, None).sum())
            if best is None or tot_of < best[0]:          # keep the best round, as real routers do
                best = (tot_of, uh.copy(), uv.copy(), xm.copy(), ym.copy(), useA.copy())

        _, uh, uv, xm, ym, useA = best
        # ---- per-net cost, for the "externality on everyone else" accounting -------------
        seg_cost = np.where(useA,
                            np.abs(x1 - xm) + np.abs(xm - x2) + np.abs(y1 - y2),
                            np.abs(y1 - ym) + np.abs(ym - y2) + np.abs(x1 - x2))
        lam = hist_h + hist_v
        mx = np.where(useA, xm, (x1 + x2) // 2); my = np.where(useA, (y1 + y2) // 2, ym)
        seg_pen = lam[y1, np.clip(mx, 0, G - 1)] + lam[np.clip(my, 0, G - 1), mx] + lam[y2, np.clip(mx, 0, G - 1)]
        net_cost = np.zeros(d.NNET)
        np.add.at(net_cost, snet, seg_cost * (1.0 + seg_pen))

        oh = np.clip(uh - cap_h, 0, None); ov = np.clip(uv - cap_v, 0, None)
        ovfl = oh + ov
        # DRC markers: a GCell is written up once per v_unit of overflow beyond a tolerance band,
        # which is how a signoff DRC deck turns continuous congestion into a countable violation.
        cap_t = cap_h + cap_v
        v_unit = 0.25 * self.base_cap
        hotspot = np.clip(ovfl - 0.10 * cap_t, 0, None) / v_unit
        return dict(uh=uh, uv=uv, cap_h=cap_h, cap_v=cap_v, ovfl=ovfl, hotspot=hotspot,
                    lam=lam, wl=float(seg_cost.sum()), net_cost=net_cost)

    # -- the oracle call ----------------------------------------------------------------
    def evaluate(self, macro_xy, noise=0.0) -> Result:
        macro_xy = np.asarray(macro_xy, float)
        cell_xy = self.place(macro_xy)
        seed = int(hashlib.md5(np.round(macro_xy, 3).tobytes()).hexdigest()[:8], 16)
        r = self.route(cell_xy, macro_xy, seed=seed)
        self.calls += 1
        drc = float(r["hotspot"].sum())
        if noise:                                             # optional tool nondeterminism
            drc *= float(np.random.default_rng(seed + 1).normal(1.0, noise))
        return Result(drc=drc, n_hotspot=int((r["hotspot"] >= 1.0).sum()), wl=r["wl"],
                      hotspot=r["hotspot"], ovfl=r["ovfl"], usage=r["uh"] + r["uv"],
                      cap=r["cap_h"] + r["cap_v"], lam=r["lam"], cell_xy=cell_xy,
                      net_cost=r["net_cost"], macro_xy=macro_xy.copy())


ORACLES = {k: Oracle(d, CFG) for k, d in DESIGNS.items()}
BASE: Dict[str, Result] = {}
for k, o in ORACLES.items():
    t0 = time.time()
    BASE[k] = o.evaluate(DESIGNS[k].base_sites)
    t = time.time() - t0
    nb = o.evaluate(DESIGNS[k].null_sites)
    print(f"{k:11s} base DRC={BASE[k].drc:8.1f} hotspot-GCells={BASE[k].n_hotspot:4d} "
          f"WL={BASE[k].wl:8.0f} | peripheral-'off' DRC={nb.drc:7.1f} | {t*1000:6.0f} ms/call")
ORACLE_SECONDS_PER_CALL = t
