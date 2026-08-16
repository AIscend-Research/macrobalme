# =====================================================================================
# 9. Congestion pricing: what does this macro cost *everybody else*?
# =====================================================================================
# Transport economics has a name for the quantity a designer actually wants. A driver entering
# a congested road pays their own travel time but not the delay they add to everyone behind
# them; the Pigou/Vickrey congestion price is exactly that unpriced externality, and the LP
# dual of the capacity constraint is its shadow price. A macro is an agent on the routing
# graph: it consumes capacity (blockage) and it generates demand (its nets), and both push
# cost onto nets that have nothing to do with it.
#
#   (a) interventional externality   E_m = C_-m(m at its site) - C_-m(m at the default site)
#       -- C_-m sums routed cost over the nets NOT incident to m.  Vickrey, measured exactly.
#   (b) dual / shadow price          p_m = sum_g lambda_g * (capacity m removes at g)
#                                        + sum_g lambda_g * (demand m's own nets place at g)
#       -- lambda_g is the router's negotiated-congestion history cost, which is a running
#       estimate of the Lagrange multiplier on GCell g's capacity constraint (PathFinder).
#       One run, no interventions, and it comes with a *price field* over candidate sites.

def others_cost(name, res: Result, m: int) -> float:
    o = ORACLES[name]
    mask = np.ones(DESIGNS[name].NNET, bool)
    mask[o.nets_of_macro[m]] = False
    return float(res.net_cost[mask].sum())


def pigou_interventional(name) -> np.ndarray:
    """The exact Vickrey externality, one real P&R run per macro."""
    d, o = DESIGNS[name], ORACLES[name]
    E = np.zeros(d.NM)
    for m in range(d.NM):
        xy = d.base_sites.copy(); xy[m] = d.null_sites[m]
        r_off = o.evaluate(xy)
        E[m] = others_cost(name, BASE[name], m) - others_cost(name, r_off, m)
    return E


def price_dual(name, res: Result = None, macro_xy=None) -> np.ndarray:
    """First-order shadow price from a single run: no extra P&R calls."""
    d, o = DESIGNS[name], ORACLES[name]
    res = res or BASE[name]
    macro_xy = res.macro_xy if macro_xy is None else macro_xy
    lam = res.lam
    p = np.zeros(d.NM)
    for m in range(d.NM):
        blockage = float((lam * o.macro_cover_one(macro_xy, m)).sum()) * o.base_cap * 0.92 * 2
        demand = float(res.net_cost[o.nets_of_macro[m]].sum())
        p[m] = blockage + demand
    return p


def _net_anchor(name, res: Result, m: int):
    """Where m's own nets pull it: the centroid of every *other* pin on each incident net."""
    d, o = DESIGNS[name], ORACLES[name]
    pos = np.empty((len(o.pin_node), 2))
    pos[o.pin_is_cell] = res.cell_xy[o.pin_cell_id[o.pin_is_cell]]
    pos[o.pin_is_macro] = res.macro_xy[o.pin_macro_id[o.pin_is_macro]]
    pos[o.pin_is_io] = d.io_xy[o.pin_io_id[o.pin_is_io]]
    nets = o.nets_of_macro[m]
    anchors = []
    for n_ in nets:
        sel = (o.pin_net == n_) & ~(o.pin_is_macro & (o.pin_macro_id == m))
        if sel.any(): anchors.append(pos[sel].mean(0))
    return np.array(anchors) if anchors else res.macro_xy[m][None]


def price_field(name, m, res: Result = None, lattice=17):
    """Two first-order prices over every legal site for macro m, from a single routed run:

        externality(s) = sum_g lambda_g * (capacity m removes at g if placed at s)
        private(s)     = m's own nets' congestion-weighted length if m sits at s

    The *score* we attribute to a macro is the externality alone -- that is the Pigouvian
    quantity. The *action* the price recommends is argmin (private + externality): charge the
    agent for the congestion it imposes and let it choose, which is what a congestion charge
    does. Minimising the externality alone would just park every macro in a corner."""
    d, o = DESIGNS[name], ORACLES[name]
    res = res or BASE[name]
    sites = candidate_sites(d, res.macro_xy, m, k=10 ** 9, lattice=lattice)
    ext = np.array([float((res.lam * o.macro_cover_one(
        np.concatenate([res.macro_xy[:m], s[None], res.macro_xy[m + 1:]]), m)).sum())
        for s in sites]) * o.base_cap * 0.92 * 2
    anc = _net_anchor(name, res, m)
    mid = (sites[:, None, :] + anc[None, :, :]) / 2
    gi = np.clip((mid / o.gs).astype(int), 0, o.G - 1)
    lam_mid = res.lam[gi[..., 1], gi[..., 0]]
    dist = np.abs(sites[:, None, :] - anc[None, :, :]).sum(-1) / o.gs
    priv = (dist * (1.0 + lam_mid)).sum(1)
    return sites, ext, priv


PRICE = {}
for name in DESIGNS:
    Ei = pigou_interventional(name)
    Pd = price_dual(name)
    PRICE[name] = dict(pigou=Ei, dual=Pd)
    d = DESIGNS[name]
    order = np.argsort(-Ei)
    print(f"{name:11s} rho(interventional, dual) = {spearman(Ei, Pd):+.2f}   "
          f"priciest macros: " + ", ".join(f"{d.macro_names[m]} ({Ei[m]:+.0f})" for m in order[:3]))

# the price also tells you where to put it
PRICE_MOVE = {}
for name in DESIGNS:
    d = DESIGNS[name]
    mv = {}
    for m in range(d.NM):
        s, ext, priv = price_field(name, m)
        tot = ext + priv
        j = int(np.argmin(tot))
        at = np.argmin(np.abs(s - d.base_sites[m]).sum(1))     # the site nearest to where it is
        mv[m] = dict(site=s[j], gain=float(tot[at] - tot[j]), sites=s, ext=ext, priv=priv,
                     total=tot)
    PRICE_MOVE[name] = mv
    best = max(mv, key=lambda m: mv[m]["gain"])
    print(f"{name:11s} price gradient says: move {d.macro_names[best]} from "
          f"({d.base_sites[best,0]:.0f},{d.base_sites[best,1]:.0f}) to "
          f"({mv[best]['site'][0]:.0f},{mv[best]['site'][1]:.0f})  "
          f"[predicted price drop {mv[best]['gain']:.0f}]")
