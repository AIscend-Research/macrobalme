# =====================================================================================
# 7. Surrogate B -- a GNN over the macro graph: the amortised set-function v-hat
# =====================================================================================
# The causal quantities are functions on the lattice of macro configurations (2^NM corners,
# more once relocations are allowed). The oracle cannot be called that many times, so we fit
# v-hat(configuration) -> (total DRC, DRC inside each named hotspot region) on the real
# interventions and query *it* for the combinatorics -- then re-verify the top of every
# ranking with real oracle runs (section 11).

NMAX = max(d.NM for d in DESIGNS.values())
DESIGN_ID = {n: i for i, n in enumerate(DESIGNS)}

NETSHARE = {}
for name, o in ORACLES.items():
    d = DESIGNS[name]
    S = np.zeros((d.NM, d.NM))
    for i in range(d.NM):
        for j in range(d.NM):
            if i != j:
                S[i, j] = len(np.intersect1d(o.nets_of_macro[i], o.nets_of_macro[j]))
    S = S + (DESIGNS[name].macro_module[:, None] == DESIGNS[name].macro_module[None, :]) * 1.0
    np.fill_diagonal(S, 0)
    NETSHARE[name] = S / max(S.max(), 1.0)

MACRO_DEG = {n: np.array([len(ORACLES[n].nets_of_macro[m]) for m in range(DESIGNS[n].NM)], float)
             for n in DESIGNS}


def gnn_features(name: str, macro_xy: np.ndarray):
    """Node features and adjacency for one configuration. Pure numpy, ~40 us per call, which is
    what makes 10^5 counterfactual queries affordable."""
    d = DESIGNS[name]; NM = d.NM
    xy = macro_xy / d.die
    base = d.base_sites / d.die; null = d.null_sites / d.die
    deg = MACRO_DEG[name] / max(MACRO_DEG[name].max(), 1)
    off = (np.linalg.norm(macro_xy - d.null_sites, axis=1) < 1e-6).astype(float)
    dsp = xy - base
    ctr = np.linalg.norm(xy - 0.5, axis=1)
    wh = d.macro_wh / d.die
    # crowding: how much macro area sits within 0.2 die of me
    D = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1) + np.eye(NM) * 9
    crowd = ((wh.prod(1)[None, :] * np.exp(-(D / 0.18) ** 2)).sum(1))
    f = np.stack([xy[:, 0], xy[:, 1], wh[:, 0], wh[:, 1], deg, off, dsp[:, 0], dsp[:, 1],
                  ctr, crowd, np.linalg.norm(xy - null, axis=1),
                  np.full(NM, NM / NMAX), np.full(NM, d.NNET / 6000.0)], 1)
    A = NETSHARE[name] + np.exp(-(D / 0.22) ** 2) * (1 - np.eye(NM))
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
    F_ = np.zeros((NMAX, f.shape[1]), np.float32); F_[:NM] = f
    A_ = np.zeros((NMAX, NMAX), np.float32); A_[:NM, :NM] = A
    M_ = np.zeros(NMAX, np.float32); M_[:NM] = 1
    return F_, A_, M_


NF = gnn_features(list(DESIGNS)[0], DESIGNS[list(DESIGNS)[0]].base_sites)[0].shape[1]

# ---- assemble the training tensors in the same row order as the U-Net dataset ----------
GF, GA, GM, GID, GY = [], [], [], [], []
for name, v in DATA.items():
    masks = [r["mask"] for r in REGIONS[name]]
    while len(masks) < 3: masks.append(np.zeros_like(BASE[name].hotspot, bool))
    for i in range(len(v["df"])):
        f, a, m = gnn_features(name, v["xy"][i])
        GF.append(f); GA.append(a); GM.append(m); GID.append(DESIGN_ID[name])
        h = v["hot"][i].astype(np.float32)
        GY.append([h.sum()] + [float(h[mk].sum()) for mk in masks])
GF, GA, GM = np.stack(GF), np.stack(GA), np.stack(GM)
GID = np.array(GID); GY = np.log1p(np.stack(GY))
GY_MU, GY_SD = GY[TRN_IDX].mean(0), GY[TRN_IDX].std(0) + 1e-6
GYn = (GY - GY_MU) / GY_SD
print(f"GNN dataset: nodes<= {NMAX}, features {NF}, samples {len(GF)}, targets {GY.shape[1]}")

if HAS_TORCH:
    class MacroGNN(nn.Module):
        def __init__(s, nf=NF, w=96, layers=3, nd=len(DESIGNS)):
            super().__init__()
            s.emb = nn.Embedding(nd, 8)
            s.inp = nn.Linear(nf + 8, w)
            s.mp = nn.ModuleList([nn.Sequential(nn.Linear(3 * w, w), nn.SiLU(), nn.Linear(w, w))
                                  for _ in range(layers)])
            s.rd = nn.Sequential(nn.Linear(2 * w, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                                 nn.Linear(w, 4))
        def forward(s, f, a, m, did):
            e = s.emb(did)[:, None, :].expand(-1, f.shape[1], -1)
            h = torch.relu(s.inp(torch.cat([f, e], -1))) * m[..., None]
            for L in s.mp:
                agg = torch.bmm(a, h)
                glob = (h.sum(1) / m.sum(1, keepdim=True).clamp(min=1))[:, None].expand_as(h)
                h = (h + L(torch.cat([h, agg, glob], -1))) * m[..., None]
            pooled = torch.cat([h.sum(1) / m.sum(1, keepdim=True).clamp(min=1), h.max(1).values], -1)
            return s.rd(pooled)

    def train_gnn():
        net = MacroGNN().to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 4e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.gnn_epochs)
        f = torch.tensor(GF, device=DEV); a = torch.tensor(GA, device=DEV)
        m = torch.tensor(GM, device=DEV); did = torch.tensor(GID, device=DEV)
        y = torch.tensor(GYn, dtype=torch.float32, device=DEV)
        tr = torch.tensor(TRN_IDX, device=DEV); va = torch.tensor(VAL_IDX, device=DEV)
        hist = []
        for ep in range(CFG.gnn_epochs):
            net.train()
            p = net(f[tr], a[tr], m[tr], did[tr]); loss = F.mse_loss(p, y[tr])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad():
                vl = float(F.mse_loss(net(f[va], a[va], m[va], did[va]), y[va]))
            hist.append((float(loss), vl))
            if ep % max(1, CFG.gnn_epochs // 6) == 0 or ep == CFG.gnn_epochs - 1:
                print(f"  ep {ep:4d}  train {float(loss):.4f}  val {vl:.4f}")
        return net, np.array(hist)

    t0 = time.time(); GNN, GHIST = train_gnn()
    print(f"GNN trained in {time.time() - t0:.0f}s")

    @torch.no_grad()
    def v_hat_batch(name, XYs) -> np.ndarray:
        """Amortised counterfactual: array of configurations -> (n, 4) predicted
        [total DRC, DRC in R0, R1, R2]."""
        out = []
        did = DESIGN_ID[name]
        for b in range(0, len(XYs), 512):
            chunk = XYs[b:b + 512]
            fs, as_, ms = zip(*[gnn_features(name, x) for x in chunk])
            p = GNN(torch.tensor(np.stack(fs), device=DEV), torch.tensor(np.stack(as_), device=DEV),
                    torch.tensor(np.stack(ms), device=DEV),
                    torch.full((len(chunk),), did, device=DEV, dtype=torch.long))
            out.append(p.cpu().numpy())
        return np.expm1(np.concatenate(out) * GY_SD + GY_MU)
else:                                                          # pragma: no cover
    GNN, GHIST = None, np.zeros((1, 2))
    _Z = np.concatenate([GF.reshape(len(GF), -1), GM], 1); _Z = np.c_[_Z, np.ones(len(_Z))]
    _WG = np.linalg.lstsq(_Z[TRN_IDX], GYn[TRN_IDX], rcond=1e-3)[0]
    def v_hat_batch(name, XYs):
        z = np.stack([np.r_[gnn_features(name, x)[0].ravel(), gnn_features(name, x)[2], 1.0]
                      for x in XYs])
        return np.expm1((z @ _WG) * GY_SD + GY_MU)

def v_hat(name, xy): return v_hat_batch(name, np.asarray(xy)[None])[0]

GNN_FID = []
for name in DESIGNS:
    idx = [i for i in VAL_IDX if D_ALL[i] == name]
    loc = [list(np.where(D_ALL == name)[0]).index(i) for i in idx]
    pred = v_hat_batch(name, DATA[name]["xy"][loc])[:, 0]
    true = np.expm1(GY[idx, 0])
    GNN_FID.append(dict(design=name, n=len(idx), spearman=spearman(true, pred),
                        mae=float(np.abs(true - pred).mean()),
                        rel_mae=float((np.abs(true - pred) / np.maximum(true, 1)).mean())))
    print(f"{name:11s} held-out DRC   Spearman {GNN_FID[-1]['spearman']:+.3f}   "
          f"MAE {GNN_FID[-1]['mae']:.1f} markers ({100 * GNN_FID[-1]['rel_mae']:.1f}%)")
pd.DataFrame(GNN_FID).to_csv(P("tables", "gnn_fidelity.csv"), index=False)
t0 = time.time(); _ = v_hat_batch(list(DESIGNS)[0], np.repeat(DESIGNS[list(DESIGNS)[0]].base_sites[None], 512, 0))
print(f"amortised query cost: {(time.time() - t0) / 512 * 1e6:.0f} us per counterfactual "
      f"vs {ORACLE_SECONDS_PER_CALL * 1e6:.0f} us for a real P&R run "
      f"({ORACLE_SECONDS_PER_CALL / ((time.time() - t0) / 512):.0f}x)")
