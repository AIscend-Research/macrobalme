# =====================================================================================
# 6. Surrogate A -- a U-Net that predicts the DRC-marker map from the macro floorplan
# =====================================================================================
# Inputs are functions of the macro placement ONLY (no leakage from the oracle's own
# standard-cell placement or routing), so the network is a differentiable stand-in for the
# whole flow: floorplan -> placement -> routing -> DRC. That differentiability is exactly
# what the saliency baselines need, and it is what makes them look reasonable a priori.

def unet_inputs(name: str, macro_xy: np.ndarray) -> np.ndarray:
    d, o = DESIGNS[name], ORACLES[name]
    G, gs = o.G, o.gs
    cov = o.macro_density(macro_xy)
    covb = _gauss(cov, max(1, G // 10))
    deg = np.array([len(o.nets_of_macro[m]) for m in range(d.NM)], float)
    deg = deg / max(deg.max(), 1.0)
    pinf = np.zeros((G, G))
    yy, xx = np.mgrid[0:G, 0:G] + 0.5
    for m in range(d.NM):
        cx, cy = macro_xy[m] / gs
        s = max(1.5, (d.macro_wh[m].mean() / gs))
        pinf += deg[m] * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    iof = np.zeros((G, G))
    for x, y in d.io_xy:
        iof[min(int(y / gs), G - 1), min(int(x / gs), G - 1)] += 1
    iof = _gauss(iof, max(1, G // 14)); iof /= max(iof.max(), 1e-9)
    free = _gauss(1.0 - cov, max(1, G // 8))
    free = free / max(free.mean(), 1e-9)
    rx = np.tile(np.linspace(-1, 1, G), (G, 1)); ry = rx.T
    return np.stack([cov, covb, pinf / max(pinf.max(), 1e-9), iof, free / 2.0, rx, ry]).astype(np.float32)


CH_IN = 7
X_ALL, Y_ALL, D_ALL = [], [], []
for name, v in DATA.items():
    for i in range(len(v["df"])):
        X_ALL.append(unet_inputs(name, v["xy"][i])); Y_ALL.append(v["hot"][i].astype(np.float32))
        D_ALL.append(name)
X_ALL = np.stack(X_ALL); Y_ALL = np.stack(Y_ALL); D_ALL = np.array(D_ALL)
Y_LOG = np.log1p(Y_ALL)
rng = np.random.default_rng(CFG.seed)
perm = rng.permutation(len(X_ALL))
n_val = max(12, int(0.18 * len(perm)))
VAL_IDX, TRN_IDX = perm[:n_val], perm[n_val:]
print(f"U-Net dataset: {X_ALL.shape}  train {len(TRN_IDX)}  val {len(VAL_IDX)}")


if HAS_TORCH:
    class Block(nn.Module):
        def __init__(s, i, o_):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(i, o_, 3, padding=1), nn.GroupNorm(4, o_), nn.SiLU(),
                                nn.Conv2d(o_, o_, 3, padding=1), nn.GroupNorm(4, o_), nn.SiLU())
        def forward(s, x): return s.f(x)

    class UNet(nn.Module):
        """3-level U-Net; a scalar DRC head hangs off the bottleneck so the same network gives
        both the marker map and the total, and so d(DRC)/d(input) is well defined."""
        def __init__(s, cin=CH_IN, w=32):
            super().__init__()
            s.e1, s.e2, s.e3 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w)
            s.d2, s.d1 = Block(4 * w + 2 * w, 2 * w), Block(2 * w + w, w)
            s.out = nn.Conv2d(w, 1, 1)
            s.head = nn.Sequential(nn.Linear(4 * w, 64), nn.SiLU(), nn.Linear(64, 1))
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(F.avg_pool2d(e1, 2)); e3 = s.e3(F.avg_pool2d(e2, 2))
            u2 = s.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
            u1 = s.d1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode="nearest"), e1], 1))
            m = F.softplus(s.out(u1))[:, 0]
            g = F.softplus(s.head(e3.mean((-1, -2))))[:, 0]
            return m, g

    def train_unet():
        net = UNet().to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.unet_epochs)
        Xt = torch.tensor(X_ALL, device=DEV); Yt = torch.tensor(Y_LOG, device=DEV)
        Gt = torch.log1p(torch.tensor(Y_ALL.sum((1, 2)), device=DEV))
        hist = []
        bs = 16
        for ep in range(CFG.unet_epochs):
            net.train(); idx = TRN_IDX[torch.randperm(len(TRN_IDX)).numpy()]
            tot = 0.0
            for b in range(0, len(idx), bs):
                j = idx[b:b + bs]
                m, g = net(Xt[j])
                loss = F.mse_loss(m, Yt[j]) + 0.35 * F.mse_loss(g, Gt[j])
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
            sch.step()
            net.eval()
            with torch.no_grad():
                m, g = net(Xt[VAL_IDX])
                vl = float(F.mse_loss(m, Yt[VAL_IDX])); vg = float(F.mse_loss(g, Gt[VAL_IDX]))
            hist.append((tot / len(idx), vl, vg))
            if ep % max(1, CFG.unet_epochs // 8) == 0 or ep == CFG.unet_epochs - 1:
                print(f"  ep {ep:3d}  train {hist[-1][0]:.4f}  val-map {vl:.4f}  val-drc {vg:.4f}")
        return net, np.array(hist)

    t0 = time.time(); UNET, UHIST = train_unet()
    print(f"U-Net trained in {time.time() - t0:.0f}s on {DEV}")

    @torch.no_grad()
    def unet_predict(name, macro_xy):
        x = torch.tensor(unet_inputs(name, macro_xy)[None], device=DEV)
        m, g = UNET(x)
        return np.expm1(m[0].cpu().numpy()), float(np.expm1(g.cpu().numpy()[0]))
else:                                                          # pragma: no cover
    UNET, UHIST = None, np.zeros((1, 3))
    _A = X_ALL.reshape(len(X_ALL), -1); _A = np.c_[_A, np.ones(len(_A))]
    _W = np.linalg.lstsq(_A[TRN_IDX], Y_LOG.reshape(len(X_ALL), -1)[TRN_IDX], rcond=1e-3)[0]
    def unet_predict(name, macro_xy):
        x = np.r_[unet_inputs(name, macro_xy).ravel(), 1.0]
        m = np.expm1(x @ _W).reshape(X_ALL.shape[-2:])
        return m, float(m.sum())

# ---- held-out fidelity, per design ----------------------------------------------------
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / max(np.sqrt((ra @ ra) * (rb @ rb)), 1e-12))

UNET_FID = []
for name in DESIGNS:
    idx = [i for i in VAL_IDX if D_ALL[i] == name]
    true = Y_ALL[idx].sum((1, 2))
    pred = np.array([unet_predict(name, DATA[name]["xy"][j])[1]
                     for j in [list(np.where(D_ALL == name)[0]).index(i) for i in idx]])
    UNET_FID.append(dict(design=name, n=len(idx), spearman=spearman(true, pred),
                         mae=float(np.abs(true - pred).mean()),
                         rel_mae=float((np.abs(true - pred) / np.maximum(true, 1)).mean())))
    print(f"{name:11s} held-out DRC   Spearman {UNET_FID[-1]['spearman']:+.3f}   "
          f"MAE {UNET_FID[-1]['mae']:.1f} markers ({100 * UNET_FID[-1]['rel_mae']:.1f}%)")
pd.DataFrame(UNET_FID).to_csv(P("tables", "unet_fidelity.csv"), index=False)
