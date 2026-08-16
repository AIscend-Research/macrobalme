# =====================================================================================
# 8. Halpern-Pearl actual causation: necessity, sufficiency, responsibility, blame
# =====================================================================================
# Causal model.  Endogenous variables X_1..X_NM, one per macro, ranging over sites.  The
# actual context is the as-given floorplan (X_m = base_m); the default/"off" setting is the
# canonical peripheral site (null_m).  The outcome phi is a *specific* violation:
#     phi_R  ==  "the DRC hotspot in region R survives", i.e. markers(R) >= tau * markers_base(R)
# and phi_all == "the block still fails signoff", i.e. total DRC >= tau * base.
# Everything below is a query on that model; the surrogate answers them, the oracle audits them.

# A violation "survives" if enough of its marker mass remains. The block-level outcome uses a
# signoff-budget threshold (still failing by most of its original violation count); a named
# hotspot uses a laxer one, because a hotspot that halves has, for the designer, been fixed.
TAU_ALL, TAU_REGION = 0.80, 0.55
K_EPI = 24 if not QUICK else 12  # epistemic samples (uncertainty over the designer's context)
OUTCOMES = ["all", "R0", "R1", "R2"]


def config_from_mask(name, on_mask):
    d = DESIGNS[name]
    return np.where(np.asarray(on_mask, bool)[:, None], d.base_sites, d.null_sites)


# X_m ranges over the sites the designer can actually put macro m at: its default peripheral
# slot, or any ECO-sized relocation. HP quantifies over the alternative settings of the
# variable, so this action set *is* the contrast set -- both for "would it have happened but
# for M3" and for the witness sets W.
NALT = 3
CONTRAST = {n: {m: np.concatenate([DESIGNS[n].null_sites[m][None], LOCAL_SITES[n][m][:NALT]])
                for m in range(DESIGNS[n].NM)} for n in DESIGNS}
ALT = {n: {m: CONTRAST[n][m][1:] for m in CONTRAST[n]} for n in DESIGNS}
BASE_OUT = {n: np.array([BASE[n].drc] + [float(BASE[n].hotspot[r["mask"]].sum())
                                         for r in REGIONS[n]] + [0.0] * 3)[:4] for n in DESIGNS}
THRESH = {n: np.array([TAU_ALL, TAU_REGION, TAU_REGION, TAU_REGION]) * BASE_OUT[n]
          for n in DESIGNS}


def phi(name, pred):                      # pred: (..., 4) -> boolean survival of each outcome
    return pred >= THRESH[name][None, :]


def epistemic_contexts(name, rng, k=K_EPI, jitter=0.035):
    """The designer does not know the context exactly: every macro could plausibly have been
    placed a little differently. Blame (Chockler-Halpern) averages responsibility over this."""
    d = DESIGNS[name]
    out = [d.base_sites.copy()]
    while len(out) < k:
        xy = d.base_sites + rng.normal(0, jitter * d.die, d.base_sites.shape)
        xy = np.clip(xy, d.macro_wh / 2 + 5, d.die - d.macro_wh / 2 - 5)
        if all(not overlaps(xy, d.macro_wh, i, j)
               for i in range(d.NM) for j in range(i + 1, d.NM)):
            out.append(xy)
    return np.stack(out)


# ------------------------------------------------------------------ PN / PS / PNS
def necessity_sufficiency(name):
    d = DESIGNS[name]
    rng = np.random.default_rng(CFG.seed + 5)
    ctx = epistemic_contexts(name, rng)                       # (K, NM, 2)
    p_on = v_hat_batch(name, ctx)                             # outcome with everything in place
    on_holds = phi(name, p_on)                                # (K, 4)
    PN = np.zeros((d.NM, 4)); PS = np.zeros((d.NM, 4)); PNS = np.zeros((d.NM, 4))
    for m in range(d.NM):
        # --- PN: in contexts where the violation is present, is there an available move for m
        #     that kills it?  (the minimum over m's action set -- necessity you can act on) ---
        p_off = np.min(np.stack([v_hat_batch(name, np.concatenate(
            [ctx[:, :m], np.repeat(c[None, None], len(ctx), 0), ctx[:, m + 1:]], 1))
            for c in CONTRAST[name][m]]), 0)
        killed = on_holds & ~phi(name, p_off)
        PN[m] = killed.sum(0) / np.maximum(on_holds.sum(0), 1)
        PNS[m] = killed.mean(0)
        # --- PS: start from clean coalitions (m off, violation absent), put m back ---------
        masks = rng.random((96, d.NM)) < 0.5
        masks[:, m] = False
        clean_xy = np.stack([config_from_mask(name, mk) for mk in masks])
        p_clean = v_hat_batch(name, clean_xy)
        is_clean = ~phi(name, p_clean)
        back = clean_xy.copy(); back[:, m] = d.base_sites[m]
        p_back = v_hat_batch(name, back)
        appears = is_clean & phi(name, p_back)
        PS[m] = appears.sum(0) / np.maximum(is_clean.sum(0), 1)
    return PN, PS, PNS


# ------------------------------------------- HP responsibility (Chockler & Halpern 2004)
def responsibility(name, xy_actual=None, kmax=None, budget=4000):
    """resp(m) = 1/(1+|W|) for the smallest witness set W of *other* macros whose relocation
    makes m pivotal: phi still holds under do(W=w) with m where it is (AC2(a)), and fails under
    do(W=w, m=x') for some site x' in m's action set (AC2(b)). |W| = 0 is plain but-for
    causation, so resp = 1; a macro that needs two other things to move first gets 1/3."""
    d = DESIGNS[name]
    kmax = CFG.resp_max_witness if kmax is None else kmax
    rng = np.random.default_rng(CFG.seed + 7)
    xy0 = d.base_sites if xy_actual is None else xy_actual
    settings = CONTRAST[name]
    resp = np.zeros((d.NM, 4)); frac = np.zeros((d.NM, 4)); wit = [[None] * 4 for _ in range(d.NM)]
    for m in range(d.NM):
        others = [j for j in range(d.NM) if j != m]
        done = np.zeros(4, bool)
        for k in range(0, kmax + 1):
            combos = list(itertools.combinations(others, k)) or [()]
            cfgs, meta = [], []
            for W in combos:
                choices = list(itertools.product(*[range(len(settings[j])) for j in W])) or [()]
                if len(choices) * len(combos) > budget:
                    choices = [choices[i] for i in rng.choice(len(choices),
                               min(len(choices), max(1, budget // max(len(combos), 1))), replace=False)]
                for ch in choices:
                    x = xy0.copy()
                    for j, c in zip(W, ch): x[j] = settings[j][c]
                    cfgs.append(x); meta.append((W, ch))
            cfgs = np.stack(cfgs)
            keep = phi(name, v_hat_batch(name, cfgs))                     # AC2(a): phi survives
            gone = np.zeros_like(keep)                                    # AC2(b): m is pivotal
            for c in settings[m]:                                         # under *some* move of m
                flip = cfgs.copy(); flip[:, m] = c
                gone |= ~phi(name, v_hat_batch(name, flip))
            piv = keep & gone
            for o in range(4):
                if not done[o] and piv[:, o].any():
                    i = int(np.argmax(piv[:, o]))
                    resp[m, o] = 1.0 / (1 + k); wit[m][o] = meta[i]; done[o] = True
                    # tie-break: HP responsibility is coarse (1, 1/2, 1/3, ...), so we also record
                    # how *robustly* m is pivotal -- the share of minimal witnesses that work.
                    frac[m, o] = float(piv[:, o].mean())
            if done.all(): break
    return resp, wit, frac


def blame(name, n_ctx=None):
    """Blame = E_context[responsibility], the epistemic average a court actually assigns."""
    d = DESIGNS[name]
    rng = np.random.default_rng(CFG.seed + 11)
    ctx = epistemic_contexts(name, rng, k=n_ctx or (8 if QUICK else 14))
    acc = np.zeros((d.NM, 4))
    for c in ctx:
        acc += responsibility(name, c, kmax=min(2, CFG.resp_max_witness), budget=1200)[0]
    return acc / len(ctx)


HP = {}
t0 = time.time()
for name in DESIGNS:
    PN, PS, PNS = necessity_sufficiency(name)
    R, W, RF = responsibility(name)
    B = blame(name)
    # ranking score: HP degree of responsibility, tie-broken by robustness of pivotality
    RS = R * (0.5 + 0.5 * RF)
    HP[name] = dict(PN=PN, PS=PS, PNS=PNS, resp=R, witness=W, blame=B, resp_frac=RF, resp_score=RS)
    top = np.argsort(-R[:, 0])[:3]
    print(f"{name:11s} ({time.time()-t0:5.0f}s)  most responsible for signoff failure: " +
          ", ".join(f"{DESIGNS[name].macro_names[m]} resp={R[m,0]:.2f} PN={PN[m,0]:.2f} "
                    f"PS={PS[m,0]:.2f} blame={B[m,0]:.2f}" for m in top))
