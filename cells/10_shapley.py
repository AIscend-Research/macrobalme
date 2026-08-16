# =====================================================================================
# 10. Shapley values: the cooperative-game reading of the same question
# =====================================================================================
# v(S) = DRC when the macros in S sit at their as-given sites and the rest at the default.
# phi_m is m's Shapley value of v -- the unique attribution satisfying efficiency, symmetry,
# null-player and additivity, so the per-macro numbers add up to the total violation count.
# 2^NM is out of reach for the oracle; it is nothing for the surrogate (~45 us/query), which
# is precisely the division of labour this paper argues for.

def shapley(name, n_perm=None, outcome=0, antithetic=True):
    d = DESIGNS[name]
    n_perm = n_perm or CFG.shapley_perms
    rng = np.random.default_rng(CFG.seed + 23)
    phi_ = np.zeros(d.NM); cnt = 0
    B = 64
    while cnt < n_perm:
        perms = [rng.permutation(d.NM) for _ in range(B)]
        if antithetic: perms += [p[::-1] for p in perms]
        cfgs, tag = [], []
        for pi, p in enumerate(perms):
            on = np.zeros(d.NM, bool)
            cfgs.append(config_from_mask(name, on)); tag.append((pi, -1))
            for m in p:
                on[m] = True
                cfgs.append(config_from_mask(name, on)); tag.append((pi, int(m)))
        vals = v_hat_batch(name, np.stack(cfgs))[:, outcome]
        i = 0
        for pi, p in enumerate(perms):
            prev = vals[i]; i += 1
            for m in p:
                phi_[m] += vals[i] - prev; prev = vals[i]; i += 1
        cnt += len(perms)
    phi_ /= cnt
    return phi_, cnt


SHAP = {}
for name in DESIGNS:
    t0 = time.time()
    ph, npm = shapley(name)
    SHAP[name] = ph
    d = DESIGNS[name]
    tot = v_hat(name, d.base_sites)[0] - v_hat(name, d.null_sites)[0]
    print(f"{name:11s} {npm} permutations in {time.time()-t0:.1f}s  |  efficiency check: "
          f"sum(phi) {ph.sum():+.1f} vs v(N)-v(0) {tot:+.1f}  |  top: " +
          ", ".join(f"{d.macro_names[m]} {ph[m]:+.1f}" for m in np.argsort(-ph)[:3]))

# region-specific Shapley (blame for one named hotspot, not for the whole block)
SHAP_R = {n: np.stack([shapley(n, n_perm=max(400, CFG.shapley_perms // 3), outcome=1 + r)[0]
                       for r in range(3)], 1) for n in DESIGNS}
print("region-wise Shapley computed:", {n: SHAP_R[n].shape for n in SHAP_R})
