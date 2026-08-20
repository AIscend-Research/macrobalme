# =====================================================================================
# 11. The baselines we are arguing against: gradient saliency on the surrogate
# =====================================================================================
# This is what "explainable ML for placement" usually means: train a predictor of congestion,
# then ask which pixels it looked at. The maps are pretty and the question is correlational --
# a macro in a busy region lights up whether or not it is doing anything.

def macro_masks(name, macro_xy):
    o = ORACLES[name]
    return np.stack([o.macro_cover_one(macro_xy, m) for m in range(DESIGNS[name].NM)])


def _pool(sal, masks, mode="sum"):
    s = (sal[None] * masks).sum((1, 2))
    return s / masks.sum((1, 2)) if mode == "mean" else s


if HAS_TORCH:
    def gradcam(name, macro_xy):
        x = torch.tensor(unet_inputs(name, macro_xy)[None], device=DEV, requires_grad=True)
        feats = {}
        h = UNET.e3.register_forward_hook(lambda mod, i, o_: feats.__setitem__("e3", o_))
        _, g = UNET(x); h.remove()
        e3 = feats["e3"]
        gr = torch.autograd.grad(g.sum(), e3, retain_graph=False)[0]
        cam = F.relu((gr.mean((-1, -2), keepdim=True) * e3).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0].detach().cpu().numpy()

    def input_grad(name, macro_xy, channel=0):
        x = torch.tensor(unet_inputs(name, macro_xy)[None], device=DEV, requires_grad=True)
        _, g = UNET(x)
        gr = torch.autograd.grad(g.sum(), x)[0][0, channel]
        return (gr * x[0, channel]).detach().cpu().numpy()

    def integrated_grad(name, macro_xy, steps=32, channel=0):
        x1 = torch.tensor(unet_inputs(name, macro_xy)[None], device=DEV)
        x0 = torch.tensor(unet_inputs(name, DESIGNS[name].null_sites)[None], device=DEV)
        acc = torch.zeros_like(x1)
        for a in np.linspace(1.0 / steps, 1.0, steps):
            xi = (x0 + a * (x1 - x0)).requires_grad_(True)
            _, g = UNET(xi)
            acc += torch.autograd.grad(g.sum(), xi)[0]
        ig = (acc / steps * (x1 - x0))[0, channel]
        return ig.detach().cpu().numpy()
else:                                                          # pragma: no cover
    def gradcam(name, macro_xy): return unet_predict(name, macro_xy)[0]
    def input_grad(name, macro_xy, channel=0):
        return _W[:-1, :].sum(1).reshape(CH_IN, *X_ALL.shape[-2:])[channel] * unet_inputs(name, macro_xy)[channel]
    def integrated_grad(name, macro_xy, steps=32, channel=0): return input_grad(name, macro_xy, channel)


def proximity_heuristic(name):
    """What a designer does today with the congestion map open: blame whatever macro is nearest
    the red. Purely correlational, and a surprisingly strong baseline."""
    d, o = DESIGNS[name], ORACLES[name]
    hs = BASE[name].hotspot
    yy, xx = np.mgrid[0:o.G, 0:o.G] + 0.5
    out = np.zeros(d.NM)
    for m in range(d.NM):
        cx, cy = d.base_sites[m] / o.gs
        s = max(2.0, d.macro_wh[m].mean() / o.gs)
        out[m] = float((hs * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (1.5 * s) ** 2))).sum())
    return out


SAL = {}
for name in DESIGNS:
    d = DESIGNS[name]
    mk = macro_masks(name, d.base_sites)
    cam = gradcam(name, d.base_sites)
    ig = integrated_grad(name, d.base_sites)
    xg = input_grad(name, d.base_sites)
    SAL[name] = dict(gradcam=_pool(cam, mk, "mean"), ig=_pool(ig, mk), inputgrad=_pool(xg, mk),
                     proximity=proximity_heuristic(name), cam_map=cam, ig_map=ig)
    print(f"{name:11s} Grad-CAM top: " +
          ", ".join(d.macro_names[m] for m in np.argsort(-SAL[name]['gradcam'])[:3]) +
          "   |  IG top: " + ", ".join(d.macro_names[m] for m in np.argsort(-SAL[name]['ig'])[:3]) +
          "   |  proximity top: " + ", ".join(d.macro_names[m] for m in np.argsort(-SAL[name]['proximity'])[:3]))
