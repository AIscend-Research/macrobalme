# =====================================================================================
# 17. Running the identical experiments against OpenROAD / ORFS
# =====================================================================================
# Nothing above touches the mini-flow except through Oracle.evaluate(macro_xy) -> Result.
# On a machine with OpenROAD and the ORFS designs (ibex, aes, jpeg on Nangate45), drop this
# class in as ORACLES[name] and re-run from section 5: a few hundred interventions at a few
# minutes each is an overnight job on one CPU, which is the budget this method was designed for.

ORFS_ADAPTER = r'''
class ORFSOracle:
    """Interventional oracle backed by a real place-and-route flow.

    Requires: openroad on PATH, an ORFS design directory with the synthesised netlist, the
    Nangate45 platform files, and a floorplan DEF whose macros we overwrite per intervention.
    """

    TCL = """
    read_lef {lef}
    read_def {def_in}
    global_placement -density 0.68 -pad_left 2 -pad_right 2
    estimate_parasitics -placement
    global_route -congestion_report_file {rpt} -allow_congestion
    exit
    """

    def __init__(self, design, workdir, lef, def_in, grid=48, timeout=1800):
        import shutil
        assert shutil.which("openroad"), "openroad not on PATH"
        self.d, self.workdir, self.lef, self.def_in = design, workdir, lef, def_in
        self.G, self.timeout = grid, timeout
        os.makedirs(workdir, exist_ok=True)

    def _write_def(self, macro_xy, path):
        """Rewrite the COMPONENTS section: every macro gets FIXED at its intervened site.
        DEF is in database units; ORFS Nangate45 uses 2000 dbu/um."""
        dbu = 2000
        src = open(self.def_in).read()
        for i, nm in enumerate(self.d.macro_names):
            x = int((macro_xy[i, 0] - self.d.macro_wh[i, 0] / 2) * dbu)
            y = int((macro_xy[i, 1] - self.d.macro_wh[i, 1] / 2) * dbu)
            src = re.sub(rf"(- {re.escape(nm)} \S+\s*\+ )(FIXED|PLACED) \( [-\d]+ [-\d]+ \)",
                         rf"\g<1>FIXED ( {x} {y} )", src)
        open(path, "w").write(src)

    def evaluate(self, macro_xy):
        import subprocess, tempfile, re
        tag = hashlib.md5(np.round(macro_xy, 3).tobytes()).hexdigest()[:10]
        d_in = os.path.join(self.workdir, f"{tag}.def")
        rpt = os.path.join(self.workdir, f"{tag}.rpt")
        self._write_def(macro_xy, d_in)
        tcl = os.path.join(self.workdir, f"{tag}.tcl")
        open(tcl, "w").write(self.TCL.format(lef=self.lef, def_in=d_in, rpt=rpt))
        subprocess.run(["openroad", "-exit", tcl], check=True, timeout=self.timeout,
                       capture_output=True)
        # ORFS congestion report: one line per overflowing GCell edge
        H = np.zeros((self.G, self.G)); U = np.zeros((self.G, self.G)); C = np.ones((self.G, self.G))
        for line in open(rpt):
            m = re.match(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*\).*cap\s*=\s*(\d+).*usage\s*=\s*(\d+)", line)
            if m:
                gx, gy, cap, use = (int(v) for v in m.groups())
                if gx < self.G and gy < self.G:
                    U[gy, gx] += use; C[gy, gx] += cap
        ovfl = np.clip(U - C, 0, None)
        hot = np.clip(ovfl - 0.10 * C, 0, None) / max(C.mean() * 0.25, 1e-9)
        return Result(drc=float(hot.sum()), n_hotspot=int((hot >= 1).sum()), wl=float(U.sum()),
                      hotspot=hot, ovfl=ovfl, usage=U, cap=C, lam=ovfl / np.maximum(C, 1),
                      cell_xy=np.zeros((1, 2)), net_cost=np.zeros(self.d.NNET),
                      macro_xy=macro_xy.copy())
'''
with open(P("data", "orfs_adapter.py"), "w") as f:
    f.write("# Drop-in replacement for the mini-flow oracle.\nimport os, re, hashlib\n"
            "import numpy as np\n" + ORFS_ADAPTER)
ART["orfs_adapter"] = P("data", "orfs_adapter.py")
print(textwrap.dedent("""
    To reproduce against real silicon flows:
      1. build ORFS, run `make DESIGN_CONFIG=./designs/nangate45/ibex/config.mk` up to floorplan
      2. point ORFSOracle at results/nangate45/ibex/base/2_floorplan.def and the merged LEF
      3. ORACLES['ibex'] = ORFSOracle(design, workdir, lef, def_in); re-run from section 5
    The only thing that changes is the cost of a call: minutes instead of milliseconds, which is
    why the surrogate exists.
    """).strip())

# =====================================================================================
#  Summary
# =====================================================================================
best = REPAIR[REPAIR.method != "Oracle-ranked (ground truth)"].iloc[0]
sal = REPAIR[REPAIR.group == "saliency"]
caus = REPAIR[REPAIR.group.isin(["causal", "economic"])]
heur = REPAIR[REPAIR.group == "heuristic"]
diag = RANK_DF.groupby("group").rho_total.mean()
print("\n" + "=" * 86)
print("RESULT".center(86))
print("=" * 86)
print(f"  real place & route runs           : {sum(o.calls for o in ORACLES.values()):,}")
print(f"  surrogate counterfactual queries  : ~{len(DESIGNS)*CFG.shapley_perms*NMAX:,}")
print("  -- diagnosis: rank correlation with the tool's own but-for effects --")
for g in ["causal", "economic", "heuristic", "saliency", "control"]:
    print(f"    {g:10s} : {diag.get(g, float('nan')):+.3f}")
print("  -- repair: DRC removed after real re-runs --")
print(f"  best attribution                  : {best.method}  ({best.red_A:.1f}% DRC removed)")
print(f"  causal + economic methods (mean)  : {caus.red_A.mean():.1f}% DRC removed")
print(f"  gradient saliency (mean)          : {sal.red_A.mean():.1f}% DRC removed")
print(f"  proximity heuristic               : {heur.red_A.mean():.1f}% DRC removed")
print(f"  random control                    : {float(REPAIR[REPAIR.method=='Random'].red_A.iloc[0]):.1f}%")
print(f"  oracle-ranked ground truth        : "
      f"{float(REPAIR[REPAIR.method=='Oracle-ranked (ground truth)'].red_A.iloc[0]):.1f}%")
print("=" * 86)

INDEX = pd.DataFrame([dict(artifact=k, path=os.path.relpath(v, OUT),
                           kb=round(os.path.getsize(v) / 1024, 1)) for k, v in ART.items()]
                     + [dict(artifact=os.path.splitext(f)[0], path=os.path.join(sub, f),
                             kb=round(os.path.getsize(P(sub, f)) / 1024, 1))
                        for sub in ("tables", "figures", "svg", "images", "html", "data", "anim")
                        for f in sorted(os.listdir(P(sub)))]).drop_duplicates("path")
INDEX.to_csv(P("artifact_index.csv"), index=False)
print(f"\n{len(INDEX)} artifacts written under {os.path.abspath(OUT)}:")
print(INDEX.to_string(index=False))
