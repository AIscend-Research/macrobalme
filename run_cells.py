import sys, glob, os, io, traceback
files = sorted(glob.glob("cells/*.py"))
upto = sys.argv[1] if len(sys.argv)>1 else "99"
ns = {"__name__":"__main__"}
for f in files:
    if os.path.basename(f)[:2] > upto: break
    print(f"\n===== {f} =====", flush=True)
    src = open(f).read()
    exec(compile(src, f, "exec"), ns)
