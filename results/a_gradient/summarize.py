"""Experiment A summary: per-checkpoint medians over seeds, plus ledger SHA check."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "b_endpoint"))
from b_endpoint import ledger_shas  # noqa: E402

shas = ledger_shas()
rows = [json.loads(l) for l in open(OUT / "runs.jsonl")]
for r in rows:
    key = (5, r["loss"], 0.0, r["scale"], r["seed"], "default")
    r["ledger_match"] = r["model_sha256"] in shas.get(key, set()) if key in shas else None
print("ledger match:", [(r["loss"], r["seed"], r["ledger_match"]) for r in rows])

for loss in ("l1", "gauss_l1"):
    rs = [r for r in rows if r["loss"] == loss]
    print(f"\n== {loss} (n={len(rs)}) ==")
    print("step   dB        dC        dE        dK      rho=-dB/dC  cos(u,gB) cos(u,gC)  dLB_real  dLC_real   LB      LC    core  |gB|own |gC|own cosBC  |gB|cf/own |gC|cf/own cosBC_cf")
    for step in ("0", "50", "300", "1000", "2980"):
        def med(f):
            return float(np.median([f(r["checkpoints"][step]) for r in rs]))
        w = lambda c: c["adam_window"]
        dC = lambda c: w(c)["dE"] + w(c)["dK"]
        own = lambda c: c["snapshot"]["own"]
        cf = lambda c: c["snapshot"]["counterfactual"]
        real = lambda c, g: w(c)["parts_after"][g] - w(c)["parts_before"][g]
        print(f"{step:>5} {med(lambda c: w(c)['dB']):9.2e} {med(dC):9.2e} {med(lambda c: w(c)['dE']):9.2e} {med(lambda c: w(c)['dK']):9.2e}"
              f" {med(lambda c: -w(c)['dB'] / dC(c)):10.3g} {med(lambda c: w(c)['cos_u_gB']):8.3f} {med(lambda c: w(c)['cos_u_gC']):8.3f}"
              f" {med(lambda c: real(c, 'B')):9.2e} {med(lambda c: real(c, 'E') + real(c, 'K')):9.2e}"
              f" {med(lambda c: own(c)['parts']['B']):.4f} {med(lambda c: own(c)['parts']['E'] + own(c)['parts']['K']):.4f}"
              f" {med(lambda c: c['image']['core_mean']):.3f}"
              f" {med(lambda c: own(c)['norm']['B']):.2e} {med(lambda c: own(c)['norm']['C']):.2e} {med(lambda c: own(c)['cos_BC']):6.3f}"
              f" {med(lambda c: cf(c)['norm']['B'] / own(c)['norm']['B']):9.3g} {med(lambda c: cf(c)['norm']['C'] / own(c)['norm']['C']):9.3g}"
              f" {med(lambda c: cf(c)['cos_BC']):6.3f}")
    print("trace (median core_mean / LB / LC):")
    steps = [t["step"] for t in rs[0]["trace"]]
    for i, s in enumerate(steps):
        if s in (0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 2975):
            core = np.median([r["trace"][i]["core_mean"] for r in rs])
            lb = np.median([r["trace"][i]["B"] for r in rs])
            lc = np.median([r["trace"][i]["E"] + r["trace"][i]["K"] for r in rs])
            print(f"  {s:5d} core={core:.3f} LB={lb:.5f} LC={lc:.5f}")
