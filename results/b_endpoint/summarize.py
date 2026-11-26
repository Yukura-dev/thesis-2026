"""Per-condition means of the Experiment B ledger, plus per-seed lines."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(OUT / "runs.jsonl")]
print(len(rows), "ledger_match:", dict(collections.Counter(r["ledger_match"] for r in rows)))
groups = collections.defaultdict(list)
for r in rows:
    groups[r["condition"]].append(r)
keys = ["prev", "cur", "v_direct", "v_delta", "mid", "sym", "slope_prev", "slope_cur", "error_corr"]
print("cond".ljust(18), " ".join(k[:8].rjust(8) for k in keys), "   big_v  big_sym  rdg_v  rdg_sym")
for c, rs in groups.items():
    def m(k, method="mlp"):
        return np.mean([r["probes"][method][k] for r in rs])
    print(c.ljust(18), " ".join(f"{m(k):8.3f}" for k in keys),
          f"{m('v_direct', 'big'):8.3f} {m('sym', 'big'):8.3f} {m('v_direct', 'ridge'):6.3f} {m('sym', 'ridge'):6.3f}",
          dict(collections.Counter(r["class"] for r in rs)), dict(collections.Counter(r["frames"] for r in rs)))
if "-v" in sys.argv:
    for r in rows:
        p = r["probes"]["mlp"]
        print(r["condition"].ljust(18), r["seed"], r["frames"].ljust(12), r["class"].ljust(12),
              " ".join(f"{p[k]:6.3f}" for k in keys), f"big_v={r['probes']['big']['v_direct']:.3f}",
              f"big_sym={r['probes']['big']['sym']:.3f}",
              f"cprev={r['image']['center_intensity_previous']:.2f} ccur={r['image']['center_intensity_current']:.2f}")
