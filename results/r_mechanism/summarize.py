"""Tables (tables.md) and figures (figures/*.png) for Experiment R from the jsonl outputs."""
import collections
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
ARMS = ("l1", "wl1", "gauss", "l2", "wl1_ord", "l1_affine", "l1_bn", "l1_switch300")
SHOW = (0, 1, 2, 5, 10, 25, 50, 100, 300, 1000, 3000)
lines = []


def P(*a):
    lines.append(" ".join(str(x) for x in a))


def read(name):
    f = OUT / name
    return [json.loads(l) for l in open(f)] if f.exists() else []


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def f3(x):
    return "—" if x is None or not np.isfinite(x) else (f"{x:.3f}" if abs(x) >= 1e-2 or x == 0 else f"{x:.1e}")


def table(header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "---|" * len(header))
    for r in rows:
        P("| " + " | ".join(str(c) for c in r) + " |")
    P("")


def frames(im):
    p, c = im["center_int_prev"] >= .65, im["center_int_cur"] >= .65
    return {(False, False): "bg", (True, False): "prev", (False, True): "cur", (True, True): "both"}[(p, c)]


# ---------------- training ----------------
train = read("train.jsonl")
P("# Experiment R tables\n")
P(f"Training rows: {len(train)}; base-arm ledger matches: {sum(r['ledger_match'] is True for r in train)}/"
  f"{sum(r['ledger_match'] is not None for r in train)}\n")

# ---------------- D: trajectories ----------------
traj = read("traj.jsonl")
T = collections.defaultdict(dict)
for r in traj:
    T[(r["arm"], r["radius"], r["seed"])][r["step"]] = r
P("## Endpoint (step 3000), all seeds\n")
rows = []
for arm in ARMS:
    for R in (5, 8):
        runs = [T[(arm, R, s)] for s in range(5) if (arm, R, s) in T and 3000 in T[(arm, R, s)]]
        if not runs:
            continue
        e = [x[3000] for x in runs]
        cls = collections.Counter(frames(x["image"]) for x in e)
        rows.append([arm, R, dict(cls), f3(med([x["image"]["iou"] for x in e])),
                     f3(med([x["probe_z"]["cur_mlp"] for x in e])), f3(med([x["probe_z"]["vel_mlp"] for x in e])),
                     f3(med([x["probe_big"]["z_vel"] for x in e])), f3(med([x["swap_ratio"] for x in e])),
                     " ".join(f"{x['probe_z']['vel_mlp']:.2f}" for x in e)])
table(["arm", "r", "frames", "IoU", "cur R² (MLP)", "vel R² (MLP)", "vel R² (big)", "swap ratio", "vel MLP per seed"], rows)


def t_c(run):
    """First checkpoint at which the run is collapsed (PLAN.md rule), or None."""
    from mech import collapsed  # noqa
    for s in sorted(run):
        if collapsed(run[s]["image"]):
            return s
    return None


def stall_start(run):
    """First checkpoint from which the run stays collapsed through 3000, or None."""
    from mech import collapsed  # noqa
    steps = sorted(run)
    for i, s in enumerate(steps):
        if all(collapsed(run[u]["image"]) for u in steps[i:]):
            return s
    return None


P("## D: trajectory medians over seeds (vel R² MLP / cur R² MLP / out_sd ÷ target_sd)\n")
for R in (5, 8):
    rows = []
    for arm in ARMS:
        runs = [T[(arm, R, s)] for s in range(5) if (arm, R, s) in T]
        if not runs:
            continue
        cells = []
        for st in SHOW:
            v = med([x[st]["probe_z"]["vel_mlp"] for x in runs])
            c = med([x[st]["probe_z"]["cur_mlp"] for x in runs])
            o = med([x[st]["image"]["out_sd"] / x[st]["image"]["target_sd"] for x in runs])
            cells.append(f"{v:.2f}/{c:.2f}/{o:.2f}")
        rows.append([arm] + cells)
    P(f"### r={R}\n")
    table(["arm"] + [str(s) for s in SHOW], rows)

P("## D: collapse step, share of the velocity-R² rise before it, and within-stall representation drift\n")
rows = []
for arm in ARMS:
    for R in (5, 8):
        for s in range(5):
            run = T.get((arm, R, s))
            if not run:
                continue
            tc, ss = t_c(run), stall_start(run)
            v0, v3 = run[0]["probe_z"]["vel_mlp"], run[3000]["probe_z"]["vel_mlp"]
            share = drift = enc_after = "—"
            if ss is not None:
                vt = run[ss]["probe_z"]["vel_mlp"]
                share = f3((vt - v0) / (v3 - v0)) if abs(v3 - v0) > .02 else "rise<0.02"
                stall = [u for u in sorted(run) if u >= ss]
                # ridge probe from the stall start applied at 3000, vs the 3000 probe on itself
                drift = f"{run[3000]['transfer_vel_r2'][str(ss)]:.2f} vs {run[3000]['transfer_vel_r2']['3000']:.2f}"
                enc_after = f3(run[3000]["dist_from_init_enc"] - run[ss]["dist_from_init_enc"])
            rows.append([arm, R, s, tc, ss, f3(v0), f3(run[ss]["probe_z"]["vel_mlp"]) if ss is not None else "—", f3(v3),
                         share, drift, enc_after])
table(["arm", "r", "seed", "t_c", "stall from", "vel@0", "vel@stall start", "vel@3000", "share of rise before stall",
       "ridge(stall start)→3000 vs ridge(3000)", "enc distance-from-init gained after stall start"], rows)

# swap numerator / denominator
P("## B/G5: swap numerator, denominator (absolute), asym-only velocity R² (MLP); medians\n")
for R in (5, 8):
    rows = []
    for arm in ARMS:
        runs = [T[(arm, R, s)] for s in range(5) if (arm, R, s) in T]
        if not runs:
            continue
        cells = [f"{med([x[st]['swap_num'] for x in runs]):.2g}/{med([x[st]['swap_den'] for x in runs]):.2g}/"
                 f"{med([x[st]['probe_asym']['vel_mlp'] for x in runs]):.2f}" for st in (0, 10, 100, 1000, 3000)]
        cells.append(f3(med([x[3000]["probe_big"]["asym_vel"] for x in runs])))
        cells.append(f3(med([x[3000]["probe_sym"]["vel_mlp"] for x in runs])))
        rows.append([arm] + cells)
    P(f"### r={R} (num/den/asym-vel)\n")
    table(["arm", "0", "10", "100", "1000", "3000", "asym vel big @3000", "sym vel MLP @3000"], rows)

# ---------------- training-time order signal ----------------
P("## B/G5: order-sensitive loss (loss with swapped target − loss), median over seeds, mean over step windows\n")
rows = []
from mech import run_dir  # noqa: E402
for arm in ARMS:
    for R in (5, 8):
        vals = []
        for s in range(5):
            f = run_dir(arm, R, s) / "steplog.npy"
            if f.exists():
                L = np.load(f)
                vals.append([float((L[a:b, 2] - L[a:b, 1]).mean()) for a, b in ((0, 10), (10, 100), (100, 1000), (1000, 3000))]
                            + [float((L[a:b, 1]).mean()) for a, b in ((0, 10), (2000, 3000))])
        if vals:
            v = np.median(np.array(vals), 0)
            rows.append([arm, R] + [f"{x:.2e}" for x in v])
table(["arm", "r", "gap 1-10", "gap 11-100", "gap 101-1000", "gap 1001-3000", "loss 1-10", "loss 2001-3000"], rows)

# ---------------- A: layers ----------------
layers = read("layers.jsonl")
order = ["input", "enc1", "enc2", "enc3", "enc4", "z", "from_latent", "dec1", "dec2", "dec3", "output"]
if layers:
    L = collections.defaultdict(list)
    for r in layers:
        L[(r["arm"], r["radius"], r["step"], r["layer"])].append(r)
    for R in (5, 8):
        for step in (0, 3000):
            P(f"## A: layerwise probes, r={R}, step {step} (median over seeds; ridge-full vel / ridge-full cur / MLP-PCA64 vel)\n")
            rows = []
            for arm in ("l1", "wl1", "gauss", "l2"):
                cells = []
                for layer in order:
                    e = L.get((arm, R, step, layer), [])
                    cells.append(f"{med([x['ridge_full']['vel'] for x in e]):.2f}/{med([x['ridge_full']['cur'] for x in e]):.2f}/"
                                 f"{med([x['mlp_pca64']['vel'] for x in e]):.2f}" if e else "—")
                rows.append([arm] + cells)
            table(["arm"] + order, rows)

# ---------------- G2 ----------------
g2 = read("g2.jsonl")
G = collections.defaultdict(dict)
for r in g2:
    G[(r["arm"], r["radius"], r["seed"])][r["step"]] = r
if g2:
    P("## G2: latent mean norm / input-dependent rms / input-dependent energy share at from_latent / decoder Jacobian (median)\n")
    for R in (5, 8):
        rows = []
        for arm in ARMS:
            runs = [G[(arm, R, s)] for s in range(5) if (arm, R, s) in G]
            if not runs:
                continue
            rows.append([arm] + [f"{med([x[st]['z_mean_norm'] for x in runs]):.2g}/{med([x[st]['z_var_rms'] for x in runs]):.2g}/"
                                 f"{med([x[st]['from_latent_var_frac'] for x in runs]):.1e}/{med([x[st]['jac_fro'] for x in runs]):.1e}"
                                 for st in (0, 10, 100, 1000, 3000)]
                        + [f3(med([x[3000]["part_ratio"] for x in runs])), f"{med([x[3000]['cond'] for x in runs]):.1e}",
                           f3(med([x[3000]["jac_vel_dir_gain"] / x[3000]["jac_mean_dir_gain"] for x in runs]))])
        P(f"### r={R}\n")
        table(["arm", "0", "10", "100", "1000", "3000", "PR@3000", "cond@3000", "J gain vel-dir ÷ mean-dir"], rows)

# ---------------- E ----------------
iv = read("interventions.jsonl")
if iv:
    P("## E: interventions at step 3000 (median over seeds)\n")
    rows = []
    for arm in ARMS:
        for R in (5, 8):
            e = [x for x in iv if x["arm"] == arm and x["radius"] == R]
            if not e:
                continue
            nat = med([x["natural_out_change_rms"] for x in e])
            rows.append([arm, R, f3(med([x["order_index"] for x in e])),
                         f3(med([x["swap"]["order_index_after"] for x in e])),
                         f3(med([x["swap"]["out_change_rms"] / x["natural_out_change_rms"] for x in e])),
                         f3(med([x["rotate"]["out_change_rms"] / x["natural_out_change_rms"] for x in e])),
                         f3(med([x["rotate"]["metrics_on_target"]["iou"] for x in e])),
                         f3(med([x["rotate"]["metrics_on_target"]["disp_err"] or np.nan for x in e])),
                         f3(med([x["random_dir"]["out_change_rms"] / x["natural_out_change_rms"] for x in e])),
                         f3(med([x["random_dir"]["maha"] / x["maha_real"] for x in e])),
                         " ".join(f"{k}:{med([x['asym_amplify'][k]['order_index'] for x in e]):.2f}" for k in ("1", "3", "10", "30", "100")),
                         f"{nat:.1e}"])
    table(["arm", "r", "order idx", "order idx after swap", "swap Δout ÷ natural", "rotate Δout ÷ natural",
           "rotate IoU vs rotated target", "rotate disp err (px)", "random-dir Δout ÷ natural", "random maha ÷ real",
           "asym×k order idx", "natural Δout rms"], rows)

# ---------------- G3 ----------------
g3 = [r for r in read("g3.jsonl") if "encoder" in r]
info = [r for r in read("g3.jsonl") if "map_info" in r]
if g3:
    for loss in ("l2", "l1"):
        for R in (5, 8):
            P(f"## G3: decoder retrained with {loss.upper()}, r={R}: median IoU (seeds with IoU>0.5) [Adam lr 3e-3]\n")
            rows = []
            for enc in ("untrained", "l1", "wl1", "gauss", "l2"):
                cells = []
                for pre in ("none", "center", "scale", "standardize", "whiten", "randlin"):
                    e = [x for x in g3 if x["encoder"] == enc and x["pre"] == pre and x["dec_loss"] == loss and x["radius"] == R
                         and x["opt"] == "adam" and x["lr"] == 3e-3 and not x["absorbed_from"]]
                    ious = [x["metrics"]["iou"] for x in e]
                    cells.append(f"{med(ious):.3f} ({sum(i > .5 for i in ious)}/{len(ious)})")
                rows.append([enc] + cells)
            table(["encoder", "none", "center", "scale", "standardize", "whiten", "randlin"], rows)
    P("## G3 controls: median IoU (n>0.5) ; arms on untrained/l1/l2 encoders\n")
    for loss in ("l2", "l1"):
        for R in (5, 8):
            rows = []
            variants = [("none", "adam", 3e-3, None), ("standardize", "adam", 3e-3, None), ("none", "adam", 3e-3, "standardize"),
                        ("standardize", "adam", 3e-3, "none"), ("none", "adam", 3e-4, None), ("none", "adam", 3e-2, None),
                        ("none", "sgd", 1e-2, None), ("none", "sgd", 1e-1, None), ("standardize", "sgd", 1e-2, None),
                        ("standardize", "sgd", 1e-1, None)]
            for enc in ("untrained", "l1", "l2"):
                cells = []
                for pre, opt, lr, ab in variants:
                    e = [x for x in g3 if x["encoder"] == enc and x["pre"] == pre and x["dec_loss"] == loss and x["radius"] == R
                         and x["opt"] == opt and x["lr"] == lr and x["absorbed_from"] == ab]
                    ious = [x["metrics"]["iou"] if x["finite"] else np.nan for x in e]
                    cells.append(f"{med(ious):.3f} ({sum(i > .5 for i in ious if np.isfinite(i))}/{len(ious)})")
                rows.append([enc] + cells)
            P(f"### {loss.upper()}, r={R}\n")
            table(["encoder", "none", "std", "none ← init of std", "std ← init of none", "none lr3e-4", "none lr3e-2",
                   "none SGD .01", "none SGD .1", "std SGD .01", "std SGD .1"], rows)
    P("## G3 decision rules, per seed (L2 decoder, Adam lr 3e-3)\n")

    def iou(enc, pre, R, s, loss="l2", opt="adam", lr=3e-3, ab=None):
        e = [x for x in g3 if x["encoder"] == enc and x["pre"] == pre and x["dec_loss"] == loss and x["radius"] == R
             and x["seed"] == s and x["opt"] == opt and x["lr"] == lr and x["absorbed_from"] == ab]
        return e[0]["metrics"]["iou"] if e else np.nan

    rows = []
    for R in (5, 8):
        for enc in ("l1", "wl1", "gauss", "l2"):
            for pre in ("none", "center", "standardize"):
                d = [iou(enc, pre, R, s) - iou("untrained", pre, R, s) for s in range(5)]
                rows.append([R, enc, pre, " ".join(f"{x:+.2f}" for x in d), sum(x >= .1 for x in d if np.isfinite(x))])
    table(["r", "encoder", "map", "IoU(encoder) − IoU(untrained), seeds 0-4", "seeds ≥ +0.1 (G3-info needs ≥4)"], rows)
    rows = []
    for R in (5, 8):
        g_l1 = [iou("l1", "standardize", R, s) - iou("l1", "none", R, s) for s in range(5)]
        g_l2 = [iou("l2", "standardize", R, s) - iou("l2", "none", R, s) for s in range(5)]
        rows.append([R, " ".join(f"{x:+.2f}" for x in g_l1), " ".join(f"{x:+.2f}" for x in g_l2), f3(med(g_l1) - med(g_l2))])
    table(["r", "std − none gain, l1 encoder", "std − none gain, l2 encoder", "median difference (G3-L1specific needs ≥ 0.1)"], rows)
    rows = []
    for R in (5, 8):
        for s in range(5):
            rows.append([R, s] + [f3(iou("l1", p, R, s, ab=ab)) for p, ab in (("none", None), ("none", "standardize"),
                                                                                ("standardize", "none"), ("standardize", None))])
    table(["r", "seed", "none", "none-param, std-init", "std-param, none-init", "std"], rows)
    P("Frames drawn by L1-loss decoders on l2/gauss encoders (standardize): " + "; ".join(
        f"r{R} {enc}: " + ",".join(frames(x["metrics"]) for x in g3 if x["encoder"] == enc and x["pre"] == "standardize"
                                   and x["dec_loss"] == "l1" and x["radius"] == R and x["opt"] == "adam" and x["lr"] == 3e-3
                                   and not x["absorbed_from"])
        for R in (5, 8) for enc in ("l2", "gauss")) + "\n")
    P("## G3 latent-map statistics (median over seeds)\n")
    rows = []
    for enc in ("untrained", "l1", "wl1", "gauss", "l2"):
        for R in (5, 8):
            e = [x for x in info if x["map_info"] == enc and x["radius"] == R]
            rows.append([enc, R, f3(med([x["mean_norm"] for x in e])), f3(med([x["sd_rms"] for x in e])),
                         f3(med([x["mean_norm"] / x["sd_rms"] for x in e])), f"{med([x['cov_cond'] for x in e]):.1e}",
                         sum(x["tiny_dims"] for x in e)])
    table(["encoder", "r", "|mean z|", "rms SD", "|mean| ÷ SD", "cov cond", "tiny dims (total)"], rows)
    # first-layer gradient share at step 1/100 (decoder from_latent vs rest), none vs standardize
    P("## G3 first-layer update share (from_latent update norm ÷ all-layer update norm), Adam lr 3e-3, median\n")
    rows = []
    for enc in ("untrained", "l1", "l2"):
        for loss in ("l2", "l1"):
            for R in (5, 8):
                cells = []
                for pre in ("none", "standardize"):
                    for st in (1, 100, 3000):
                        e = [x for x in g3 if x["encoder"] == enc and x["pre"] == pre and x["dec_loss"] == loss and x["radius"] == R
                             and x["opt"] == "adam" and x["lr"] == 3e-3 and not x["absorbed_from"]]
                        v = []
                        for x in e:
                            d = next(q for q in x["diag"] if q["step"] == st)
                            tot = np.sqrt(sum(u ** 2 for u in d["update"].values()))
                            v.append(np.sqrt(d["update"]["from_latent.weight"] ** 2 + d["update"]["from_latent.bias"] ** 2) / tot)
                        cells.append(f3(med(v)))
                rows.append([enc, loss, R] + cells)
    table(["encoder", "dec loss", "r", "none@1", "none@100", "none@3000", "std@1", "std@100", "std@3000"], rows)

# ---------------- F2: G3-time ----------------
gt = [r for r in read("g3_time.jsonl") if "encoder" in r]
if gt:
    P("## F2: L2 decoder on frozen plain-L1 encoders from different steps: IoU per seed (median)\n")
    steps_t = (0, 5, 10, 25, 50, 100, 300, 1000, 3000)
    for R in (5, 8):
        rows = []
        for pre in ("none", "center", "standardize"):
            for s in range(5):
                vals = []
                for st in steps_t:
                    e = [x for x in gt if x["encoder"] == f"l1@{st}" and x["pre"] == pre and x["radius"] == R and x["seed"] == s]
                    vals.append(f"{e[0]['metrics']['iou']:.2f}" if e else "—")
                rows.append([pre, s] + vals)
            meds = []
            for st in steps_t:
                e = [x["metrics"]["iou"] for x in gt if x["encoder"] == f"l1@{st}" and x["pre"] == pre and x["radius"] == R]
                meds.append(f"**{med(e):.2f}**" if e else "—")
            rows.append([pre, "median"] + meds)
        P(f"### r={R}\n")
        table(["map", "seed"] + [str(s) for s in steps_t], rows)

# ---------------- F3: union ----------------
un = read("union.jsonl")
if un:
    U = collections.defaultdict(list)
    for r in un:
        U[(r["arm"], r["radius"], r["step"])].append((r["own"], r["other"]))
    P("## F3: brightness above background on the own disk / the other frame's disk (units of fg−bg; median over seeds)\n")
    for R in (5, 8):
        rows = []
        for arm in ("l1", "wl1", "gauss", "l2", "wl1_ord", "l1_bn"):
            rows.append([arm] + [("%.2f/%.2f" % tuple(np.median(np.array(U[(arm, R, s)]), 0))) for s in
                                 (0, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 3000)])
        P(f"### r={R}\n")
        table(["arm", "0", "5", "10", "25", "50", "100", "200", "300", "500", "1000", "3000"], rows)

# ---------------- drift and cross-arm ----------------
drift = read("drift.jsonl")
if drift:
    Dd = collections.defaultdict(dict)
    for r in drift:
        Dd[(r["arm"], r["radius"], r["seed"])][(r["i"], r["j"])] = r
    P("## D: drift of z toward step 3000 (median over seeds): linear CKA / affine R² (z_3000 from z_i)\n")
    rows = []
    for arm in ("l1", "wl1", "gauss", "l2"):
        for R in (5, 8):
            runs = [Dd[(arm, R, s)] for s in range(5)]
            rows.append([arm, R] + [f"{med([d[(i, 3000)]['cka'] for d in runs]):.2f} / {med([d[(i, 3000)]['affine_r2'] for d in runs]):.2f}"
                                    for i in (0, 10, 25, 100, 300, 1000)])
    table(["arm", "r", "0", "10", "25", "100", "300", "1000"], rows)
    P("Per seed, l1 r=5, affine R² of z_3000 from z_i: " + "; ".join(
        f"s{s}: " + " ".join(f"{i}:{Dd[('l1', 5, s)][(i, 3000)]['affine_r2']:.2f}" for i in (10, 100, 300, 1000)) for s in range(5)) + "\n")
xa = read("cross_arm.jsonl")
if xa:
    X = collections.defaultdict(list)
    for r in xa:
        X[(r["radius"], r["rep"], r["a"], r["b"], r["step"])].append(r["cka"])
    for rep in ("enc4", "z"):
        P(f"## F1: linear CKA between losses at the same seed and step ({rep}; median over seeds)\n")
        rows = []
        for R in (5, 8):
            for a, b in (("l1", "init"), ("l2", "init"), ("l1", "l2"), ("l1", "gauss"), ("l1", "wl1"), ("gauss", "l2"), ("wl1", "l2")):
                rows.append([R, f"{a} vs {b}"] + [f"{med(X[(R, rep, a, b, s)]):.2f}" for s in (1, 5, 10, 25, 100, 300, 3000)])
        table(["r", "pair", "1", "5", "10", "25", "100", "300", "3000"], rows)

(OUT / "tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------- figures ----------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"l1": "#d62728", "wl1": "#9467bd", "gauss": "#2ca02c", "l2": "#1f77b4", "wl1_ord": "#ff7f0e",
       "l1_affine": "#8c564b", "l1_bn": "#e377c2", "l1_switch300": "#7f7f7f"}
if traj:
    fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for j, R in enumerate((5, 8)):
        for arm in ("l1", "wl1", "gauss", "l2", "wl1_ord"):
            runs = [T[(arm, R, s)] for s in range(5) if (arm, R, s) in T]
            if not runs:
                continue
            steps = sorted(runs[0])
            xs = [max(s, .5) for s in steps]
            for i, (key, fn) in enumerate((("vel R² (MLP)", lambda x: x["probe_z"]["vel_mlp"]),
                                            ("output SD ÷ target SD", lambda x: x["image"]["out_sd"] / x["image"]["target_sd"]),
                                            ("swap distance (numerator)", lambda x: x["swap_num"]))):
                ys = [med([fn(r[s]) for r in runs]) for s in steps]
                ax[i, j].plot(xs, ys, color=COL[arm], label=arm, marker=".")
                ax[i, j].set_ylabel(key)
                if i == 2:
                    ax[i, j].set_yscale("log")
        ax[0, j].set_title(f"r={R} (median of 5 seeds)")
        ax[2, j].set_xscale("log")
        ax[2, j].set_xlabel("step (0 drawn at 0.5)")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "trajectory.png", dpi=150)
if g2:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for arm in ("l1", "wl1", "gauss", "l2"):
        runs = [G[(arm, 5, s)] for s in range(5) if (arm, 5, s) in G]
        steps = sorted(runs[0])
        xs = [max(s, .5) for s in steps]
        for i, (key, k) in enumerate((("|mean z| ÷ input-dependent rms", None), ("input-dependent share at from_latent", "from_latent_var_frac"),
                                       ("decoder Jacobian ‖dD/dz‖", "jac_fro"))):
            ys = [med([(r[s]["z_mean_norm"] / r[s]["z_var_rms"]) if k is None else r[s][k] for r in runs]) for s in steps]
            ax[i].plot(xs, ys, color=COL[arm], label=arm, marker=".")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(key, fontsize=9)
            ax[i].set_xlabel("step")
    ax[0].legend(fontsize=8)
    fig.suptitle("r=5, median of 5 seeds")
    fig.tight_layout()
    fig.savefig(FIG / "latent_stats_r5.png", dpi=150)
if layers:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, R in enumerate((5, 8)):
        for arm in ("l1", "wl1", "gauss", "l2"):
            ys = [med([x["ridge_full"]["vel"] for x in L.get((arm, R, 3000, layer), [])]) for layer in order]
            ax[j].plot(order, ys, color=COL[arm], label=arm, marker="o")
        ys = [med([x["ridge_full"]["vel"] for a in ("l1",) for x in L.get((a, R, 0, layer), [])]) for layer in order]
        ax[j].plot(order, ys, color="k", ls="--", label="untrained", marker="x")
        ax[j].set_title(f"velocity, linear probe, r={R}, step 3000")
        ax[j].tick_params(axis="x", rotation=60)
    ax[0].set_ylabel("R² (ridge, all features)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "layers_velocity.png", dpi=150)
print("figures:", sorted(p.name for p in FIG.iterdir()))
