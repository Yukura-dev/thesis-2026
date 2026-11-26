"""Re-probe the saved Experiment B models on the thesis probe series 6 (fit) and 7 (score). No retraining."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_endpoint import BASE, CACHE, CONDITIONS, analyse, classify, collect, targets  # noqa: E402
from training import Autoencoder, _parameter_digest, resolve_device  # noqa: E402

device = resolve_device("cuda")
ledger = OUT / "runs_series67.jsonl"
done = set()
if ledger.exists():
    done = {(r["condition"], r["seed"]) for r in map(json.loads, ledger.read_text().splitlines())}
original = {(r["condition"], r["seed"]): r for r in map(json.loads, (OUT / "runs.jsonl").read_text().splitlines())}
for name, (overrides, init) in CONDITIONS.items():
    for seed in range(5):
        if (name, seed) in done:
            continue
        cfg = replace(BASE, seed=seed, **overrides)
        model = Autoencoder(cfg.latent_dim, cfg.background, initialization=init).to(device)
        model.load_state_dict(torch.load(CACHE / f"{name}_s{seed}.pt", map_location=device))
        model.eval()
        sha = _parameter_digest(model)
        assert sha == original[(name, seed)]["model_sha256"], "saved weights differ from the B ledger"
        z, c, lab = collect(model, cfg, 6, cfg.probe_train, device)
        zt, ct, labt = collect(model, cfg, 7, cfg.probe_test, device)
        probes = analyse(z, targets(c, lab), zt, targets(ct, labt))
        row = {"condition": name, "seed": seed, "model_sha256": sha, "probe_splits": [6, 7],
               "probes": probes, "class": classify(probes["mlp"])}
        with open(ledger, "a") as f:
            f.write(json.dumps(row) + "\n")
        m = probes["mlp"]
        print(name, seed, row["class"], original[(name, seed)]["class"],
              f"v={m['v_direct']:.3f} sym={m['sym']:.3f} big_v={probes['big']['v_direct']:.3f}", flush=True)
