#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path
from tqdm import tqdm
import math, csv
import matplotlib.pyplot as plt

def _get_dataset(rec):
    ds = rec.get("dataset")
    if ds: return ds
    p = rec.get("query_path") or rec.get("path","")
    if "/maestro/" in p: return "maestro"
    if "/urmp/" in p: return "urmp"
    return "unknown"

def _softmax(x, tau=0.08):
    x = np.asarray(x, np.float32) / max(tau,1e-6)
    x -= x.max()
    ex = np.exp(x, dtype=np.float32)
    return ex / (ex.sum() + 1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbors_jsonl", required=True, type=Path)
    ap.add_argument("--union_manifest", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--tau", type=float, default=0.08)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load union (for dataset tags)
    union_rows = [json.loads(l) for l in open(args.union_manifest) if l.strip()]
    # load neighbors
    NBR = [json.loads(l) for l in open(args.neighbors_jsonl) if l.strip()]

    assert len(NBR) == len(union_rows), "neighbors and union sizes must match"

    # collect per-row metrics
    rows = []
    used_aux = 0
    for i,(u,n) in enumerate(zip(union_rows, NBR)):
        used = n.get("used_index","primary")
        if used == "aux": used_aux += 1
        sims = [r["sim"] for r in n.get("neighbors",[])]
        mean_sim = float(np.mean(sims)) if sims else 0.0
        max_sim  = float(np.max(sims)) if sims else 0.0
        w = _softmax(sims, tau=args.tau) if sims else [1.0]
        # entropy of weights as “concentration” indicator
        ent = -float(np.sum([wi*math.log(wi+1e-12) for wi in w]))
        rows.append({
            "id": int(u.get("id", i)),
            "dataset": _get_dataset(u),
            "used_index": used,
            "mean_sim": mean_sim,
            "max_sim": max_sim,
            "weight_entropy": ent,
            "query_path": u.get("path")
        })

    # write CSV
    csv_path = args.out_dir / "retrieval_perquery.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    # summary (overall + per dataset)
    def summarize(sub):
        ms = np.array([r["mean_sim"] for r in sub], np.float32)
        xs = np.array([r["max_sim"]  for r in sub], np.float32)
        ents = np.array([r["weight_entropy"] for r in sub], np.float32)
        return dict(
            n=len(sub),
            mean_sim=float(ms.mean()), p10=float(np.percentile(ms,10)), p50=float(np.percentile(ms,50)),
            p90=float(np.percentile(ms,90)),
            max_mean=float(xs.mean()), max_p10=float(np.percentile(xs,10)),
            max_p50=float(np.percentile(xs,50)), max_p90=float(np.percentile(xs,90)),
            weight_entropy_mean=float(ents.mean())
        )

    overall = summarize(rows)
    per_ds = {}
    for ds in sorted(set(r["dataset"] for r in rows)):
        sub = [r for r in rows if r["dataset"]==ds]
        per_ds[ds] = summarize(sub)

    meta = dict(
        total=len(rows),
        used_aux=int(used_aux),
        frac_aux=float(used_aux/len(rows)),
        tau=float(args.tau),
        overall=overall,
        per_dataset=per_ds,
    )
    (args.out_dir / "retrieval_stats.json").write_text(json.dumps(meta, indent=2))

    # quick plots
    def hist(values, title, fname):
        v = np.asarray(values, np.float32)
        plt.figure(figsize=(6,4), dpi=110)
        plt.hist(v, bins=50)
        plt.title(title); plt.xlabel("similarity"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(args.out_dir / fname); plt.close()

    hist([r["mean_sim"] for r in rows], "Mean top-K similarity", "mean_sim_hist.png")
    hist([r["max_sim"]  for r in rows], "Max similarity", "max_sim_hist.png")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {(args.out_dir / 'retrieval_stats.json')}")
    print("Done.")

if __name__ == "__main__":
    main()