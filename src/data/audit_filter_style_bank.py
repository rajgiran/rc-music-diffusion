# audit_filter_style_bank.py
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import ClapModel, ClapProcessor

def load_text_vecs(proc, model, device, texts):
    with torch.no_grad():
        t = proc(text=texts, return_tensors="pt", padding=True).to(device)
        z = model.get_text_features(**t)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_npy", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ckpt", default="laion/clap-htsat-unfused")
    ap.add_argument("--music_text", default="instrumental music, no vocals, clean recording, high fidelity")
    ap.add_argument("--speech_text", default="speech, vocals, singing, rapping, human voice, talking, narration, choir, crowd")
    ap.add_argument("--keep_margin", type=float, default=0.08, help="Keep rows with sim(style,music)-sim(style,speech) > keep_margin")
    ap.add_argument("--max_keep", type=int, default=0, help="Optional cap on number of rows to keep (0 = no cap)")
    args = ap.parse_args()

    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = ClapProcessor.from_pretrained(args.ckpt)
    model = ClapModel.from_pretrained(args.ckpt).to(device).eval()

    S = np.load(args.agg_npy, mmap_mode="r").astype(np.float32)
    print(f"[INFO] Loaded style bank: {S.shape}")

    with torch.no_grad():
        Z = torch.from_numpy(S).to(device)
        Z = torch.nn.functional.normalize(Z, dim=-1)

    z_music  = load_text_vecs(proc, model, device, [args.music_text])[0]    # [512]
    z_speech = load_text_vecs(proc, model, device, [args.speech_text])[0]   # [512]

    sim_music  = (Z * z_music).sum(dim=-1).detach().cpu().numpy()
    sim_speech = (Z * z_speech).sum(dim=-1).detach().cpu().numpy()
    delta = sim_music - sim_speech

    # Save histograms
    plt.figure(figsize=(6,4)); plt.hist(delta, bins=60); plt.title("delta = sim(music) - sim(speech)")
    plt.xlabel("delta"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(out / "delta_hist.png", dpi=160); plt.close()

    # Basic stats
    frac_speechy = float((delta < 0).mean())
    print(f"[STATS] mean delta={delta.mean():.4f} | frac(delta<0) (speech-leaning)={frac_speechy:.3f}")

    # Keep only rows confidently musical
    keep_mask = delta > args.keep_margin
    idx = np.nonzero(keep_mask)[0]
    if args.max_keep > 0 and len(idx) > args.max_keep:
        # keep the best ones by largest delta
        top = np.argsort(-delta[idx])[:args.max_keep]
        idx = idx[top]

    print(f"[FILTER] Kept {len(idx)}/{len(S)} rows ({100.0*len(idx)/len(S):.1f}%) with margin>{args.keep_margin}")
    np.save(out / "agg_emb.filtered.npy", S[idx])
    (out / "kept_indices.json").write_text(json.dumps({"kept_indices": idx.tolist()}, indent=2))

    # Save scatter for sanity
    plt.figure(figsize=(5,5))
    plt.scatter(sim_speech, sim_music, s=6, alpha=0.5)
    plt.axline((0,0),(1,1), linestyle="--", linewidth=1)
    plt.xlabel("sim(style, speech)"); plt.ylabel("sim(style, music)")
    plt.tight_layout(); plt.savefig(out / "speech_vs_music_scatter.png", dpi=160); plt.close()

if __name__ == "__main__":
    main()