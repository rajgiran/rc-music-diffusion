#!/usr/bin/env python3
# link_manifest_view.py (safe, with audio_root + index + verify)
import argparse, json, os, hashlib
from pathlib import Path
from tqdm import tqdm
import shutil

MODES = {"symlink","hardlink","copy"}

def resolve_src(path_str: str, audio_root: Path|None) -> Path:
    p = Path(path_str)
    if p.is_absolute(): return p
    if audio_root is not None: return (audio_root / p).resolve()
    return p.resolve()

def dest_name(id_val, src: Path, prefix_id: bool, taken: set[str]) -> str:
    base = src.name
    name = f"{int(id_val):07d}__{base}" if (prefix_id and isinstance(id_val,int) and id_val>=0) else base
    if name in taken:
        stem, ext = Path(name).stem, Path(name).suffix
        k=1
        while True:
            alt=f"{stem}__dup{k}{ext}"
            if alt not in taken:
                name=alt; break
            k+=1
    taken.add(name)
    return name

def do_link(src: Path, dst: Path, mode: str, relative: bool):
    if mode=="symlink":
        target = src
        if relative:
            try: target = src.relative_to(dst.parent)
            except Exception: target = src
        dst.symlink_to(target); return
    if mode=="hardlink": os.link(src, dst); return
    if mode=="copy": shutil.copy2(src, dst); return
    raise ValueError(f"unknown mode {mode}")

def sha1(path: Path, nbytes=262144):
    # tiny hash for quick spot-checks
    h=hashlib.sha1()
    with open(path, "rb") as f:
        chunk=f.read(nbytes)
        h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--audio_root", type=Path, default=None, help="Resolve relative manifest paths against this root")
    ap.add_argument("--mode", choices=sorted(MODES), default="symlink")
    ap.add_argument("--relative_links", action="store_true")
    ap.add_argument("--no_prefix_id", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exts", type=str, default=None, help="comma list like .wav,.flac")
    ap.add_argument("--write_index", action="store_true")
    ap.add_argument("--verify", action="store_true", help="After linking, verify N random files")
    ap.add_argument("--verify_n", type=int, default=20)
    args=ap.parse_args()

    exts=set()
    if args.exts:
        for t in args.exts.split(","):
            t=t.strip().lower()
            if not t: continue
            exts.add(t if t.startswith(".") else f".{t}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    with args.manifest.open() as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            rec=json.loads(ln)
            src=resolve_src(rec["path"], args.audio_root)
            if exts and src.suffix.lower() not in exts: continue
            rows.append((rec.get("id",-1), src, rec["path"]))

    if args.limit>0: rows=rows[:args.limit]

    taken=set(); linked=0; missing=0; errs=0
    index_lines=[]
    for rid, src, orig in tqdm(rows, desc="Link view"):
        if not src.exists():
            missing+=1
            continue
        name=dest_name(rid, src, prefix_id=not args.no_prefix_id, taken=taken)
        dst=args.out_dir/name
        try:
            do_link(src, dst, mode=args.mode, relative=args.relative_links)
            linked+=1
            if args.write_index:
                index_lines.append(json.dumps({"id": rid, "link_name": name, "src_abs": str(src), "src_orig": orig}))
        except FileExistsError:
            linked+=1
        except Exception as e:
            errs+=1
            print(f"[ERR] {src} -> {dst}: {e}")

    if args.write_index:
        (args.out_dir/"view_index.jsonl").write_text("\n".join(index_lines)+"\n")

    print(json.dumps({"linked":linked,"missing":missing,"errors":errs,"total":len(rows)}, indent=2))

    if args.verify and linked>0:
        import random
        picks=random.sample([ln for ln in (args.out_dir.iterdir()) if ln.is_file()], k=min(args.verify_n, linked))
        bad=0
        for p in picks:
            rp=p
            if p.is_symlink(): rp=p.resolve()
            if not rp.exists(): bad+=1; print("[BROKEN]", p)
        print(f"[verify] sampled {len(picks)} files; broken={bad}")

if __name__=="__main__":
    main()