import os, argparse, glob, json
import numpy as np
from typing import List, Tuple
from PIL import Image

def robust_scale01(arr: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    lo, hi = np.percentile(arr, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    y = (arr - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def to_rgb(chw: np.ndarray, mode: str, select: List[int], pmin: float, pmax: float) -> np.ndarray:
    C,H,W = chw.shape
    if mode == 'select':
        idx = select[:3] + [select[-1]]*(3-len(select)) if len(select)<3 else select[:3]
        rgb = np.stack([chw[i] if i < C else np.zeros((H,W),dtype=chw.dtype) for i in idx], axis=0)
    elif mode == 'pca3':
        X = chw.reshape(C, -1).T  # (Npix, C)
        # center
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        # SVD for top3 components
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:3].T  # (Npix,3)
        rgb = Z.T.reshape(3, H, W)
    elif mode == 'mean3':
        # tile or average channels into 3 planes
        if C >= 3:
            rgb = chw[:3]
        else:
            rgb = np.vstack([chw, np.repeat(chw[-1:],[3-C], axis=0)])
    else:
        raise ValueError(mode)
    # scale per-channel to [0,255]
    out = []
    for k in range(3):
        ch01 = robust_scale01(rgb[k], pmin, pmax)
        out.append((ch01*255.0).astype(np.uint8))
    return np.stack(out, axis=0)  # (3,H,W)

def load_npy_as_chw(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array: {path} shape={arr.shape}')
    # accept HWC or CHW
    if arr.shape[0] <= 8 and arr.shape[-1] > 8:
        arr = np.transpose(arr, (2,0,1))
    return arr.astype(np.float32)

def convert_dir(in_dir: str, out_dir: str, mode: str, select: List[int], pmin: float, pmax: float, pattern: str):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, pattern)))
    if not files:
        raise SystemExit(f'No files match {in_dir} / {pattern}')
    for i, f in enumerate(files):
        chw = load_npy_as_chw(f)
        rgb = to_rgb(chw, mode=mode, select=select, pmin=pmin, pmax=pmax)
        img = np.transpose(rgb, (1,2,0))  # HWC
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, base + '.png')
        Image.fromarray(img).save(out_path)
        if (i+1) % 100 == 0:
            print(f'[CONVERT] {i+1}/{len(files)} -> {out_dir}')
    # write a tiny meta for reproducibility
    meta = {
        'mode': mode,
        'select': select,
        'pmin': pmin,
        'pmax': pmax,
        'count': len(files),
    }
    with open(os.path.join(out_dir, 'npy_to_png_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[DONE] Wrote {len(files)} PNGs to {out_dir}')

def main():
    ap = argparse.ArgumentParser(description='Convert .npy patches (HWC/CHW) to 3-channel PNGs for CycleGAN.')
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--pattern', default='*.npy')
    ap.add_argument('--mode', choices=['select','pca3','mean3'], default='select')
    ap.add_argument('--select', type=str, default='0,1,2', help='Channel indices for mode=select, e.g. 0,1,2')
    ap.add_argument('--pmin', type=float, default=0.5)
    ap.add_argument('--pmax', type=float, default=99.5)
    args = ap.parse_args()

    select = [int(x) for x in args.select.split(',') if x.strip()]
    convert_dir(args.in_dir, args.out_dir, args.mode, select, args.pmin, args.pmax, args.pattern)

if __name__ == '__main__':
    main()
