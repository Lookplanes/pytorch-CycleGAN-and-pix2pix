import os, json, math, argparse, time, random
import numpy as np
from typing import List, Tuple, Optional
ImzML_import_error = None
ImzMLParser = None  # will lazy load
_IMPORT_TRIED = []

def lazy_load_pyimzml(auto_install: bool = True):
    """Attempt import of pyimzML with case variants; capture diagnostics if fails."""
    global ImzMLParser, ImzML_import_error, _IMPORT_TRIED
    if ImzMLParser is not None:
        return ImzMLParser
    candidates = [
        ('pyimzML.ImzMLParser', 'ImzMLParser'),
        ('pyimzml.ImzMLParser', 'ImzMLParser'),
    ]
    for mod, attr in candidates:
        try:
            pkg = __import__(mod, fromlist=[attr])
            ImzMLParser = getattr(pkg, attr)
            _IMPORT_TRIED.append((mod, 'OK'))
            return ImzMLParser
        except Exception as e:
            _IMPORT_TRIED.append((mod, f'FAIL: {e}'))
            ImzML_import_error = e
    if auto_install:
        try:
            import subprocess, sys
            print('[INFO] 自动安装 pyimzML ...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pyimzML'])
            return lazy_load_pyimzml(auto_install=False)
        except Exception as e2:
            ImzML_import_error = (ImzML_import_error, e2)
    return None

"""
Convert paired imzML + ibd MSI data into (H,W,C) float32 numpy arrays and optional patch tiles.
Supported channel construction modes:
  1) targets: provide explicit m/z target list (--mz_targets) with tolerance (--mz_tol)
  2) auto_peaks: automatically select top-K frequent peaks (--auto_top_k, --bin_width)
  3) bin: fixed m/z range binned at width (--mz_min, --mz_max, --bin_width)
  4) pca: construct binned matrix then apply PCA (--pca_components)

Output:
  - single cube: out_dir/cube.npy (shape H,W,C) unless --patches_only
  - patches: out_dir/patches/patch_XXXXX.npy (H_patch,W_patch,C) if --patch>0
  - metadata: out_dir/meta.json (contains m/z list or PCA components description, scaling info)

Channel normalization pipeline:
  1) TIC normalization (optional --tic)
  2) log1p (optional --log1p)
  3) per-channel robust percentile scaling to [0,1] (pmin,pmax) OR min-max (--robust_pmin/pmax)
  4) optionally map to [-1,1] (--range_minus1_1) for GAN usage

Sampling & filtering for patches:
  - --sample_rate <1.0 random keep ratio
  - --min_mean discard patches whose average intensity < threshold after scaling
  - --max_patches limit number of generated patches

Reconstruction note:
  - targets and bin modes maintain explicit m/z channels (lossless aside from scaling)
  - auto_peaks loses all but selected peaks
  - PCA is lossy; meta.json stores mean and components for approximate inverse transform
"""

def parse_mz_targets(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]

def robust_scale_channel(arr: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    lo, hi = np.percentile(arr, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    y = (arr - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)

def bin_spectra(mzs: np.ndarray, ints: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # simple histogram weighted by max intensity per bin
    idxs = np.searchsorted(bin_edges, mzs, side='right') - 1
    valid = (idxs >= 0) & (idxs < len(bin_edges)-1)
    out = np.zeros(len(bin_edges)-1, dtype=np.float32)
    if np.any(valid):
        # choose max intensity in bin (alternative: sum)
        for i, val in zip(idxs[valid], ints[valid]):
            if val > out[i]:
                out[i] = val
    return out

def select_auto_peaks(all_mzs_list: List[np.ndarray], all_ints_list: List[np.ndarray], top_k: int, bin_width: float) -> List[float]:
    # accumulate intensities by rounding to nearest bin center
    freq = {}
    for mzs, ints in zip(all_mzs_list, all_ints_list):
        if mzs.size == 0: continue
        bins = np.round(mzs / bin_width) * bin_width
        for m, v in zip(bins, ints):
            freq[m] = freq.get(m, 0.0) + float(v)
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [m for m,_ in ranked[:top_k]]

def build_cube(parser, mode: str, args) -> Tuple[np.ndarray, List[float], dict]:
    coordinates_raw = parser.coordinates
    # some imzML may provide (x,y,z) or (x,y,other); we only use first two
    coordinates = [(c[0], c[1]) for c in coordinates_raw]
    xs = np.array([c[0] for c in coordinates], dtype=int)
    ys = np.array([c[1] for c in coordinates], dtype=int)
    H, W = ys.max(), xs.max()
    spectra_mzs = []
    spectra_ints = []
    for i in range(len(coordinates)):
        mzs, ints = parser.getspectrum(i)
        spectra_mzs.append(np.array(mzs, dtype=np.float32))
        spectra_ints.append(np.array(ints, dtype=np.float32))
    meta = {"mode": mode}

    if mode == 'targets':
        targets = parse_mz_targets(args.mz_targets)
        meta['targets'] = targets
        cube = np.zeros((H, W, len(targets)), dtype=np.float32)
        for (x,y), mzs, ints in zip(coordinates, spectra_mzs, spectra_ints):
            for k, t in enumerate(targets):
                if mzs.size == 0:
                    continue
                idx = np.argmin(np.abs(mzs - t))
                if abs(mzs[idx] - t) <= args.mz_tol:
                    cube[y-1, x-1, k] = ints[idx]
    elif mode in ('bin','pca','auto_peaks'):
        # prepare bin edges
        mz_min = args.mz_min
        mz_max = args.mz_max
        bin_width = args.bin_width
        n_bins = int(math.ceil((mz_max - mz_min) / bin_width))
        bin_edges = mz_min + np.arange(n_bins + 1) * bin_width
        if mode == 'auto_peaks':
            # pick top peaks first
            auto_targets = select_auto_peaks(spectra_mzs, spectra_ints, args.auto_top_k, bin_width)
            meta['auto_targets'] = auto_targets
            cube = np.zeros((H, W, len(auto_targets)), dtype=np.float32)
            for (x,y), mzs, ints in zip(coordinates, spectra_mzs, spectra_ints):
                if mzs.size == 0: continue
                for k, t in enumerate(auto_targets):
                    idx = np.argmin(np.abs(mzs - t))
                    if abs(mzs[idx] - t) <= bin_width * 0.5:  # tolerance heuristic
                        cube[y-1, x-1, k] = ints[idx]
        else:
            # full bin cube
            cube = np.zeros((H, W, n_bins), dtype=np.float32)
            for (x,y), mzs, ints in zip(coordinates, spectra_mzs, spectra_ints):
                if mzs.size == 0: continue
                binned = bin_spectra(mzs, ints, bin_edges)
                cube[y-1, x-1, :] = binned
            meta['bin_edges'] = bin_edges.tolist()
            if mode == 'pca':
                # reshape to pixels x bins then PCA
                from sklearn.decomposition import PCA
                pixels = cube.reshape(H*W, n_bins)
                # remove all-zero rows to stabilize PCA
                valid_mask = pixels.sum(axis=1) > 0
                pixels_valid = pixels[valid_mask]
                pca = PCA(n_components=args.pca_components)
                comp = pca.fit_transform(pixels_valid)
                # allocate compressed cube
                comp_cube = np.zeros((H*W, comp.shape[1]), dtype=np.float32)
                comp_cube[valid_mask] = comp.astype(np.float32)
                cube = comp_cube.reshape(H, W, comp.shape[1])
                meta['pca_components'] = pca.components_.astype(np.float32).tolist()
                meta['pca_mean'] = pca.mean_.astype(np.float32).tolist()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return cube, meta.get('targets') or meta.get('auto_targets') or meta.get('bin_edges'), meta

def apply_scaling(cube: np.ndarray, args) -> Tuple[np.ndarray, dict]:
    scale_meta = {}
    if args.tic:
        tic = cube.sum(axis=2, keepdims=True)
        tic[tic==0] = 1
        cube = cube / tic
        scale_meta['tic'] = True
    if args.log1p:
        cube = np.log1p(cube)
        scale_meta['log1p'] = True
    # per-channel robust scaling
    C = cube.shape[2]
    mins = []
    maxs = []
    for k in range(C):
        ch = cube[:,:,k]
        lo, hi = np.percentile(ch, [args.robust_pmin, args.robust_pmax])
        if hi <= lo:
            ch_scaled = np.zeros_like(ch, dtype=np.float32)
        else:
            ch_scaled = (ch - lo) / (hi - lo)
            ch_scaled = np.clip(ch_scaled, 0, 1)
        cube[:,:,k] = ch_scaled
        mins.append(float(lo)); maxs.append(float(hi))
    scale_meta['robust_pmin'] = args.robust_pmin
    scale_meta['robust_pmax'] = args.robust_pmax
    scale_meta['channel_lo'] = mins
    scale_meta['channel_hi'] = maxs
    if args.range_minus1_1:
        cube = cube * 2.0 - 1.0
        scale_meta['range'] = '[-1,1]'
    else:
        scale_meta['range'] = '[0,1]'
    return cube.astype(np.float32), scale_meta

def save_patches(cube: np.ndarray, out_dir: str, patch: int, stride: int, sample_rate: float, min_mean: float, max_patches: Optional[int]):
    os.makedirs(out_dir, exist_ok=True)
    H,W,C = cube.shape
    pid = 0
    t0 = time.time()
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            tile = cube[y:y+patch, x:x+patch, :]
            if tile.shape[0] != patch or tile.shape[1] != patch:
                continue
            if min_mean > 0.0 and (tile.mean()) < min_mean:
                continue
            np.save(os.path.join(out_dir, f"patch_{pid:05d}.npy"), tile)
            pid += 1
            if pid % 50 == 0:
                print(f"[PATCH] {pid} generated at (y,x)=({y},{x})")
            if max_patches and pid >= max_patches:
                print(f"[PATCH] Reached max_patches={max_patches}, stopping early.")
                return pid, time.time()-t0
    return pid, time.time()-t0

def main():
    ap = argparse.ArgumentParser(description='Convert imzML+ibd MSI data to cube and patches.')
    ap.add_argument('--imzml', required=True, help='Path to .imzML file (paired .ibd must exist).')
    ap.add_argument('--out_dir', required=True, help='Output directory.')
    ap.add_argument('--mode', choices=['targets','bin','pca','auto_peaks'], default='targets')
    # targets mode
    ap.add_argument('--mz_targets', type=str, default='', help='Comma list of m/z targets for targets mode.')
    ap.add_argument('--mz_tol', type=float, default=0.05, help='Tolerance for target peak matching.')
    # bin / pca / auto_peaks shared
    ap.add_argument('--mz_min', type=float, default=100.0)
    ap.add_argument('--mz_max', type=float, default=1000.0)
    ap.add_argument('--bin_width', type=float, default=1.0)
    # auto peaks
    ap.add_argument('--auto_top_k', type=int, default=64)
    # pca
    ap.add_argument('--pca_components', type=int, default=64)
    # scaling
    ap.add_argument('--tic', action='store_true')
    ap.add_argument('--log1p', action='store_true')
    ap.add_argument('--robust_pmin', type=float, default=0.5)
    ap.add_argument('--robust_pmax', type=float, default=99.5)
    ap.add_argument('--range_minus1_1', action='store_true')
    # patches
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--sample_rate', type=float, default=1.0)
    ap.add_argument('--min_mean', type=float, default=0.0)
    ap.add_argument('--max_patches', type=int, default=0, help='0=unlimited')
    ap.add_argument('--patches_only', action='store_true', help='Do not save full cube, only patches.')
    # stub fallback
    ap.add_argument('--stub_random', action='store_true', help='Skip parsing, create random cube for pipeline test.')
    ap.add_argument('--stub_hw', type=str, default='256,256', help='Stub H,W size.')
    ap.add_argument('--stub_channels', type=int, default=4, help='Stub channel count.')
    args = ap.parse_args()

    if not args.stub_random and lazy_load_pyimzml(auto_install=True) is None:
        import sys, pkgutil, site
        diag = {
            'tried': _IMPORT_TRIED,
            'sys_executable': sys.executable,
            'sys_path_head': sys.path[:8],
            'site_packages': [p for p in site.getsitepackages() if os.path.isdir(p)],
            'iter_imz_modules': [m.name for m in pkgutil.iter_modules() if 'imz' in m.name.lower()],
        }
        print('[DIAG] Import diagnostics:\n' + json.dumps(diag, indent=2))
        raise SystemExit(f'pyimzML 导入失败: {ImzML_import_error}\n可加 --stub_random 继续测试流程。')
    if not os.path.exists(args.imzml):
        if not args.stub_random:
            raise SystemExit(f'imzML not found: {args.imzml}')
    # case-insensitive ibd check
    base, ext = os.path.splitext(args.imzml)
    ibd_guess = base + '.ibd'
    if not os.path.exists(ibd_guess):
        # try upper/lower variants
        ibd_alt = base + '.IBD'
        if os.path.exists(ibd_alt):
            ibd_guess = ibd_alt
        else:
            print(f'[WARN] Paired ibd not found: {ibd_guess}. 如果文件扩展名不同请手动确认。')

    os.makedirs(args.out_dir, exist_ok=True)
    if args.stub_random:
        H,W = [int(x) for x in args.stub_hw.split(',')]
        C = args.stub_channels
        cube = np.random.rand(H,W,C).astype(np.float32)
        channels = list(range(C))
        meta = {'mode':'stub_random','channels':C}
        print(f'[STUB] Random cube shape={cube.shape}')
    else:
        print('[LOAD] Parsing imzML ...')
        parser = ImzMLParser(args.imzml)
        t0 = time.time()
        cube, channels, meta = build_cube(parser, args.mode, args)
        print(f'[CUBE] Raw cube shape: {cube.shape}, mode={args.mode}, channels={len(channels) if channels else cube.shape[2]}')
    cube, scale_meta = apply_scaling(cube, args)
    meta['scaling'] = scale_meta
    meta['final_shape'] = cube.shape
    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path,'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[META] Saved meta.json')

    if not args.patches_only:
        cube_path = os.path.join(args.out_dir, 'cube.npy')
        np.save(cube_path, cube.astype(np.float32))
        print(f'[SAVE] Full cube saved: {cube_path}')

    patches_dir = os.path.join(args.out_dir, 'patches')
    max_p = args.max_patches if args.max_patches > 0 else None
    n_p, dt = save_patches(cube, patches_dir, args.patch, args.stride, args.sample_rate, args.min_mean, max_p)
    print(f'[DONE] Generated {n_p} patches in {dt:.1f}s')
    print('[NEXT] Use cube.npy or patches/*.npy with custom dataset (aligned_npy / unaligned_npy).')

if __name__ == '__main__':
    main()
