import os, argparse, time
import numpy as np
from tifffile import TiffFile
from PIL import Image
import random

def robust_norm(x, pmin=0.5, pmax=99.5):
    lo, hi = np.percentile(x, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)

def pick_level(series, level, auto_max_mpx=None):
    # 返回 (page, axes, level_index)
    levels = getattr(series, "levels", None)
    if level >= 0 or not levels:
        page = series if level == 0 or not levels else levels[level]
        axes = getattr(page, "axes", getattr(series, "axes", ""))
        return page, axes, level
    # 自动选择：找 H*W <= auto_max_mpx 的最小层
    best_i, best_page, best_axes = 0, series, getattr(series, "axes", "")
    if not auto_max_mpx:
        return best_page, best_axes, 0
    try:
        for i, pg in enumerate([series] + list(levels)):
            axes = getattr(pg, "axes", best_axes)
            shp = pg.shape
            ax = {a: shp[idx] for idx, a in enumerate(axes)} if axes and shp else {}
            H, W = ax.get("Y", 0), ax.get("X", 0)
            mpx = (H or 0) * (W or 0) / 1e6
            if H and W:
                print(f"  Level {i}: shape={shp} axes={axes} ~{mpx:.1f}MP")
            if H and W and mpx <= auto_max_mpx:
                return pg, axes, i
            best_i, best_page, best_axes = i, pg, axes
    except Exception:
        pass
    return best_page, best_axes, best_i

def load_cyx(path, series=0, level=0, auto_max_mpx=None):
    print(f"[INFO] 打开文件: {path}")
    t0 = time.time()
    with TiffFile(path) as tf:
        s = tf.series[series]
        page, axes, sel_level = pick_level(s, level, auto_max_mpx)
        print(f"[INFO] 选择 series={series}, level={sel_level}, axes={axes}，开始读取...")
        try:
            arr = page.asarray()
        except AttributeError as e:
            print("检测到 NumPy/tifffile 不兼容：", e)
            print('修复示例：pip install -U "tifffile>=2024.8.30" imagecodecs 或 pip install "numpy<2" -U')
            raise
    print(f"[INFO] 读取完成，用时 {time.time()-t0:.1f}s，原始形状 {arr.shape}, axes={axes}")
    # 压 T/Z 轴为第0帧
    for ax in ("T", "Z"):
        if ax in axes:
            i = axes.index(ax)
            arr = np.take(arr, indices=0, axis=i)
            axes = axes.replace(ax, "")
    # 若无通道轴，补一个
    if "C" not in axes:
        arr = arr[np.newaxis, ...]
        axes = "C" + axes
    assert "Y" in axes and "X" in axes, f"未在轴标记中找到 Y/X: axes={axes}"
    order = [axes.index("C"), axes.index("Y"), axes.index("X")]
    cyx = np.moveaxis(arr, order, [0,1,2]).astype(np.float32)  # (C,H,W)
    while cyx.ndim > 3:
        cyx = np.take(cyx, indices=0, axis=3)
    C,H,W = cyx.shape
    print(f"[INFO] 重排到 CYX: (C,H,W)=({C},{H},{W})")
    return cyx

def to_rgb(cyx, select=None):
    C,H,W = cyx.shape
    idx = list(range(min(3, C))) if select is None else [i for i in select if 0 <= i < C]
    if not idx:
        idx = list(range(min(3, C)))
    chans = []
    for k in range(3):
        src = cyx[idx[k]] if k < len(idx) else cyx[idx[-1]]
        chans.append(robust_norm(src))
    rgb = np.stack(chans, 0).transpose(1,2,0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)

def save_patches_for_file(tiff_path, out_dir, patch, stride, channels, series, level,
                          auto_max_mpx=None, max_patches=None, sample_rate=1.0, min_mean=0.0):
    cyx = load_cyx(tiff_path, series=series, level=level, auto_max_mpx=auto_max_mpx)
    C,H,W = cyx.shape
    select = None if channels is None else [int(x) for x in channels.split(",")]
    os.makedirs(out_dir, exist_ok=True)
    pid = 0
    t0 = time.time()
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            # 随机采样，快速降块
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue
            tile = cyx[:, y:y+patch, x:x+patch]
            if tile.shape[1] != patch or tile.shape[2] != patch:
                continue
            rgb = to_rgb(tile, select)
            # 跳过平均亮度过低（背景）块
            if min_mean > 0.0 and (rgb.mean() / 255.0) < min_mean:
                continue
            Image.fromarray(rgb).save(os.path.join(out_dir, f"patch_{pid:05d}.png"))
            pid += 1
            if pid % 50 == 0:
                print(f"[INFO] 进度: {pid} patches, 位置(y,x)=({y},{x})")
            if max_patches and pid >= max_patches:
                print(f"[INFO] 达到 max_patches={max_patches}，提前结束。")
                break
        if max_patches and pid >= max_patches:
            break
    dt = time.time()-t0
    print(f"[DONE] {os.path.basename(tiff_path)} -> {out_dir} | 生成 {pid} 块，用时 {dt:.1f}s (C={C}, H={H}, W={W})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiff", type=str, help="输入 OME-TIFF 文件路径")
    ap.add_argument("--tiff_dir", type=str, default=None, help="批量处理该目录下的 .tif/.tiff 文件")
    ap.add_argument("--out_dir", required=True, help="输出目录（PNG 小块）或作为批量模式的根输出目录")
    ap.add_argument("--patch", type=int, default=256, help="块大小")
    ap.add_argument("--stride", type=int, default=256, help="滑窗步长")
    ap.add_argument("--channels", type=str, default=None, help="通道索引，如 0,1,2；留空取前三个")
    ap.add_argument("--series", type=int, default=0, help="多Series文件选择哪个（默认0）")
    ap.add_argument("--level", type=int, default=-1, help="金字塔层（0为最高分辨率，-1表示自动）")
    ap.add_argument("--auto_max_mpx", type=float, default=25.0, help="自动选择层的最大像素数（百万像素）阈值")
    ap.add_argument("--max_patches", type=int, default=100, help="最多生成多少块（0或负数表示不限制）")
    ap.add_argument("--sample_rate", type=float, default=1.0, help="0-1，随机保留比例，用于降采样块数")
    ap.add_argument("--min_mean", type=float, default=0.0, help="0-1，平均亮度阈值，低于则跳过（过滤背景）")
    args = ap.parse_args()

    if args.tiff_dir:
        import glob
        files = sorted(list(glob.glob(os.path.join(args.tiff_dir, "*.tif"))) +
                       list(glob.glob(os.path.join(args.tiff_dir, "*.tiff"))))
        if not files:
            raise SystemExit(f"目录内未找到 .tif/.tiff: {args.tiff_dir}")
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            out_sub = os.path.join(args.out_dir, stem)
            os.makedirs(out_sub, exist_ok=True)
            save_patches_for_file(fp, out_sub, args.patch, args.stride, args.channels,
                                  args.series, args.level, args.auto_max_mpx,
                                  None if args.max_patches and args.max_patches <= 0 else args.max_patches,
                                  args.sample_rate, args.min_mean)
        return

    if not args.tiff:
        raise SystemExit("请提供 --tiff 或 --tiff_dir")
    save_patches_for_file(args.tiff, args.out_dir, args.patch, args.stride, args.channels,
                          args.series, args.level, args.auto_max_mpx,
                          None if args.max_patches and args.max_patches <= 0 else args.max_patches,
                          args.sample_rate, args.min_mean)

if __name__ == "__main__":
    main()