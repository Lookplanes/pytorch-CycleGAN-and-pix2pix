import os
import time
import numpy as np
import glob
from tifffile import TiffFile
from pyimzml.ImzMLParser import ImzMLParser
from sklearn.decomposition import PCA
from tqdm import tqdm

# ==========================================
#              用户配置区域 (请在此处修改)
# ==========================================

# 1. 输入路径设置   
IF_DIR = r"/data2/xujr/msi_if_registration/registered" 
MSI_DIR = r"/data2/xujr/msi_if_registration/fixed"

# 2. 输出路径设置
OUT_DIR = r"/data2/xujr/msi_if_npy_full"

# 3. 处理参数设置
PATCH_SIZE = None
STRIDE = 64
PCA_COMPONENTS = 3

# ==========================================
#            以下是核心逻辑代码
# ==========================================

def process_imzml_to_latent_array(imzml_path, n_components=3):
    """
    使用PCA对整个.imzML文件进行降维，返回一个(C, H, W)的NumPy数组。
    """
    print(f"\n[MSI] 开始处理文件: {imzml_path}")
    t0 = time.time()
    try:
        p = ImzMLParser(imzml_path, include_spectra_metadata=None)
    except Exception as e:
        print(f"[ERROR] 无法解析文件 {imzml_path}: {e}")
        return None

    try:
        # 使用列表推导式读取所有光谱
        spectra_list = []
        for i in tqdm(range(len(p.coordinates)), desc="加载质谱数据"):
             m, i_int = p.getspectrum(i)
             spectra_list.append(i_int)
        data_matrix = np.array(spectra_list)
        del spectra_list
    except Exception as e:
        print(f"[ERROR] 读取光谱时出错: {e}")
        return None
    
    print(f"[MSI] 数据加载完成，矩阵形状: {data_matrix.shape}，用时 {time.time()-t0:.1f}s")
    
    t1 = time.time()
    data_matrix[np.isnan(data_matrix)] = 0
    
    print(f"[MSI] 正在计算PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data_matrix)
    print(f"[MSI] PCA计算完成，用时 {time.time()-t1:.1f}s")
    
    H, W = p.imzmldict['max count of pixels y'], p.imzmldict['max count of pixels x']
    
    image_hwc = np.zeros((H, W, n_components), dtype=np.float32)
    for i, (x, y, z) in enumerate(p.coordinates):
        if 1 <= y <= H and 1 <= x <= W:
            image_hwc[y-1, x-1, :] = transformed_data[i, :]
            
    image_chw = image_hwc.transpose(2, 0, 1)
    
    print(f"[MSI] Latent MSI数组生成完毕，形状 (C,H,W): {image_chw.shape}")
    return image_chw


def process_tif_to_channel_array(tif_path):
    """
    加载多通道TIFF文件，返回一个(C, H, W)的NumPy数组。
    【修复版】：基于形状判断，兼容性更强。
    """
    print(f"\n[IF] 开始处理文件: {tif_path}")
    t0 = time.time()
    try:
        with TiffFile(tif_path) as tf:
            page = tf.series[0].pages[0]
            arr = page.asarray()
    except Exception as e:
        print(f"[ERROR] 无法读取TIFF文件 {tif_path}: {e}")
        return None
    
    print(f"[IF] 读取完成，用时 {time.time()-t0:.1f}s，原始形状 {arr.shape}")
    
    # 1. 移除所有单维度 (Squeeze)，清理可能的 T=1, Z=1 或 Q=1
    arr = np.squeeze(arr)
    
    # 2. 根据维度判断布局
    if arr.ndim == 2:
        # 情况 A: 只有 (H, W)，这是单通道灰度图
        # 增加通道维度 -> (1, H, W)
        arr = arr[np.newaxis, ...]
        print("[IF] 检测到2D灰度图，自动调整为 (1, H, W)")
        
    elif arr.ndim == 3:
        # 情况 B: 3维数据。我们需要判断是 (C, H, W) 还是 (H, W, C)
        # 启发式判断：通道数通常较小 (<=10)，而图像尺寸通常较大
        d0, d1, d2 = arr.shape
        
        if d2 <= 10 and d0 > 10:
            # 它是 (H, W, C)，需要转置
            arr = arr.transpose(2, 0, 1)
            print("[IF] 检测到 (H, W, C) 布局，转置为 (C, H, W)")
        elif d0 <= 10 and d2 > 10:
            # 它是 (C, H, W)，保持不变
            print("[IF] 检测到 (C, H, W) 布局，保持不变")
        else:
            print(f"[WARNING] 无法确定 (C,H,W) 布局，当前形状 {arr.shape}，假设已正确。")
            
    else:
        print(f"[ERROR] 不支持的图像维度: {arr.ndim}")
        return None

    print(f"[IF] IF数组准备完毕，最终形状 (C,H,W): {arr.shape}")
    return arr.astype(np.float32)


def main():
    if not os.path.exists(IF_DIR):
        print(f"[FATAL ERROR] IF目录不存在: {IF_DIR}")
        return
    if not os.path.exists(MSI_DIR):
        print(f"[FATAL ERROR] MSI目录不存在: {MSI_DIR}")
        return

    out_a_dir = os.path.join(OUT_DIR, 'A') 
    out_b_dir = os.path.join(OUT_DIR, 'B') 
    os.makedirs(out_a_dir, exist_ok=True)
    os.makedirs(out_b_dir, exist_ok=True)

    print(f"配置确认:\n  IF目录: {IF_DIR}\n  MSI目录: {MSI_DIR}\n  输出目录: {OUT_DIR}\n  Patch大小: {PATCH_SIZE}\n  PCA分量: {PCA_COMPONENTS}")

    if_files = sorted(glob.glob(os.path.join(IF_DIR, "*.tif*")))
    if not if_files:
        print(f"[ERROR] 在目录 {IF_DIR} 中未找到 .tif 或 .tiff 文件")
        return

    total_patches = 0
    
    for if_path in if_files:
        stem = os.path.splitext(os.path.basename(if_path))[0]
        msi_path = os.path.join(MSI_DIR, f"{stem}.imzML")

        if not os.path.exists(msi_path):
            print(f"[WARNING] 未找到与 {os.path.basename(if_path)} 配对的MSI文件，跳过...")
            continue
        
        print(f"\n{'='*20} 正在处理文件对: {stem} {'='*20}")
        
        # 1. 处理MSI
        msi_chw = process_imzml_to_latent_array(msi_path, n_components=PCA_COMPONENTS)
        if msi_chw is None: continue

        # 2. 处理IF
        if_chw = process_tif_to_channel_array(if_path)
        if if_chw is None: continue

        # 3. 【新增】自动尺寸对齐 (Crop to min)
        # 解决 (674, 606) vs (673, 605) 的问题
        h_if, w_if = if_chw.shape[1], if_chw.shape[2]
        h_msi, w_msi = msi_chw.shape[1], msi_chw.shape[2]
        
        if (h_if != h_msi) or (w_if != w_msi):
            print(f"[WARNING] 尺寸微小不匹配: IF({h_if}, {w_if}) vs MSI({h_msi}, {w_msi})")
            min_h = min(h_if, h_msi)
            min_w = min(w_if, w_msi)
            print(f"          >>> 自动裁剪至 ({min_h}, {min_w})")
            
            if_chw = if_chw[:, :min_h, :min_w]
            msi_chw = msi_chw[:, :min_h, :min_w]
        
        # 4. 根据 PATCH_SIZE 决定处理方式
        if PATCH_SIZE is None:
            # --- 新增逻辑：直接保存整张图 ---
            print(f"[INFO] PATCH_SIZE is None，正在保存整张图像...")
            
            full_image_name = f"{stem}_full.npy"
            np.save(os.path.join(out_a_dir, full_image_name), if_chw)
            np.save(os.path.join(out_b_dir, full_image_name), msi_chw)
            
            patch_count_for_file = 1
            print(f"[INFO] 文件对 {stem} 已保存为单个完整文件。")
            total_patches += patch_count_for_file
            
        else:
            # --- 原有逻辑：同步滑窗切割 ---
            _, H, W = if_chw.shape
            patch_count_for_file = 0
            
            y_coords = range(0, H - PATCH_SIZE + 1, STRIDE)
            x_coords = range(0, W - PATCH_SIZE + 1, STRIDE)
            
            total_ops = len(y_coords) * len(x_coords)
            if total_ops == 0:
                print(f"[WARNING] 图像太小 ({H}x{W}) 无法切割出 {PATCH_SIZE}x{PATCH_SIZE} 的块，跳过。")
                continue

            with tqdm(total=total_ops, desc=f"切割 {stem}") as pbar:
                for y in y_coords:
                    for x in x_coords:
                        patch_idx = patch_count_for_file
                        
                        if_patch = if_chw[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        msi_patch = msi_chw[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        
                        patch_name = f"{stem}_patch_{patch_idx:05d}.npy"
                        
                        np.save(os.path.join(out_a_dir, patch_name), if_patch)
                        np.save(os.path.join(out_b_dir, patch_name), msi_patch)
                        
                        patch_count_for_file += 1
                        pbar.update(1)
            
            print(f"[INFO] 文件对 {stem} 生成了 {patch_count_for_file} 个配对图块。")
            total_patches += patch_count_for_file

    print(f"\n[SUCCESS] 全部处理完成！总共生成了 {total_patches} 个图块/文件。")
    print(f"结果保存在: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()