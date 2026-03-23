import os
import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ==========================================
#              用户配置区域
# ==========================================

# 1. 结果路径设置
# 指向 test 生成结果的 images 文件夹
# RESULTS_DIR = r"/home/xujr/pytorch-CycleGAN-and-pix2pix/results/if_msi_pix2pix_v1_BtoA/test_latest/images"
RESULTS_DIR = r"/home/xujr/pytorch-CycleGAN-and-pix2pix/results/if_msi_cyclegan_ddp_v1/test_latest/images"

# 2. 匹配关键词设置
KEYWORD_REAL = "_real_"
KEYWORD_FAKE = "_fake_"


def calculate_ncc(img1, img2):
    """计算归一化互相关"""
    img1 = img1 - img1.mean()
    img2 = img2 - img2.mean()
    if np.std(img1) == 0 or np.std(img2) == 0:
        return 0.0
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    return numerator / denominator

def get_edges(img_gray):
    """提取Canny边缘"""
    img_u8 = (img_gray * 255).astype(np.uint8)
    v = np.median(img_u8)
    # 自动调整阈值
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img_u8, lower, upper)
    return edges

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"[ERROR] 目录不存在: {RESULTS_DIR}")
        return

    # 1. 搜索所有包含 "_real_" 的图片 (不区分 A 或 B)
    # 例如: UPEC_12_real_A.png, UPEC_12_real_B.png
    search_pattern = os.path.join(RESULTS_DIR, f"*{KEYWORD_REAL}*.png")
    real_files = sorted(glob.glob(search_pattern))
    
    if not real_files:
        print(f"[ERROR] 未找到任何包含 '{KEYWORD_REAL}' 的图片。")
        return

    print(f"扫描到 {len(real_files)} 个潜在的真值文件，开始寻找配对...")
    
    pairs_found = []
    
    # 2. 建立配对列表
    for real_path in real_files:
        basename = os.path.basename(real_path)
        if KEYWORD_FAKE in basename: continue
        
        fake_basename = basename.replace(KEYWORD_REAL, KEYWORD_FAKE)
        fake_path = os.path.join(RESULTS_DIR, fake_basename)
        
        if os.path.exists(fake_path):
            pairs_found.append((real_path, fake_path))
            
    if not pairs_found:
        print("[ERROR] 未找到任何配对的 (Real, Fake) 图像。请检查文件名格式。")
        return

    print(f"成功匹配 {len(pairs_found)} 对图像，开始计算指标...")

    ncc_scores = []
    edge_ssim_scores = []
    pixel_ssim_scores = [] 
    
    # 3. 遍历计算
    for real_path, fake_path in tqdm(pairs_found):
        # 读取图像为灰度
        img_real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        img_fake = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)
        
        if img_real is None or img_fake is None:
            continue
            
        # 尺寸对齐 (防止 CycleGAN/Pix2Pix Padding 导致的微小差异)
        if img_real.shape != img_fake.shape:
            h, w = img_real.shape
            img_fake = cv2.resize(img_fake, (w, h), interpolation=cv2.INTER_LINEAR)

        # 归一化到 0-1 float32
        img_real = img_real.astype(np.float32) / 255.0
        img_fake = img_fake.astype(np.float32) / 255.0
        
        # --- 指标计算 ---
        
        # A. NCC (配准度/空间一致性)
        ncc = calculate_ncc(img_real, img_fake)
        ncc_scores.append(ncc)
        
        # B. Edge SSIM (轮廓一致性)
        edges_real = get_edges(img_real)
        edges_fake = get_edges(img_fake)
        score_edge, _ = ssim(edges_real, edges_fake, full=True, data_range=255)
        edge_ssim_scores.append(score_edge)

        # C. Full SSIM (纹理+结构相似度)
        score_pixel, _ = ssim(img_real, img_fake, full=True, data_range=1.0)
        pixel_ssim_scores.append(score_pixel)

    avg_ncc = np.mean(ncc_scores)
    avg_edge_ssim = np.mean(edge_ssim_scores)
    avg_pixel_ssim = np.mean(pixel_ssim_scores)
    
    print("\n" + "="*50)
    print("   生成质量评估报告 (Any Real vs Fake)")
    print("="*50)
    print(f"评估样本对数: {len(ncc_scores)}")
    print("-" * 50)
    print(f"NCC (空间对齐度): {avg_ncc:.4f}")
    print("-" * 50)
    print(f"Edge SSIM (轮廓一致性): {avg_edge_ssim:.4f}")
    print("  -> 衡量结构边缘是否重合")
    print("-" * 50)
    print(f"Full SSIM (全图相似度): {avg_pixel_ssim:.4f}")
    print("  -> 衡量整体生成质量")
    print("="*50)
    
if __name__ == "__main__":
    main()