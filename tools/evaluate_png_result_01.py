import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def calc_mutual_information(img1, img2, bins=20):
    """
    计算两张图像的互信息 (Mutual Information, MI)。
    MI 越高，说明两张图共享的信息越多，配准潜力越好。
    """
    # 计算2D联合直方图
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    
    # 转换为概率分布
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, None] * py[None, :] # P(x)P(y)
    
    nzs = pxy > 0
    
    # MI公式: sum(P(x,y) * log(P(x,y) / (P(x)P(y))))
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

def evaluate_pair(fake_path, real_path):
    """
    评测一对图像
    fake_path: CycleGAN 生成的伪 MSI (Prediction)
    real_path: 真实的 PCA-MSI (Ground Truth)
    """
    # 1. 读取图像 (以灰度图模式读取，因为PCA主成分通常是单通道)
    # 如果你是3通道PCA可视化，可以去掉 0 参数，改为 cv2.IMREAD_COLOR
    img_fake = cv2.imread(fake_path, 0) 
    img_real = cv2.imread(real_path, 0)

    if img_fake is None or img_real is None:
        print("错误：无法读取图像，请检查路径。")
        return

    # 2. 确保尺寸一致
    if img_fake.shape != img_real.shape:
        img_fake = cv2.resize(img_fake, (img_real.shape[1], img_real.shape[0]))

    print(f"--- 正在评测: {fake_path} vs {real_path} ---")

    # 3. 计算指标
    
    # --- A. 结构相似性 (SSIM) [关键指标] ---
    # 范围 [-1, 1]，越接近 1 说明结构、纹理、边缘保留得越好。
    score_ssim, _ = ssim(img_fake, img_real, full=True)
    
    # --- B. 峰值信噪比 (PSNR) ---
    # 单位 dB，越高越好。反映像素级的重建误差。
    # 如果并未完美配准，这个分数会很低，仅供参考。
    score_psnr = psnr(img_real, img_fake)
    
    # --- C. 皮尔逊相关系数 (PCC) ---
    # 范围 [-1, 1]，反映两者强度的线性相关性。
    score_pcc, _ = pearsonr(img_fake.ravel(), img_real.ravel())
    
    # --- D. 互信息 (MI) [配准关键指标] ---
    # 值越大越好。
    # 证明生成图和真实图在统计学上共享信息，利于配准。
    score_mi = calc_mutual_information(img_fake, img_real)

    # 4. 打印结果
    print(f"-- SSIM: {score_ssim:.4f} ( >0.5 较好, >0.8 优秀)")
    print(f"-- MI: {score_mi:.4f} ")
    print(f"-- PCC: {score_pcc:.4f}")
    print(f"-- PSNR: {score_psnr:.2f}")


if __name__ == "__main__":
    fake_img_path = "/home/xujr/pytorch-CycleGAN-and-pix2pix/results/if_msi_cyclegan_ddp_v1/test_latest/images/UPEC_13_full_rec_B.png" 
    real_img_path = "/home/xujr/pytorch-CycleGAN-and-pix2pix/results/if_msi_cyclegan_ddp_v1/test_latest/images/UPEC_13_full_real_B.png"
    
    evaluate_pair(fake_img_path, real_img_path)