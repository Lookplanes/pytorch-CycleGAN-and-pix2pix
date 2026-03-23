import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    from pyimzml.ImzMLParser import ImzMLParser
except ImportError:
    ImzMLParser = None
from skimage.transform import resize

# --- 1. 设置文件路径 ---
# !! 请将以下路径修改为你自己的文件路径 !!
PATH_IMZML = r"E:\DowLoad\downloadData\msi_if_registration\msi_if_registration\fixed\UPEC_13.imzML"
PATH_REGISTERED_TIF = r"E:\DowLoad\downloadData\msi_if_registration\msi_if_registration\registered\UPEC_13.tif" # 选择一个有代表性的tif
PATH_MASK_TIF = r"E:\DowLoad\downloadData\msi_if_registration\msi_if_registration\fixed\mask\UPEC_13.tif"


# --- 2. 加载并处理MSI数据 ---
print("正在加载并处理MSI数据...")
if ImzMLParser is None:
    print("错误：未安装 pyimzml 库。请运行: pip install pyimzml")
    sys.exit(1)
for path_label, path_value in [("IMZML", PATH_IMZML), ("REGISTERED_TIF", PATH_REGISTERED_TIF), ("MASK_TIF", PATH_MASK_TIF)]:
    if not os.path.isfile(path_value):
        print(f"错误：{path_label} 文件不存在: {path_value}")
        sys.exit(1)

# 使用 pyimzml 解析 .imzML 文件
try:
    p = ImzMLParser(PATH_IMZML)
except Exception as e:
    print(f"解析 imzML 文件时发生错误: {e}")
    sys.exit(1)

# 创建一个空的二维数组来存储总离子流 (TIC) 图像
# 获取图像维度
coords = p.coordinates
max_x = max(c[0] for c in coords)
max_y = max(c[1] for c in coords)
msi_tic_image = np.zeros((max_y + 1, max_x + 1))

# 计算每个像素的总离子流并填充到数组中
total = len(coords)
step = max(1, total // 50)
for i, (x, y, z) in enumerate(coords):
    try:
        intensities = p.getspectrum(i)[1]
    except Exception as e:
        # 若单个像素出错，跳过避免整体失败
        print(f"  警告：第{i}个像素读取失败，跳过。{e}")
        continue
    if y < msi_tic_image.shape[0] and x < msi_tic_image.shape[1]:
        msi_tic_image[y, x] = np.sum(intensities)
    if i % step == 0:
        print(f"  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='\r')
print()  # 换行

print("MSI数据加载完成。")


# --- 3. 加载TIF图像 ---
print("正在加载TIF图像...")
try:
    # 加载已配准的IF图像
    reg_tif_image = Image.open(PATH_REGISTERED_TIF)
    reg_tif_array = np.array(reg_tif_image)

    # 加载Mask图像
    mask_image = Image.open(PATH_MASK_TIF)
    mask_array = np.array(mask_image)
    
    print("TIF图像加载完成。")

except FileNotFoundError as e:
    print(f"错误：找不到文件。请检查你的路径设置。 {e}")
    exit()

# 如果IF图像是多通道的（如RGB），转换为灰度图以便于比较
if reg_tif_array.ndim == 3:
    # 使用亮度公式或者直接取平均值，这里取第一个通道作为示例
    reg_tif_array_gray = reg_tif_array[:, :, 0] 
else:
    reg_tif_array_gray = reg_tif_array

# 确保所有图像的尺寸一致，以防万一出现细微差别
# 以mask的尺寸为基准进行调整
target_shape = mask_array.shape
if msi_tic_image.shape != target_shape:
    print(f"警告: MSI图像尺寸 {msi_tic_image.shape} 与Mask尺寸 {target_shape} 不匹配。正在调整大小...")
    msi_tic_image = resize(msi_tic_image, target_shape, anti_aliasing=True)

if reg_tif_array_gray.shape != target_shape:
    print(f"警告: TIF图像尺寸 {reg_tif_array_gray.shape} 与Mask尺寸 {target_shape} 不匹配。正在调整大小...")
    reg_tif_array_gray = resize(reg_tif_array_gray, target_shape, anti_aliasing=True)


# --- 4. 可视化与叠加 ---
print("正在生成可视化图像...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 子图1: 显示MSI的TIC图像
axes[0].imshow(msi_tic_image, cmap='viridis')
axes[0].set_title('MSI Total Ion Current (TIC)')
axes[0].axis('off')

# 子图2: 显示配准后的IF图像
axes[1].imshow(reg_tif_array_gray, cmap='gray')
axes[1].set_title('Registered IF Image')
axes[1].axis('off')

# 子图3: 叠加图像以检查配准效果
axes[2].imshow(reg_tif_array_gray, cmap='gray') # IF图像作为背景
# 使用masked array只显示mask中非零的部分
masked_overlay = np.ma.masked_where(mask_array == 0, mask_array)
axes[2].imshow(masked_overlay, cmap='autumn', alpha=0.5) # 半透明叠加Mask
# 在最上层绘制MSI数据的轮廓线
axes[2].contour(msi_tic_image, colors='cyan', linewidths=0.8) # 叠加MSI轮廓
axes[2].set_title('Overlay Check (IF + Mask + MSI Contour)')
axes[2].axis('off')

plt.tight_layout()
plt.suptitle('Registration Alignment Check', fontsize=16)
plt.subplots_adjust(top=0.9) # 调整总标题位置
plt.show()
print("完成。")