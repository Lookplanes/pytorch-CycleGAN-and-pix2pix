import os
import numpy as np
import random
import torch
from data.base_dataset import BaseDataset

class UnalignedNpyDataset(BaseDataset):
    """
    加载 .npy 格式的非配对数据。
    假设 .npy 文件已经是 (C, H, W) 格式。
    会自动将数据归一化到 [-1, 1] 以适配 CycleGAN。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 允许用户在命令行设置是否需要翻转等（虽然这里我们硬编码了逻辑，但保留接口是个好习惯）
        parser.add_argument('--npy_no_norm', action='store_true', help='If specified, do not normalize to [-1, 1].')
        # 默认不进行 resize_and_crop，因为我们在预处理阶段已经切好 Patch 了
        if is_train:
            parser.set_defaults(preprocess='none') 
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        dir_A_name = opt.dir_A if hasattr(opt, 'dir_A') and opt.dir_A else opt.phase + 'A'
        dir_B_name = opt.dir_B if hasattr(opt, 'dir_B') and opt.dir_B else opt.phase + 'B'
        self.dir_A = os.path.join(opt.dataroot, dir_A_name)
        self.dir_B = os.path.join(opt.dataroot, dir_B_name)

        # 加载所有 .npy 文件
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy')])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith('.npy')])

        if len(self.A_paths) == 0:
            raise RuntimeError(f"Found 0 .npy files in {self.dir_A}")
        if len(self.B_paths) == 0:
            raise RuntimeError(f"Found 0 .npy files in {self.dir_B}")

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # 简单的通道数检查 (读取第一个文件)
        try:
            sample_A = np.load(self.A_paths[0])
            sample_B = np.load(self.B_paths[0])
            print(f"[Init] Dataset A shape: {sample_A.shape}, Dataset B shape: {sample_B.shape}")
            
            # 警告：如果配置的 input_nc 不等于实际数据的通道数
            if sample_A.shape[0] != opt.input_nc:
                print(f"[WARNING] opt.input_nc={opt.input_nc} but data A has {sample_A.shape[0]} channels!")
            if sample_B.shape[0] != opt.output_nc:
                print(f"[WARNING] opt.output_nc={opt.output_nc} but data B has {sample_B.shape[0]} channels!")
                
        except Exception as e:
            print(f"[WARNING] Failed to inspect first sample: {e}")

    def _normalize(self, tensor):
        """
        Instance-wise Min-Max Normalization to [-1, 1]
        将每个样本独立归一化到 [-1, 1]
        """
        if self.opt.npy_no_norm:
            return tensor
            
        min_val = tensor.min()
        max_val = tensor.max()
        
        # 避免除以零
        if max_val - min_val > 1e-5:
            # 先归一化到 [0, 1]
            tensor = (tensor - min_val) / (max_val - min_val)
            # 再映射到 [-1, 1]
            tensor = tensor * 2.0 - 1.0
        else:
            # 如果是纯色图片（例如全黑），直接设为0
            tensor = torch.zeros_like(tensor)
            
        return tensor

    def __getitem__(self, index):
        # 1. 确定索引
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 2. 加载 .npy (假设已经是 C, H, W)
        A_np = np.load(A_path).astype(np.float32)
        B_np = np.load(B_path).astype(np.float32)

        # 3. 转为 Tensor
        A_tensor = torch.from_numpy(A_np)
        B_tensor = torch.from_numpy(B_np)

        # 4. 归一化 (Crucial!)
        A_tensor = self._normalize(A_tensor)
        B_tensor = self._normalize(B_tensor)

        # 5. 数据增强 (随机翻转)
        if self.opt.phase == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                A_tensor = torch.flip(A_tensor, [2]) # [C, H, W], flip W
            if random.random() > 0.5:
                B_tensor = torch.flip(B_tensor, [2])

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)