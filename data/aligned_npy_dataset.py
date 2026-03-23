import os
import numpy as np
import random
import torch
from data.base_dataset import BaseDataset

class AlignedNpyDataset(BaseDataset):
    """
    加载【配对】的 .npy 数据。
    适用于 pix2pix 模型训练。
    
    核心特性：
    1. 严格按照文件名对应加载 (A/file1.npy <-> B/file1.npy)。
    2. 实现同步数据增强 (如果 A 翻转，B 必须跟着翻转)。
    3. 自动归一化到 [-1, 1]。
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 允许用户在命令行设置是否跳过归一化
        parser.add_argument('--npy_no_norm', action='store_true', help='If specified, do not normalize to [-1, 1].')
        # 对于配准切片数据，通常不需要 resize_and_crop，因为预处理时已经切好了
        if is_train:
            parser.set_defaults(preprocess='none') 
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        dir_A_name = opt.dir_A if hasattr(opt, 'dir_A') and opt.dir_A else opt.phase + 'A'
        dir_B_name = opt.dir_B if hasattr(opt, 'dir_B') and opt.dir_B else opt.phase + 'B'
        self.dir_A = os.path.join(opt.dataroot, dir_A_name)
        self.dir_B = os.path.join(opt.dataroot, dir_B_name)

        # 1. 获取 A 的所有文件
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy')])
        
        # 2. 严格推导 B 的路径
        # 我们假设: 如果 A 是 /path/to/A/sample1.npy，那么 B 必须是 /path/to/B/sample1.npy
        self.B_paths = []
        for a_path in self.A_paths:
            filename = os.path.basename(a_path)
            b_path = os.path.join(self.dir_B, filename)
            
            # 完整性检查：如果A有但B没有，直接报错，防止训练数据错位
            if not os.path.exists(b_path):
                raise RuntimeError(f"[Error] 配对文件丢失! \nFound: {a_path}\nMissing: {b_path}")
            
            self.B_paths.append(b_path)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths) # 应该相等

        if self.A_size == 0:
            raise RuntimeError(f"在 {self.dir_A} 中未找到 .npy 文件")

        # 3. 检查通道数 (读取第一张作为样本)
        try:
            sample_A = np.load(self.A_paths[0])
            sample_B = np.load(self.B_paths[0])
            print(f"[Init] AlignedNpyDataset - {opt.phase} set - {self.A_size} pairs")
            print(f"       Sample Shape: A={sample_A.shape}, B={sample_B.shape}")
            
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
        """
        if self.opt.npy_no_norm:
            return tensor
            
        min_val = tensor.min()
        max_val = tensor.max()
        
        # 避免除以零 (处理纯黑背景块)
        if max_val - min_val > 1e-5:
            tensor = (tensor - min_val) / (max_val - min_val)
            tensor = tensor * 2.0 - 1.0
        else:
            tensor = torch.zeros_like(tensor)
        return tensor

    def __getitem__(self, index):
        # 1. 获取路径 (严格配对)
        A_path = self.A_paths[index]
        B_path = self.B_paths[index] 

        # 2. 加载 .npy (假设已经是 C, H, W)
        A_np = np.load(A_path).astype(np.float32)
        B_np = np.load(B_path).astype(np.float32)

        # 3. 转为 Tensor
        A_tensor = torch.from_numpy(A_np)
        B_tensor = torch.from_numpy(B_np)

        # 4. 归一化
        A_tensor = self._normalize(A_tensor)
        B_tensor = self._normalize(B_tensor)

        # 5. 【关键】同步数据增强
        if self.opt.phase == 'train' and not self.opt.no_flip:
            # 50% 概率水平翻转
            if random.random() > 0.5:
                # torch.flip 翻转最后一个维度 (Width)
                # 假设 tensor 是 (C, H, W)，所以是 [2]
                A_tensor = torch.flip(A_tensor, [2]) 
                B_tensor = torch.flip(B_tensor, [2])

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size