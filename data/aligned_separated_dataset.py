import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class AlignedSeparatedDataset(BaseDataset):
    """
    A dataset class for paired image dataset stored in separate directories.

    It assumes that the directory contains two subdirectories (e.g., 'trainA' and 'trainB'
    or custom 'dir_A' and 'dir_B' via command line) that host paired images with the 
    EXACT SAME FILE NAMES.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)
        
        dir_A_name = opt.dir_A if hasattr(opt, 'dir_A') and opt.dir_A else opt.phase + 'A'
        dir_B_name = opt.dir_B if hasattr(opt, 'dir_B') and opt.dir_B else opt.phase + 'B'
        
        self.dir_A = os.path.join(opt.dataroot, dir_A_name)
        self.dir_B = os.path.join(opt.dataroot, dir_B_name)

        # 1. 获取 A 的所有文件
        A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        
        # 2. 严格推导 B 的路径
        self.A_paths = []
        self.B_paths = []
        
        for a_path in A_paths:
            filename = os.path.basename(a_path)
            b_path = os.path.join(self.dir_B, filename)
            
            # 完整性检查：如果A有但B没有，直接报错或跳过
            if not os.path.exists(b_path):
                print(f"[Warning] Paired file missing! Found: {a_path}, but not found: {b_path}. Skipping.")
                continue
            
            self.A_paths.append(a_path)
            self.B_paths.append(b_path)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert self.A_size > 0, f"Found 0 paired files in {self.dir_A} and {self.dir_B}."
        
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # apply the same transform to both A and B
        # Ensure that A and B have the same size!
        assert A_img.size == B_img.size, f"A and B must have the same size, but A={A_img.size} B={B_img.size} for {A_path}"
        
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size
