import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
# from data.base_dataset import __make_power_2

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))
        
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        # print(self.C_size)
        # print(self.D_size)
        
        self.BtoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if self.BtoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if self.BtoA else self.opt.output_nc      # get the number of channels of output image
        
        if self.BtoA: 
            self.transform_A = get_transform(self.opt, self.opt.preprocessB, 'B', grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, self.opt.preprocessA, 'A', grayscale=(output_nc == 1))
            self.transform_C = get_transform(self.opt, self.opt.preprocessA, 'C', grayscale=(output_nc == 1))
            self.transform_D = get_transform(self.opt, self.opt.preprocessA, 'D', grayscale=(output_nc == 1))
        else:
            self.transform_A = get_transform(self.opt, self.opt.preprocessA, 'A', grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, self.opt.preprocessB, 'B', grayscale=(output_nc == 1))
            self.transform_C = get_transform(self.opt, self.opt.preprocessB, 'C', grayscale=(output_nc == 1))
            self.transform_D = get_transform(self.opt, self.opt.preprocessB, 'D', grayscale=(output_nc == 1))

    

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        C_path = self.C_paths[index_B]
        D_path = self.D_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')
        
        sizeA = A_img.size
        
        if ("padding" in self.opt.preprocessA and not self.BtoA):
            # breakpoint()
            w, h = make_power_2(A_img.size[0], A_img.size[1], 256)
            result = Image.new(A_img.mode, (w, h), (0,0,0))
            result.paste(A_img, (0, 0))
            A_img = result
            
        elif ("padding" in self.opt.preprocessB and self.BtoA):
            w, h = make_power_2(B_img.size[0], B_img.size[1], 256)
            result = Image.new(B_img.mode, (w, h), (0,0,0))
            result.paste(B_img, (0, 0))
            B_img = result
            
        # apply image transformation
        transform_list = []
        transform_list += [transforms.ToTensor()]
        f = transforms.Compose(transform_list)
        
        A = self.transform_A(A_img)
        B = self.transform_D(B_img)


        crop_w, crop_h = A.shape[1], A.shape[2]
        crop_indices = transforms.RandomCrop.get_params(
                    C_img, output_size=(crop_w, crop_h))
        i, j, _, _ = crop_indices

        blur_tensor = transforms.functional.crop(C_img, i, j, crop_w, crop_h)
        sharp_tensor = transforms.functional.crop(D_img, i, j, crop_w, crop_h)
        blur_tensor = f(blur_tensor)
        sharp_tensor = f(sharp_tensor)

        return {'A': A, 'B': B, 'C': blur_tensor, 'D': sharp_tensor, 
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path,
                'sizeA': sizeA}

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # breakpoint()
        _x = x.clone()
        # p = self.opt.patch_size
        x = x.reshape(16*16, 16*16*3)
        L, D = x.shape  # batch, length, dim
        len_keep = int(L * mask_ratio)
        
        noise = torch.rand(L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([L], device=x.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)
        mask = mask.unsqueeze(-1).repeat(1, D)
        mask = mask.reshape(16, 16, 16,16, 3).permute(4,0,2,1,3).reshape(3, 256, 256)
        print(mask)
        print(_x)

        return mask * _x
    

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

def make_power_2(ow, oh, base, method=transforms.InterpolationMode.BICUBIC):
    # method = __transforms2pil_resize(method)
    if oh % base != 0:
        h = int((int(oh / base) + 1) * base)
    else: h = oh

    if ow % base != 0:
        w = int((int(ow / base) + 1) * base)
    else: w = ow
    return (w, h)
