import os
import torch
import functools
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    #print(image_name)
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)

# AIGC数据集，用于训练AIGC模型，数据只有图像，csv文件只有对应图像名称、MOS、对图像的一段文本描述
class AIGCDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        # # 读取xlsx文件
        # self.data = pd.read_excel(csv_file, header=None)
        # # Print and remove the first row
        # print(self.data.iloc[0])
        # self.data = self.data.iloc[1:]

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            if self.blind:
                mos = 0.0
            else:
                mos = float(self.data.iloc[index, 2])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, 2])
            # mos = proc_label(mos)
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 1]

        model_name = image_name.split('_')
        model_name = model_name[0]        
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)


class AIGCDataset_3k(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())
        # print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            if self.blind:
                mos = 0.0
            else:
                mos = float(self.data.iloc[index, 5])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, 5])
            # mos = proc_label(mos)
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 1]

        model_name = image_name.split('_')
        model_name = model_name[0]        
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)



class AIGCIQA2023Dataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 blind = False,
                 get_loader=get_default_img_loader):

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        # self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 1]
        image_path = os.path.join(self.img_dir, self.data.iloc[index, 0],image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                        n_channels,
                                                                                                        kernel_h,
                                                                                                        kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            mos = float(self.data.iloc[index, 3])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
            sel = sel.long()
            mos = float(self.data.iloc[index, 3])
        patches = patches[sel, ...]

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)
        
        prompt = self.data.iloc[index, 5]

        model_name = image_name.split('_')
        model_name = model_name[0]        
        prompt_name = model_name + ' ' + prompt

        sample = {'I': patches, 'mos': mos,'prompt':prompt, 'prompt_name':prompt_name, 'image_name':image_name}
        return sample

    def __len__(self):
        return len(self.data.index)