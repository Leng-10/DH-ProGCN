from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from loaddata.Get_MRI_data import Get_MRI_data
import glob
import SimpleITK as sitk
import numpy as np
import os
from monai.transforms import apply_transform
from scipy import ndimage

# List = ['NC', 'SMC', 'EMCI', 'LMCI', 'AD']
# List = ['NC', 'SMC', 'EMCI', 'LMCI', 'AD', 'sMCI', 'pMCI']
# List = ['NC', 'SMC', 'EMCI', 'LMCI', 'AD', 'NCSMC']
# List = ['NC', 'SMC', 'EMCI', 'LMCI', 'AD', 'NCSMC', 'SMCAD', 'NCAD']
# List = ['NIFD-C', 'NIFD-N-193']
List = ['PPMI-HC', 'PPMI-PD-352', 'PPMI-PD']


class Train_Dataset(data.Dataset):
    def __init__(self, root, class1, class2, transforms):
        self.transforms = transforms
        self.root = root
        self.input_D = 91
        self.input_H = 112
        self.input_W = 91

        # LABELS = [List[class1], List[class2]]
        LABELS = [class1, class2]
        labels1 = {name: index for index in range(len(LABELS)) for name in
                   glob.glob(self.root + '/MRI/' + LABELS[index] + '/train' + '/*.nii')}
        # labels1 = {name: index for index in range(len(LABELS)) for name in
        #            glob.glob(self.root + '/PET/' + LABELS[index] + '/train' + '/*.nii')}

        self.labels1 = labels1
        self.names1 = list(sorted(labels1.keys()))
        # self.names2 = list(sorted(labels2.keys()))
        self.shape = (len(labels1), 1, 1, 1)  # 这句没看懂；

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):

        im, rotation, flip, enhance = np.unravel_index(idx, self.shape)
        label = self.labels1[self.names1[im]]
        img_MRI = sitk.ReadImage(self.names1[im])
        MRI_image = sitk.GetArrayFromImage(img_MRI)

        MRI_image = np.transpose(MRI_image, (2, 1, 0)).astype(np.float32)

        # img_PET = sitk.ReadImage(self.names2[im])
        # PET_image = sitk.GetArrayFromImage(img_PET)
        # PET_image = np.transpose(PET_image, (2, 1, 0)).astype(np.float32)

        # sample1 = {'MRI_image': MRI_image, 'PET_image': PET_image}
        sample1 = {'MRI_image': MRI_image}
        # sample1 = sample1.cuda()
        sample1 = apply_transform(self.transforms, sample1)

        # MRI_image, PET_image = sample1['MRI_image'], sample1['PET_image']
        MRI_image = sample1['MRI_image']
        # MRI_image = self.__resize_data__(MRI_image)
        # PET_image = self.__resize_data__(PET_image)

        # sample = {'label': label, 'MRI_image': MRI_image, 'PET_image': PET_image}
        sample = {'label': label, 'MRI_image': MRI_image}

        return sample

    def __resize_data__(self, data):
        """
            Resize the loaddata to the input size
        """
        [n, depth, height, width] = data.shape
        scale = [n, self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data




class Test_Dataset(data.Dataset):
    def __init__(self, image_dir, class1, class2, transform):
        super().__init__()
        self.transforms = transform
        self.root = image_dir
        self.input_D = 56
        self.input_H = 448
        self.input_W = 448

        # LABELS = [List[class1], List[class2]]
        LABELS = [class1, class2]
        labels1 = {name: index for index in range(len(LABELS)) for name in
                   glob.glob(self.root + '/MRI/' + LABELS[index] + '/test' + '/*.nii')}
        # labels1 = {name: index for index in range(len(LABELS)) for name in
        #            glob.glob(self.root + '/PET/' + LABELS[index] + '/test' + '/*.nii')}

        self.labels = labels1
        self.names1 = list(sorted(labels1.keys()))
        # self.names2 = list(sorted(labels2.keys()))
        self.shape = (len(labels1), 1, 1, 1)  # 这句没看懂；
        # self.augment_size = np.prod(self.shape) / len(labels1)  # 这句没看懂；


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im, rotation, flip, enhance = np.unravel_index(idx, self.shape)

        label = self.labels[self.names1[im]]
        img_MRI = sitk.ReadImage(self.names1[im])
        MRI_image = sitk.GetArrayFromImage(img_MRI)
        MRI_image = np.transpose(MRI_image, (2, 1, 0)).astype(np.float32)

        # img_PET = sitk.ReadImage(self.names2[im])
        # PET_image = sitk.GetArrayFromImage(img_PET)
        # PET_image = np.transpose(PET_image, (2, 1, 0)).astype(np.float32)

        sample1 = {'MRI_image': MRI_image}
        # sample1 = sample1.cuda()
        sample1 = apply_transform(self.transforms, sample1)

        MRI_image = sample1['MRI_image']
        # MRI_image = self.__resize_data__(MRI_image)
        # PET_image = self.__resize_data__(PET_image)

        sample = {'label': label, 'MRI_image': MRI_image}

        return sample

    def __resize_data__(self, data):
        """
            Resize the loaddata to the input size
        """
        [n, depth, height, width] = data.shape
        scale = [n, self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data





class Val_Dataset(data.Dataset):
    def __init__(self, image_dir, class1, class2, transform=None):
        super().__init__()

        self.image_ids = os.listdir(image_dir)
        self.image_dir1 = image_dir
        LABELS = [List[class1], List[class2]]

        labels1 = {name: index for index in range(len(LABELS)) for name in glob.glob(image_dir + '/' + LABELS[index] + '/*.jpg')}  # 给两个文件夹D，N下面的图片打上标签；

        self.labels = labels1
        self.names1 = list(sorted(labels1.keys()))

        self.shape = (len(labels1), 1, 1, 1)  # 这句没看懂；
        self.augment_size = np.prod(self.shape) / len(labels1)  # 这句没看懂；
        self.transforms = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im, rotation, flip, enhance = np.unravel_index(idx, self.shape)

        MRI_image = Get_MRI_data(self.names1[im])

        label = self.labels[self.names1[im]]
        image = self.transforms(MRI_image)

        return image, label


