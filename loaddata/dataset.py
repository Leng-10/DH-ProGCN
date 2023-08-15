import glob
import numpy as np
import SimpleITK as sitk
from torch.utils import data
from monai.transforms import AddChanneld, Compose, RandAffined, RandFlipd, ToTensord, RandZoomd
from monai.transforms import apply_transform
from scipy import ndimage


def Get_MRI_data(path):
    keys_ = ['T1']
    img_MRI = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_MRI)
    ori_img = np.transpose(img, (2, 1, 0)).astype(np.float32)

    sample1 = {'T1': ori_img}
    transforms = Compose([AddChanneld(keys=keys_), ToTensord(keys=keys_)])
    sample1 = apply_transform(transforms, sample1)
    MRI_image = sample1['T1']

    return ori_img, MRI_image


class Dataset_single(data.Dataset):
    def __init__(self, sets, phase):
        self.phase = phase
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W

        keys_ = ['T1']
        if self.phase == 'train':
            self.transforms = Compose([
                AddChanneld(keys=keys_),
                RandAffined(keys=keys_, prob=0.5, translate_range=(4, 6, 6),
                            rotate_range=(np.pi / 36, np.pi / 18, np.pi / 18),
                            mode="bilinear"), RandFlipd(keys=keys_, prob=0.5),
                RandZoomd(keys=keys_, mode="trilinear", prob=0.5, min_zoom=0.9, max_zoom=1.1, align_corners=True),
                ToTensord(keys=keys_, )])
        elif self.phase == 'test':
            self.transforms = Compose([AddChanneld(keys=keys_), ToTensord(keys=keys_)])

        root = sets.root_traindata + '/' + sets.modal
        classes = sets.classes

        labels1 = {name: index for index in range(len(classes)) for name in
                   glob.glob(root + '/' + classes[index] + '/' + phase + '/*.nii')}

        self.labels1 = labels1
        self.names1 = list(sorted(labels1.keys()))
        self.shape = (len(labels1), 1, 1, 1)

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):

        im, rotation, flip, enhance = np.unravel_index(idx, self.shape)
        label = self.labels1[self.names1[im]]
        img_MRI = sitk.ReadImage(self.names1[im])
        MRI_image = sitk.GetArrayFromImage(img_MRI)
        MRI_image = np.transpose(MRI_image, (2, 1, 0)).astype(np.float32)

        sample1 = {'T1': MRI_image}
        sample1 = apply_transform(self.transforms, sample1)

        MRI_image = sample1['T1']
        # MRI_image = self.__resize_data__(MRI_image)

        sample = {'label': label, 'T1': MRI_image}

        return sample

    def __resize_data__(self, data):
        """
            Resize the loaddata to the input size
        """
        [n, depth, height, width] = data.shape
        scale = [n, self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data



class Dataset_dual(data.Dataset):
    def __init__(self, sets, phase):
        self.phase = phase
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W

        keys_ = ['T1', 'PET']
        if self.phase == 'train':
            self.transforms = Compose([
                AddChanneld(keys=keys_),
                RandAffined(keys=keys_, prob=0.5, translate_range=(4, 6, 6),
                            rotate_range=(np.pi / 36, np.pi / 18, np.pi / 18),
                            mode="bilinear"), RandFlipd(keys=keys_, prob=0.5),
                RandZoomd(keys=keys_, mode="trilinear", prob=0.5, min_zoom=0.9, max_zoom=1.1, align_corners=True),
                ToTensord(keys=keys_, )])
        elif self.phase == 'test':
            self.transforms = Compose([AddChanneld(keys=keys_), ToTensord(keys=keys_)])

        root = sets.root_traindata
        classes = sets.classes
        modal = sets.modal

        labels1 = {name: index for index in range(len(classes)) for name in
                   glob.glob(root + '/' + modal[0] + '/' + classes[index] + '/' + phase + '/*.nii')}
        labels2 = {name: index for index in range(len(classes)) for name in
                   glob.glob(root + '/' + modal[1] + '/' + classes[index] + '/' + phase + '/*.nii')}


        self.labels1 = labels1
        self.names1 = list(sorted(labels1.keys()))
        self.names2 = list(sorted(labels2.keys()))
        self.shape = (len(labels1), 1, 1, 1)


    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):

        im, rotation, flip, enhance = np.unravel_index(idx, self.shape)
        label = self.labels1[self.names1[im]]

        img_MRI = sitk.ReadImage(self.names1[im])
        MRI_image = sitk.GetArrayFromImage(img_MRI)
        MRI_image = np.transpose(MRI_image, (2, 1, 0)).astype(np.float32)

        img_PET = sitk.ReadImage(self.names2[im])
        PET_image = sitk.GetArrayFromImage(img_PET)
        PET_image = np.transpose(PET_image, (2, 1, 0)).astype(np.float32)

        sample1 = {'T1': MRI_image, 'PET': PET_image}
        sample1 = apply_transform(self.transforms, sample1)
        MRI_image, PET_image = sample1['T1'], sample1['PET']

        # MRI_image, PET_image = self.__resize_data__(MRI_image), self.__resize_data__(PET_image)

        sample = {'label': label, 'T1': MRI_image, 'PET': PET_image}

        return sample

    def __resize_data__(self, data):
        """
            Resize the loaddata to the input size
        """
        [n, depth, height, width] = data.shape
        scale = [n, self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data




# class Val_Dataset(loaddata.Dataset):
#     def __init__(self, image_dir, class1, class2, transform=None):
#         super().__init__()
#
#         self.image_ids = os.listdir(image_dir)
#         self.image_dir1 = image_dir
#         LABELS = [List[class1], List[class2]]
#
#         labels1 = {name: index for index in range(len(LABELS)) for name in glob.glob(image_dir + '/' + LABELS[index] + '/*.jpg')}  # 给两个文件夹D，N下面的图片打上标签；
#
#         self.labels = labels1
#         self.names1 = list(sorted(labels1.keys()))
#
#         self.shape = (len(labels1), 1, 1, 1)  # 这句没看懂；
#         self.augment_size = np.prod(self.shape) / len(labels1)  # 这句没看懂；
#         self.transforms = transform
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         im, rotation, flip, enhance = np.unravel_index(idx, self.shape)
#
#         MRI_image = Get_MRI_data(self.names1[im])
#
#         label = self.labels[self.names1[im]]
#         image = self.transforms(MRI_image)
#
#         return image, label





