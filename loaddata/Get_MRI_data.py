import numpy as np
import SimpleITK as sitk


class Get_MRI_data():
    def __int__(self, img_path):
        self.img_path = img_path

    def read_nii(self):
        img_T2 = sitk.ReadImage(self.img_path)
        T2_image = sitk.GetArrayFromImage(img_T2)
        T2_image = np.transpose(T2_image, (2, 1, 0)).astype(np.float32)

        return T2_image
