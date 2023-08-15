import os
import cv2
import numpy as np
from scipy import ndimage
from loaddata.dataset import Get_MRI_data
import torch.nn.functional as F

def FeatureExtractor(net, extracted_layers, x):
    outputs = []
    for name, module in net._modules.items():
        if name is "fc": x = x.view(x.size(0), -1)
        x = module(x)
        # print(name)
        if name in extracted_layers:
            feature = x.squeeze(dim=0).cpu()

            outputs.append(feature)
    return outputs


def returnCAM(feature_conv, weight_softmax, class_idx, target_size):
    # generate the class activation maps upsample to 256x256
    # size_upsample = (128, 128, 128)
    # feature_conv = feature_conv.detach().numpy()
    nc, h, w, d = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w * d)))
        cam = cam.reshape(1, h, w, d)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)

        # output_cam.append(cv2.resize(cam_img, size_upsample))
        result = []
        for img in cam_img:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                # img = cv2.resize(img, target_size)
                [d, h, w] = img.shape
                scale = [target_size[0] * 1.0 / d, target_size[1] * 1.0 / h, target_size[2] * 1.0 / w]
                img = ndimage.interpolation.zoom(img, scale, order=0)  # 0最近邻，1双线性插值

            result.append(img)
        result = np.float32(result)

    # return output_cam
    return result

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        # return np.mean(grads, axis=(2, 3), keepdims=True)
        return np.mean(grads, axis=(2, 3, 4), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        # width, height = input_tensor.size(-1), input_tensor.size(-2)
        # return width, height
        depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
        return depth, width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                # img = cv2.resize(img, target_size)
                [d, h, w] = img.shape
                scale = [target_size[0] * 1.0 / d, target_size[1] * 1.0 / h, target_size[2] * 1.0 / w]
                img = ndimage.interpolation.zoom(img, scale, order=0) #0最近邻，1双线性插值

            result.append(img)
        result = np.float32(result)

        return result



    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def generate_camapping(parameters, features_blobs, idx, ori_img, imgname, label, opt):
    params = list(parameters)
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    # generate class activation mapping
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]], ori_img.shape)

    # render the CAM and output
    for j in range(3):
        img = ori_img
        gray_cam = CAMs[0]
        img_num = os.path.split(imgname)[1]
        save_path = r'{}/{} {}/{}({}, {}, {}) {}/axis={}'.format(opt.gradcam_savepath, label, img_num[:-4],
              opt.backbone, opt.backbone_patch, opt.backbone_depth, opt.backbone_dim, os.path.split(opt.model4cluster)[1], j)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(img.shape[j]):
            if j == 0: img2d, gray_cam2d = img[i, :, :], gray_cam[i, :, :]
            if j == 1: img2d, gray_cam2d = img[:, i, :], gray_cam[:, i, :]
            if j == 2: img2d, gray_cam2d = img[:, :, i], gray_cam[:, :, i]

            img2d, gray_cam2d = np.expand_dims(img2d, axis=2), np.expand_dims(gray_cam2d, axis=2)
            img2d = np.concatenate((img2d, img2d, img2d), axis=2)
            gray_cam2d = np.concatenate((gray_cam2d, gray_cam2d, gray_cam2d), axis=2)

            v_2d = show_cam_on_image(img2d.astype(dtype=np.float32) / np.max(img), gray_cam2d, use_rgb=True)
            v_2d = cv2.cvtColor(v_2d, cv2.COLOR_RGB2BGR)
            v_2d = cv2.rotate(v_2d, cv2.ROTATE_90_CLOCKWISE)
            v_2d = cv2.flip(v_2d, 0)

            cv2.imwrite(r"{}/{}.jpg".format(save_path, i), v_2d)

    print('Class activation map is saved in {}'.format(save_path))




def generate_cam(imgname, label, backbone, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

    # get the softmax weight
    params = list(backbone.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    ori_img, pre_img = Get_MRI_data(imgname)
    input_img = pre_img.unsqueeze(0).cuda()


    ## forward pass, the result is the probability of [private, public]
    features_blobs = FeatureExtractor(backbone, opt.target_layer, input_img)
    _, logit = backbone.forward(input_img)
    h_x = F.softmax(logit.cpu(), 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    print('RESULT ON {}: {}'.format(imgname, idx[0]))


    ## generate class activation mapping
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]], ori_img.shape)

    # render the CAM and output
    for j in range(3):
        img = ori_img
        gray_cam = CAMs[0]
        img_num = os.path.split(imgname)[1]
        save_path = r'{}/{} {}/{}({}, {}, {}) {}/axis={}'.format(opt.gradcam_savepath, label, img_num[:-4],
                    opt.backbone, opt.backbone_patch, opt.backbone_depth, opt.backbone_dim, os.path.split(opt.model4cluster)[1], j)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(img.shape[j]):
            if j == 0: img2d, gray_cam2d = img[i, :, :], gray_cam[i, :, :]
            if j == 1: img2d, gray_cam2d = img[:, i, :], gray_cam[:, i, :]
            if j == 2: img2d, gray_cam2d = img[:, :, i], gray_cam[:, :, i]

            img2d, gray_cam2d = np.expand_dims(img2d, axis=2), np.expand_dims(gray_cam2d, axis=2)
            img2d = np.concatenate((img2d, img2d, img2d), axis=2)
            gray_cam2d = np.concatenate((gray_cam2d, gray_cam2d, gray_cam2d), axis=2)

            v_2d = show_cam_on_image(img2d.astype(dtype=np.float32) / np.max(img), gray_cam2d, use_rgb=True)
            v_2d = cv2.cvtColor(v_2d, cv2.COLOR_RGB2BGR)
            v_2d = cv2.rotate(v_2d, cv2.ROTATE_90_CLOCKWISE)
            v_2d = cv2.flip(v_2d, 0)

            cv2.imwrite(r"{}/{}.jpg".format(save_path, i), v_2d)

    return save_path



def generate_gradcam(imgname, label, backbone, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    # get the softmax weight
    params = list(backbone.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax < 0] = 0

    ori_img, pre_img = Get_MRI_data(imgname)
    input_img = pre_img.unsqueeze(0).cuda()


    ## forward pass, the result is the probability of [private, public]
    features_blobs = []
    logit = backbone.forward(input_img)
    h_x = F.softmax(logit.cpu(), 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    print('RESULT ON ' + imgname + ': ' + probs)


    ## generate class activation mapping
    # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]], ori_img.shape)
    GradCAMs = GradCAM(model=backbone.cpu(), target_layers=opt.target_layers, use_cuda=False)
    CAMs =GradCAMs(input_tensor=input_img.cpu(), target_category=target_category)

    # render the CAM and output
    for j in range(3):
        img = ori_img
        gray_cam = CAMs[0]
        img_num = os.path.split(imgname)[1]
        save_path = r'{}/{} {}/{}({}, {}, {})/axis={}'.format(opt.gradcam_savepath, label, img_num[:-4],
                                                              opt.backbone, opt.backbone_patch, opt.backbone_depth, opt.backbone_dim, j)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(img.shape[j]):
            if j == 0: img2d, gray_cam2d = img[i, :, :], gray_cam[i, :, :]
            if j == 1: img2d, gray_cam2d = img[:, i, :], gray_cam[:, i, :]
            if j == 2: img2d, gray_cam2d = img[:, :, i], gray_cam[:, :, i]

            img2d, gray_cam2d = np.expand_dims(img2d, axis=2), np.expand_dims(gray_cam2d, axis=2)

            img2d = np.concatenate((img2d, img2d, img2d), axis=2)
            gray_cam2d = np.concatenate((gray_cam2d, gray_cam2d, gray_cam2d), axis=2)

            v_2d = show_cam_on_image(img2d.astype(dtype=np.float32) / np.max(img), gray_cam2d, use_rgb=True)
            # plt.imshow(v_2d)
            # plt.show()
            v_2d = cv2.cvtColor(v_2d, cv2.COLOR_RGB2BGR)
            v_2d = cv2.rotate(v_2d, cv2.ROTATE_90_CLOCKWISE)
            v_2d = cv2.flip(v_2d, 0)

            cv2.imwrite(r"{}/{}.jpg".format(save_path, i), v_2d)

    print('Class activation map is saved in {}'.format(save_path))