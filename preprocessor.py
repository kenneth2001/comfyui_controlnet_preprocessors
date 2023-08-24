import torch
import cv2
from controlnet_aux import CannyDetector, MidasDetector, ZoeDetector, LeresDetector, NormalBaeDetector, OpenposeDetector, MLSDdetector, LineartDetector, LineartAnimeDetector, HEDdetector, PidiNetDetector, ContentShuffleDetector
import numpy as np
from torch.hub import download_url_to_file, get_dir
import os
from urllib.parse import urlparse


def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone() * 255.0
    return img_tensor.squeeze().numpy().astype(np.uint8)


def img_np_to_tensor(img_np_list):
    return torch.from_numpy(img_np_list.astype(np.float32) / 255.0).unsqueeze(0)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(
            url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


class Canny_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image):
        image_np = img_tensor_to_np(image)

        canny = CannyDetector()
        processed_image = canny(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Depth_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "mode": (["depth_midas", "depth_zoe", "depth_leres", "depth_leres++"], {"default": "depth_midas"}), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image, mode):
        image_np = img_tensor_to_np(image)

        try:
            if mode == 'depth_midas':
                depth = MidasDetector.from_pretrained('ckpts')
            elif mode == 'depth_zoe':
                depth = ZoeDetector.from_pretrained('ckpts')
            elif mode in ['depth_leres', 'depth_leres++']:
                depth = LeresDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt',
                               model_dir='ckpts', file_name='ZoeD_M12_N.pt')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt',
                               model_dir='ckpts', file_name='dpt_hybrid-midas-501f0c75.pt')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth',
                               model_dir='ckpts', file_name='res101.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/latest_net_G.pth',
                               model_dir='ckpts', file_name='latest_net_G.pth')
            if mode == 'depth_midas':
                depth = MidasDetector.from_pretrained('ckpts')
            elif mode == 'depth_zoe':
                depth = ZoeDetector.from_pretrained('ckpts')
            elif mode in ['depth_leres', 'depth_leres++']:
                depth = LeresDetector.from_pretrained('ckpts')

        if mode == 'depth_leres++':
            processed_image = depth(image_np, boost=True)
        elif mode == 'depth_leres':
            processed_image = depth(image_np, boost=False)
        else:
            processed_image = depth(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Normal_Bae_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image):
        image_np = img_tensor_to_np(image)

        try:
            normal_bae = NormalBaeDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt',
                               model_dir='ckpts', file_name='scannet.pt')
            normal_bae = NormalBaeDetector.from_pretrained('ckpts')

        processed_image = normal_bae(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Openpose_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "include_face": ([True, False], {"default": False}),
                             "include_hand": ([True, False], {"default": False}),
                             "include_body": ([True, False], {"default": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image, include_face, include_hand, include_body):
        image_np = img_tensor_to_np(image)

        try:
            openpose = OpenposeDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth',
                               model_dir='ckpts', file_name='body_pose_model.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth',
                               model_dir='ckpts', file_name='hand_pose_model.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth',
                               model_dir='ckpts', file_name='facenet.pth')
            openpose = OpenposeDetector.from_pretrained('ckpts')

        processed_image = openpose(
            image_np, include_face=include_face, include_hand=include_hand, include_body=include_body)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class MLSD_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image):
        image_np = img_tensor_to_np(image)

        try:
            mlsd = MLSDdetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth',
                               model_dir='ckpts', file_name='mlsd_large_512_fp32.pth')
            mlsd = MLSDdetector.from_pretrained('ckpts')

        processed_image = mlsd(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Lineart_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "mode": (["coarse", "anime", "realistic"], {"default": "anime"}), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image, mode):
        image_np = img_tensor_to_np(image)

        try:
            if mode in ('coarse', 'realistic'):
                linear_art = LineartDetector.from_pretrained('ckpts')
            elif mode == 'anime':
                linear_art = LineartAnimeDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth',
                               model_dir='ckpts', file_name='netG.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth',
                               model_dir='ckpts', file_name='sk_model.pth')
            linear_art = LineartDetector.from_pretrained('ckpts') if mode in (
                'coarse', 'realistic') else LineartAnimeDetector.from_pretrained('ckpts')

        if mode in ('coarse', 'realistic'):
            processed_image = linear_art(
                image_np, coarse=True if mode == 'coarse' else False)
        elif mode == 'anime':
            processed_image = linear_art(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Softedge_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "mode": (["softedge_hed", "softedge_hedsafe", "softedge_pidinet", "softedge_pidsafe"], {"default": "softedge_hed"}), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image, mode):
        image_np = img_tensor_to_np(image)

        try:
            if mode in ('softedge_hed', 'softedge_hedsafe'):
                softedge = HEDdetector.from_pretrained('ckpts')
            elif mode in ('softedge_pidinet', 'softedge_pidsafe'):
                softedge = PidiNetDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth',
                               model_dir='ckpts', file_name='ControlNetHED.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth',
                               model_dir='ckpts', file_name='table5_pidinet.pth')
            softedge = HEDdetector.from_pretrained('ckpts') if mode in (
                'softedge_hed', 'softedge_hedsafe') else PidiNetDetector.from_pretrained('ckpts')

        processed_image = softedge(image_np, safe=True if mode in (
            'softedge_hedsafe', 'softedge_pidsafe') else False, scribble=False)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Scribble_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "mode": (["scribble_hed", "scribble_hedsafe", "scribble_pidinet", "scribble_pidsafe"], {"default": "scribble_hed"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image, mode):
        image_np = img_tensor_to_np(image)

        try:
            if mode in ('scribble_hed', 'scribble_hedsafe'):
                scribble = HEDdetector.from_pretrained('ckpts')
            elif mode in ('scribble_pidinet', 'scribble_pidsafe'):
                scribble = PidiNetDetector.from_pretrained('ckpts')
        except:
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth',
                               model_dir='ckpts', file_name='ControlNetHED.pth')
            load_file_from_url('https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth',
                               model_dir='ckpts', file_name='table5_pidinet.pth')
            scribble = HEDdetector.from_pretrained('ckpts') if mode in (
                'scribble_hed', 'scribble_hedsafe') else PidiNetDetector.from_pretrained('ckpts')

        processed_image = scribble(image_np, safe=True if mode in (
            'scribble_hedsafe', 'scribble_pidsafe') else False, scribble=True)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


class Shuffle_Detector_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "controlnet_preprocessor"

    CATEGORY = "controlnet_preprocessor"

    def controlnet_preprocessor(self, image):
        image_np = img_tensor_to_np(image)

        shuffle = ContentShuffleDetector()
        processed_image = shuffle(image_np)

        processed_image_np = np.array(processed_image)
        processed_image_tensor = img_np_to_tensor(processed_image_np)
        return (processed_image_tensor,)


NODE_CLASS_MAPPINGS = {
    "Canny_Detector_Preprocessor": Canny_Detector_Preprocessor,
    "Depth_Detector_Preprocessor": Depth_Detector_Preprocessor,
    "Normal_Bae_Detector_Preprocessor": Normal_Bae_Detector_Preprocessor,
    "Openpose_Detector_Preprocessor": Openpose_Detector_Preprocessor,
    "MLSD_Detector_Preprocessor": MLSD_Detector_Preprocessor,
    "Lineart_Detector_Preprocessor": Lineart_Detector_Preprocessor,
    "Softedge_Detector_Preprocessor": Softedge_Detector_Preprocessor,
    "Scribble_Detector_Preprocessor": Scribble_Detector_Preprocessor,
    "Shuffle_Detector_Preprocessor": Shuffle_Detector_Preprocessor,
}
