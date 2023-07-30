from custom_nodes.comfyui_controlnet_preprocessors import preprocessor

# NODE_CLASS_MAPPINGS = {
#     "Canny_Detector_Preprocessor": Canny_Detector_Preprocessor,
#     "Depth_detetor_Preprocessor": Depth_detetor_Preprocessor,
#     "Normal_Bae_Detector_Preprocessor": Normal_Bae_Detector_Preprocessor,
#     "Openpose_Detector_Preprocessor": Openpose_Detector_Preprocessor,
#     "MLSD_Detector_Preprocessor": MLSD_Detector_Preprocessor,
#     "Lineart_Detector_Preprocessor": Lineart_Detector_Preprocessor,
#     "Softedge_Detector_Preprocessor": Softedge_Detector_Preprocessor,
#     "Scribble_Detector_Preprocessor": Scribble_Detector_Preprocessor,
#     "Shuffle_Detector_Preprocessor": Shuffle_Detector_Preprocessor,
# }

NODE_CLASS_MAPPINGS = {
    **preprocessor.NODE_CLASS_MAPPINGS
}
