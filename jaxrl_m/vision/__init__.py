from jaxrl_m.vision.impala import impala_configs
from jaxrl_m.vision.bigvision_resnetv2 import resnetv2_configs
from jaxrl_m.vision.small_encoders import small_configs
from jaxrl_m.vision.resnet_v1 import resnetv1_configs
from jaxrl_m.vision.clip import *

encoders = dict()
encoders.update(impala_configs)
encoders.update(resnetv2_configs)
encoders.update(resnetv1_configs)
encoders.update(small_configs)
encoders["clip_text_with_projection"] = CLIPTextEncoderWithProjection
encoders["clip_vision_with_projection"] = CLIPVisionEncoderWithProjection
encoders["clip_text_with_ftmap"] = CLIPTextEncoderWithFtMap
encoders["clip_vision_with_ftmap"] = CLIPVisionEncoderWithFtMap
encoders["muse"] = MUSEPlaceHolder
