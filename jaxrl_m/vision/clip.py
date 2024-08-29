from transformers import FlaxCLIPModel, CLIPProcessor
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import flax.linen as nn
import flax
import jax
from typing import Dict, Any, Optional
from jaxrl_m.common.common import MLP
from jaxrl_m.vision.resnet_v1 import resnetv1_configs
import flax


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def process_image(images):
    inputs = processor(images=images, return_tensors="np", padding=True)["pixel_values"]
    inputs = jnp.transpose(inputs, (0, 2, 3, 1))
    return inputs


def process_text(text):
    inputs = processor(
        text=text,
        return_tensors="np",
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    inputs = {k: jnp.array(v) for k, v in inputs.items()}
    inputs["position_ids"] = jnp.expand_dims(
        jnp.arange(inputs["input_ids"].shape[1]), axis=0
    ).repeat(inputs["input_ids"].shape[0], axis=0)
    return inputs


clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_def, clip_variables = clip.module, {"params": clip.params}
clip_bind = clip_def.bind(clip_variables)

# clip hack to make it accept 2 images
vision_model, vision_model_vars = clip_bind.vision_model.unbind()
# print(type(vision_model_vars))
pek_key = "embeddings/patch_embedding/kernel".split("/")
vision_model_vars = flax.core.frozen_dict.FrozenDict(vision_model_vars).unfreeze() # vision_model_vars.unfreeze() # 
pek_params = vision_model_vars["params"]["embeddings"]["patch_embedding"]["kernel"]
sg_pek_params = jnp.concatenate([pek_params, pek_params], axis=2) / 2.0
# print(type(vision_model_vars["params"]["embeddings"]["patch_embedding"]))
vision_model_vars["params"]["embeddings"]["patch_embedding"]["kernel"] = sg_pek_params
vision_model_vars = freeze(vision_model_vars)

text_model, text_model_vars = clip_bind.text_model.unbind()
visual_projection, visual_projection_vars = clip_bind.visual_projection.unbind()
text_projection, text_projection_vars = clip_bind.text_projection.unbind()
clip_module_defs = {
    "vision_model": vision_model,
    "text_model": text_model,
    "visual_projection": visual_projection,
    "text_projection": text_projection,
}
clip_module_vars = {
    "vision_model": vision_model_vars["params"],
    "text_model": text_model_vars["params"],
    "visual_projection": visual_projection_vars["params"],
    "text_projection": text_projection_vars["params"],
}


class CLIPVisionEncoderWithProjection(nn.Module):
    # use these keys to locate params
    clip_visual_projection: nn.Module = visual_projection
    clip_vision_model: nn.Module = vision_model
    freeze_encoder: bool = False
    freeze_projection: bool = False
    mlp_kwargs: Dict = None
    normalize: bool = False

    @nn.compact
    def __call__(self, observations, goals, using_embeds=False):
        inputs = jnp.concatenate([observations, goals], axis=-1)
        x = self.clip_vision_model(inputs).pooler_output
        if self.freeze_encoder:
            x = jax.lax.stop_gradient(x)
        x = self.clip_visual_projection(x)
        if self.freeze_projection:
            x = jax.lax.stop_gradient(x)
        if self.normalize:
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        if self.mlp_kwargs is not None:
            x = MLP(**self.mlp_kwargs)(x)
        return x


class CLIPTextEncoderWithProjection(nn.Module):
    clip_text_projection: nn.Module = text_projection
    clip_text_model: nn.Module = text_model
    freeze_encoder: bool = False
    freeze_projection: bool = False
    mlp_kwargs: Dict = None
    normalize: bool = False

    # ignores observations input, but need to match signature
    @nn.compact
    def __call__(self, observations, text_inputs, using_embeds=False):
        inputs = text_inputs
        # TODO our CLIP was tuned with this, but the original CLIP code does .pooler_output
        # fix when we train on more labels? fixed now with new checkpoint
        # print("\n\n\n\n\n")
        # print(inputs)
        # print(inputs.shape)
        # print("\n\n\n\n\n")
        x = self.clip_text_model(**inputs).pooler_output
        if self.freeze_encoder:
            x = jax.lax.stop_gradient(x)
        x = self.clip_text_projection(x)
        if self.normalize:
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        if self.freeze_projection:
            x = jax.lax.stop_gradient(x)
        if self.mlp_kwargs is not None:
            x = MLP(**self.mlp_kwargs)(x)
        return x


class CLIPVisionEncoderWithFtMap(nn.Module):
    resnet_config: str
    clip_kwargs: dict
    resnet_kwargs: dict

    def setup(self):
        self.clip = CLIPVisionEncoderWithProjection(**self.clip_kwargs)
        self.resnet = resnetv1_configs[self.resnet_config](**self.resnet_kwargs)

    @nn.compact
    def __call__(self, observations, goals, raw, using_embeds=False):
        embed = self.clip(observations, goals, using_embeds)
        return self.resnet(raw, embed)


class CLIPTextEncoderWithFtMap(nn.Module):
    resnet_config: str
    clip_kwargs: dict
    resnet_kwargs: dict

    def setup(self):
        self.clip = CLIPTextEncoderWithProjection(**self.clip_kwargs)
        self.resnet = resnetv1_configs[self.resnet_config](**self.resnet_kwargs)

    # ignores observations input, but need to match signature
    @nn.compact
    def __call__(self, observations, text_inputs, raw, using_embeds=False):
        embed = self.clip(observations, text_inputs, using_embeds)
        return self.resnet(raw, embed)


class MUSEPlaceHolder(nn.Module):
    # do nothing, cuz multi_embed already called in batch processing
    @nn.compact
    def __call__(self, observations, text_inputs):
        return text_inputs


from flax.training import checkpoints


def cache_checkpoint(ckpt_dir, cache_dir):
    raw = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    checkpoints.save_checkpoint(cache_dir, raw["state"], raw["step"], keep=1)


def load_params_from_contrastive_checkpoint(ckpt_dir):
    # params are keyed differently in the contrastive script,
    # we just hardcode the mappings here
    raw = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    params = raw["state"]["params"]["params"]
    clip_params = {
        "vision_model": params["encoders_image"]["pretrained_image_encoder"],
        "visual_projection": params["encoders_image"]["image_projection"],
        "text_model": params["encoders_language"]["pretrained_lang_encoder"],
        "text_projection": params["encoders_language"]["text_projection"],
        "temperature": params["temperature"],
    }
    return clip_params


def find_and_replace(params, key, replacement):
    for k in params.keys():
        if k == key:
            params[k] = replacement
            print(f"replaced {key} in params")
            return
        if isinstance(params[k], type(params)):
            find_and_replace(params[k], key, replacement)


def create_from_checkpoint(ckpt_dir):
    global clip_variables
    global clip_def
    clip_component_params = load_params_from_contrastive_checkpoint(ckpt_dir)
    # clip_variables = clip_variables.unfreeze()
    for key, prms in clip_component_params.items():
        find_and_replace(clip_variables, key, prms)
    clip_variables = freeze(clip_variables)
    clip_bind = clip_def.bind(clip_variables)
    return clip_bind


if __name__ == "__main__":
    import numpy as np
    import os
    from PIL import Image

    # ckpt_dir = "gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/clip_ss2_bridge_lang_aug_20230515_222548/checkpoint_1200"
    # ckpt_dir = "gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/clip_ss2_bridge_us_20230517_212730/checkpoint_1200"
    ckpt_dir = "gs://rail-tpus-andre/logs/jaxrl_m_bridgedata/clip_ss2_bridge_normal_20230517_212831/checkpoint_3400"

    clip_bind = create_from_checkpoint(ckpt_dir)

    # # test image
    # image = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
    # image = process_image(image)

    # image = jnp.array(image)
    # image = jnp.concatenate([image, image], axis=-1)

    # # test text
    # text = "some text"
    # text = process_text(text)

    # test encoding
    # output = clip_bind(pixel_values=image, **text)
    # print(output.keys())

    # read labels
    labels = []
    with open("/home/andre/jaxrl_minimal/debug_images/labels.txt", "r") as f:
        for line in f.readlines():
            labels.append([x.strip() for x in line.split(",")])

    img_root_dir = "/home/andre/jaxrl_minimal/debug_images/"

    image_inputs = []
    text_inputs = []

    for s0_path, g_path, lang in labels:
        s0_path = os.path.join(img_root_dir, s0_path)
        g_path = os.path.join(img_root_dir, g_path)

        s0 = np.array(Image.open(s0_path))
        g = np.array(Image.open(g_path))

        # flip image top to bottom, and only rgb channels
        s0 = s0[::-1, ::-1, :][:, :, :3]
        g = g[::-1, ::-1, :][:, :, :3]

        s0 = process_image(s0)
        g = process_image(g)

        image_inputs.append(np.concatenate([s0, g], axis=-1))
        text_inputs.append(lang)

    image_inputs = np.concatenate(image_inputs, axis=0)
    text_inputs = process_text(text_inputs)

    import pickle as pkl

    output = clip_bind(pixel_values=image_inputs, **text_inputs)
    # output 3: newer checkpoint, 3000 steps

    output_name = "_".join(ckpt_dir.split("/")[-2:])

    with open(f"/home/andre/jaxrl_minimal/debug_images/{output_name}.pkl", "wb") as f:
        pkl.dump(output, f)
