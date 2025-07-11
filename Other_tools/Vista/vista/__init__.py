from __future__ import annotations

import torch
from transformers.utils import hub

from . import sample, sample_utils
import os

def create_model():
    current_dir=os.path.dirname(os.path.abspath(__file__))
    return sample_utils.init_model(
        {
            "config": os.path.join(current_dir, 'configs', 'inference', 'vista.yaml'),
            "ckpt": os.path.join(current_dir, 'vista.safetensors')
        }
    )
#"./vista/configs/inference/vista.yaml",
#"./vista/vista.safetensors",

def run_sampling(
    log_queue,
    first_frame_file_name,
    height,
    width,
    n_rounds,
    n_frames,
    n_steps,
    cfg_scale,
    cond_aug,
    motion_bucket_id,
    model=None,
) -> None:
    if model is None:
        model = create_model()

    unique_keys = set([x.input_key for x in model.conditioner.embedders])
    value_dict = sample_utils.init_embedder_options(unique_keys)

    action_dict = None

    first_frame = sample.load_img(first_frame_file_name, height, width, "cuda")[None]
    repeated_frame = first_frame.expand(n_frames, -1, -1, -1)
    #torch.Size([1, 3, 160, 320]) ->torch.Size([25, 3, 160, 320])
    value_dict = sample_utils.init_embedder_options(unique_keys)# {'fps': 10, 'fps_id': 9, 'motion_bucket_id': 127}
    value_dict["motion_bucket_id"]=motion_bucket_id*10
    cond_img = first_frame
    value_dict["cond_frames_without_noise"] = cond_img
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = cond_img + cond_aug * torch.randn_like(cond_img)#条件噪声
    if action_dict is not None:
        for key, value in action_dict.items():
            value_dict[key] = value

    if n_rounds > 1:
        guider = "TrianglePredictionGuider"
    else:
        guider = "VanillaCFG"
    sampler = sample_utils.init_sampling(
        guider=guider,
        steps=n_steps,
        cfg_scale=cfg_scale,
        num_frames=n_frames,
    )

    uc_keys = [
        "cond_frames",
        "cond_frames_without_noise",
        "command",
        "trajectory",
        "speed",
        "angle",
        "goal",
    ]

    generated_images, samples_z, inputs = sample_utils.do_sample(
        repeated_frame,
        model,
        sampler,
        value_dict,
        num_rounds=n_rounds,
        num_frames=n_frames,
        force_uc_zero_embeddings=uc_keys,
        initial_cond_indices=[0],
        log_queue=log_queue,
    )

    log_queue.put("done")
    return generated_images, samples_z, inputs
