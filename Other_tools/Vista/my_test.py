import queue
from vista import sample, sample_utils, create_model, run_sampling
import torch
import sys
import os
import rerun as rr
import gradio_rerun
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def generate_video(first_frame_file_name,speed,height=576,width=1024,n_rounds=1,n_frames=25,n_steps=15,cfg_scale=2.5,cond_aug=0.0):
    motion_bucket_id = speed
    model = create_model()
    # 创建一个简单的队列来接收日志信息
    log_queue = queue.SimpleQueue()
    # 调用 run_sampling 函数
    generated_images, samples_z, inputs = run_sampling(log_queue, first_frame_file_name, height, width, n_rounds, n_frames, n_steps, cfg_scale, cond_aug,
                 motion_bucket_id,model)
    to_pil = transforms.ToPILImage()
    for i in range(generated_images.shape[0]):
        img = to_pil(generated_images[i])
        img.save(f"test/image_{i + 1}.png")
    return generated_images


class VideoGenerator:
    def __init__(self,  height=576, width=1024,n_rounds=1, n_frames=25, n_steps=15, cfg_scale=2.5, cond_aug=0.0):
        self.model = create_model()
        self.to_pil = transforms.ToPILImage()
        self.height = height
        self.width = width
        self.n_rounds = n_rounds
        self.n_frames = n_frames
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.cond_aug = cond_aug


    def __call__(self, first_frame_file_name,speed):
        motion_bucket_id = speed * 10 +100
        log_queue = queue.SimpleQueue()

        generated_images, samples_z, inputs = run_sampling(
            log_queue,
            first_frame_file_name,
            self.height,
            self.width,
            self.n_rounds,
            self.n_frames,
            self.n_steps,
            self.cfg_scale,
            self.cond_aug,
            motion_bucket_id,
            self.model
        )

        return generated_images

# 使用示例
#first_frame_path = "test_1.png"
#video_frames = generate_video(first_frame_path,speed=20)
print(1)