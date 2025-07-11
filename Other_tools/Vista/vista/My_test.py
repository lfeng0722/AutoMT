import queue
from vista import sample, sample_utils, create_model, run_sampling
import torch


def generate_video(
        first_frame_file_name,
        height=576,
        width=1024,
        n_rounds=2,
        n_frames=10,
        n_steps=15,
        cfg_scale=2.5,
        cond_aug=0.0
):
    model = create_model()

    # 创建一个简单的队列来接收日志信息
    log_queue = queue.SimpleQueue()

    # 调用 run_sampling 函数
    run_sampling(
        log_queue,
        first_frame_file_name,
        height,
        width,
        n_rounds,
        n_frames,
        n_steps,
        cfg_scale,
        cond_aug,
        model
    )

    # 从队列中获取生成的图像
    generated_images = []
    while True:
        msg = log_queue.get()
        if msg == "done":
            break
        else:
            entity_path, entity, times = msg
            if entity_path == "generated_image":
                generated_images.append(entity)

    return generated_images


# 使用示例
first_frame_path = "path/to/your/first_frame.jpg"
video_frames = generate_video(first_frame_path)

# 现在 video_frames 包含了生成的视频帧
# 你可以进一步处理这些帧，比如保存为视频文件或显示它们