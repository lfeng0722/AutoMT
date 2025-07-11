
import os
import pandas as pd
def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
import torchvision.transforms as transforms
import sys
import os
import torch
from diffusers.utils import load_image
from PIL import Image, ImageOps
import math
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import cv2
def resize_image_dimensions(original_resolution_wh, maximum_dimension=1024):
    width, height = original_resolution_wh
    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)
    return new_width, new_height

class Vista:
    def __init__(self):
        sys.path.append(os.path.join("Other_tools", "Vista"))

        from my_test import VideoGenerator
        sys.stdout = open(os.devnull, 'w')
        self.vista_gen = VideoGenerator()
    def gen(self,image_path,speed):
        speed = speed
        generated_images = self.vista_gen(image_path, speed)
        return generated_images

def get_allimgs(excel):
    sheet_data = pd.read_excel(excel, sheet_name="Sheet1")
    images = sheet_data['Image File'].tolist()
    allspeed = sheet_data['Vehicle Speed'].tolist()
    allimg = []
    speeds = []
    for image in images:
        allimg.append(os.path.basename(image))
    for speed in allspeed:
        speeds.append(speed)
    return allimg,speeds

class Pix2PixProcessor:
    def __init__(self,strength=1):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        device = "cuda:1"
        self.strength = strength
        self.pipe.to(device)
        seed=42
        self.generator = torch.manual_seed(seed)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
    def __call__(self, diffusion_prompt, image):
        # Resize image
        image = load_image(image).convert("RGB")
        width, height = image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


        prompt = f"""what would it look like if it were {diffusion_prompt}?"""
        images = self.pipe(
            prompt,
            image=image,
            guidance_scale=self.strength+6.5,
            num_inference_steps=15,
            image_guidance_scale=1.5,
            generator = self.generator,
            disable_progress_bar = True  #
        ).images
        return images[0]


def Get_videos():
    vista = Vista()
    #pix2pix = Pix2PixProcessor(strength=1)
    Results_dirs = ["Data/Results/GPT", "Data/Results/Qwen"]
    list(map(lambda dir: Check_file(dir), Results_dirs))
    source_images = "Data/test/RAW"
    mask_images = "Data/test/OneFormer"
    Csv_dirs = ["Save/GPT_results", "Save/Qwen_results"]


    videos_dirs = ["Data/Results/GPT_videos", "Data/Results/Qwen_videos"]
    list(map(lambda dir: Check_file(dir), videos_dirs))
    Support_pix2pix = ["fog", "rainy", "winter", "wet road", "dark", "night"]

    for ho in range(1):
        Csv_dir = Csv_dirs[ho]
        Results_dir = Results_dirs[ho]
        csv_path = os.path.join(Results_dir, "save.csv")
        df = pd.read_csv(csv_path, encoding='utf-8')

        images = df["Image File"].tolist()
        MRs = df["MR"].tolist()
        prompts = df["analysis"].tolist()
        allimgs, speeds = get_allimgs("Data/test_dataset.xlsx")
        results = []
        iii =10
        for cont in range(len(images)):
            MR = MRs[cont]
            if iii==0:
                normalized_path = os.path.basename(images[cont].replace("\\", "/"))
                prompt = prompts[cont]
                index = allimgs.index(normalized_path)
                matched_images = allimgs[index: index + 10]
                for k in range(10):
                    image_path_ = os.path.join(videos_dirs[ho],matched_images[k])
                    #img = pix2pix(prompt, image_path_)
                    #img.save(image_path_)


            else:
                normalized_path = os.path.basename(images[cont].replace("\\", "/"))
                image = os.path.join(Results_dir, normalized_path)
                image = os.path.abspath(image)
                prompt = prompts[cont]
                index = allimgs.index(normalized_path)
                matched_images = allimgs[index: index + 10]
                if os.path.exists(os.path.join(videos_dirs[ho],matched_images[0])):
                    continue

                speed = speeds[index]
                images_video = vista.gen(image, speed)

                for k in range(10):
                    image_path_ = os.path.join(videos_dirs[ho],matched_images[k])
                    to_pil = transforms.ToPILImage()
                    img = to_pil(images_video[k])
                    img.save(image_path_)


def Generate_videos(input_images, save_path):

    first_image_name = os.path.splitext(os.path.basename(input_images[0]))[0]
    video_path = os.path.join(save_path, f"{first_image_name}.mp4")
    frames = []
    for image_path in input_images:
        frame = cv2.imread(image_path)
        frame_resized = cv2.resize(frame, (320, 160), interpolation=cv2.INTER_AREA)
        frames.append(frame_resized)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
    return video_path

def compare_videos():
    Results_dirs = ["Data/Results/GPT", "Data/Results/Qwen"]
    Csv_dirs = ["Save/GPT_results", "Save/Qwen_results"]

    videos_dirs = ["Data/Results/GPT_videos", "Data/Results/Qwen_videos"]
    list(map(lambda dir: Check_file(dir), videos_dirs))
    Results_dir = Results_dirs[0]
    csv_path = os.path.join("Save/GPT_results/MRs.xlsx")
    df = pd.read_excel(csv_path, sheet_name="Sheet1")

    images = df["frames"].tolist()
    allimgs, speeds = get_allimgs("Data/test_dataset.xlsx")
    for cont in range(len(images)):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        index = allimgs.index(normalized_path)
        matched_images = allimgs[index: index + 10]
        all_img = []
        for k in range(10):
            image_path_ = os.path.join(videos_dirs[0], matched_images[k])
            if os.path.exists(image_path_):
                all_img.append(image_path_)

        if len(all_img)<10:
            continue
        else:
            Generate_videos(all_img,videos_dirs[0])


def Get_videos():
    vista = Vista()
    #pix2pix = Pix2PixProcessor(strength=1)
    Results_dirs = ["Data/Results/GPT", "Data/Results/Qwen"]
    list(map(lambda dir: Check_file(dir), Results_dirs))
    source_images = "Data/test/RAW"
    mask_images = "Data/test/OneFormer"
    Csv_dirs = ["Save/GPT_results", "Save/Qwen_results"]


    videos_dirs = ["Data/Results/GPT_videos", "Data/Results/Qwen_videos"]
    list(map(lambda dir: Check_file(dir), videos_dirs))
    Support_pix2pix = ["fog", "rainy", "winter", "wet road", "dark", "night"]

    for ho in range(1):
        Csv_dir = Csv_dirs[ho]
        Results_dir = Results_dirs[ho]
        csv_path = os.path.join(Results_dir, "save.csv")
        df = pd.read_csv(csv_path, encoding='utf-8')

        images = df["Image File"].tolist()
        MRs = df["MR"].tolist()
        prompts = df["analysis"].tolist()
        allimgs, speeds = get_allimgs("Data/test_dataset.xlsx")
        results = []
        iii =10
        for cont in range(len(images)):
            MR = MRs[cont]
            if iii==0:
                normalized_path = os.path.basename(images[cont].replace("\\", "/"))
                prompt = prompts[cont]
                index = allimgs.index(normalized_path)
                matched_images = allimgs[index: index + 10]
                for k in range(10):
                    image_path_ = os.path.join(videos_dirs[ho],matched_images[k])
                    #img = pix2pix(prompt, image_path_)
                    #img.save(image_path_)


            else:
                normalized_path = os.path.basename(images[cont].replace("\\", "/"))
                image = os.path.join(Results_dir, normalized_path)
                image = os.path.abspath(image)
                prompt = prompts[cont]
                index = allimgs.index(normalized_path)
                matched_images = allimgs[index: index + 10]
                if os.path.exists(os.path.join(videos_dirs[ho],matched_images[0])):
                    continue

                speed = speeds[index]
                images_video = vista.gen(image, speed)

                for k in range(10):
                    image_path_ = os.path.join(videos_dirs[ho],matched_images[k])
                    to_pil = transforms.ToPILImage()
                    img = to_pil(images_video[k])
                    img.save(image_path_)
#Get_videos()
def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
#compare_videos()
vista = Vista()
def RQ(vista,file,validation_csv_path):
    df = pd.read_csv(validation_csv_path)
    diff_col = df['valid_flag_diff']
    savepath =  os.path.join(file, "video")
    Check_file(savepath)
    to_pil = transforms.ToPILImage()
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")

    # 提取字段
    image_files = matched_data["Image File"]
    steering_angles = matched_data["Steering Angle"]
    vehicle_speeds = matched_data["Vehicle Speed"]
    for cont in range(len(image_files)):
        normalized_path = os.path.basename(image_files[cont].replace("\\", "/"))
        if cont%10==0:
            image = os.path.join(file,normalized_path)
            if diff_col[cont//10]==0:
                continue

            if os.path.exists(os.path.join(savepath,  os.path.basename(image_files[cont+0].replace("\\", "/")))):
                continue
            images_video = vista.gen(image, vehicle_speeds[cont])
            index = 0
            for k in range(10):
                image_path_ = os.path.join(savepath,  os.path.basename(image_files[cont+index].replace("\\", "/")))
                img = to_pil(images_video[k])
                img.save(image_path_)
                index+=1
savedirs = ["Data/Results/0","Data/Results/1","Data/Results/2","Data/Results/3"]

for Num in range(5):
    for i in range(4):
        validation_csv_path = f"{savedirs[i]}/{Num}/Validition.csv"
        Temp = savedirs[i] + "/" + str(Num)

        RQ(vista, Temp,validation_csv_path)