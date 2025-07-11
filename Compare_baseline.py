import re

from numpy import iinfo
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import pandas as pd
from qwen_vl_utils import process_vision_info
import os
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
from tqdm import tqdm
import json
import base64
import random
from nunchaku.models import NunchakuFluxTransformer2dModel
import pandas as pd
import torch
from PIL import Image, ImageOps
import gc
import cv2
from diffusers import FluxFillPipeline
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers.utils import load_image
import math
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

import torch
import cv2
import os
import engine.ADS_model as auto_models
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import warnings
import textwrap
warnings.filterwarnings("ignore", category=UserWarning)
class FluxInpainting:
    def __init__(self, image_size=1024):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        device = "cuda:0"
        self.IMAGE_SIZE = image_size
        transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-fill-dev")
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
    def create_mask(self,image_path, mask_path, type="less", bottom_white_percentage=0.6):

        image = Image.open(image_path)
        mask = cv2.imread(mask_path)
        height, width = mask.shape[:2]
        mask_1 = np.ones((height, width), dtype=np.uint8) * 255

        if type == "less":
            elements_to_keep = [10,11,12,13,14,15,16,17,18]
            kernel_size = 30
        elif type == "more":
            elements_to_keep = [2,10,11,12,13,14,15,16,17,18]
            kernel_size = 30
        elif type == "little":
            elements_to_keep = [0,1,2,10]
            kernel_size = 20  # 改成正数

        # 构造初步 mask_1，保留想要的区域为 0，其他为 255
        if len(mask.shape) == 3:
            for element in elements_to_keep:
                mask_1[np.all(mask == element, axis=-1)] = 0
        else:
            for element in elements_to_keep:
                mask_1[mask == element] = 0



        # 执行膨胀操作（放大目标区域）
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        iterations = 1  # 可以增加来更大范围扩张
        mask_1 = cv2.dilate(mask_1, kernel, iterations=iterations)


        mask = Image.fromarray(mask_1)
        return image, mask

    #img, masks = create_mask(image, mask)

    def __call__(self, image_path, mask_path, prompt, type="less", seed=42, strength=0.85, num_inference_steps=40):
        image, mask = self.create_mask(image_path, mask_path, "little")
        # width, height = resize_image_dimensions(image.size, self.IMAGE_SIZE)
        width, height = 512, 512  # image.width,image.height
        # if type == "little":
        #    width, height = 256, 256
        resized_image = image.resize((width, height), Image.LANCZOS)
        resized_mask = mask.resize((width, height), Image.LANCZOS)
        generator = torch.Generator("cpu").manual_seed(seed)
        prompt = prompt
        #prompt = "photorealistic,high detail," + prompt + ",masterpiece,best quality,official art,extremely detailed, 8K, photo, real, realistic,"  # prompt+",photorealistic, (the road facing directly towards the viewer),high detail"#,(The viewpoint must drive on the road, and in the center lane of the road),
        prompt = prompt
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=resized_image,
                mask_image=resized_mask,
                height=width,
                width=height,
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
            ).images[0]
        return result

#23 #349
from PIL import Image, ImageDraw, ImageFont

def save_annotated_image(
        original_image_path: str,
        save_dir: str,
        filename: str,
        mr_text: str,
        max_mr_lines: int = 5,
        font_size: int = 10  # 可手动传入更小字号
):
    """
    将原图缩放为320x160，底部添加注释文字区域（根据像素宽度自动换行），宽度保持320不变。
    使用 Times New Roman 字体进行清晰矢量式渲染，最终保存为 PNG 或 JPG（位图）。
    """
    import os
    from PIL import Image, ImageDraw, ImageFont

    def wrap_text_by_pixel_width(text, font, max_width, draw):
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if draw.textlength(test_line, font=font) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    os.makedirs(save_dir, exist_ok=True)

    try:
        image = Image.open(original_image_path).convert("RGB").resize((320, 160))
    except Exception as e:
        print(f"无法打开图像 {original_image_path}: {e}")
        return

    # 使用 Times New Roman 字体（矢量 .ttf）
    font = None
    try:
        font_path = "C:/Windows/Fonts/times.ttf"  # Windows
        font = ImageFont.truetype(font_path, font_size)
    except:
        try:
            font_path = "/Library/Fonts/Times New Roman.ttf"  # macOS
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"无法加载 Times New Roman 字体，使用默认字体: {e}")
            font = ImageFont.load_default()

    # 清理 MR 文本
    mr_text_clean = mr_text.strip().split("#")[0].strip().replace("AUTOMT", "method").replace("\n", " ")
    dummy_img = Image.new("RGB", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy_img)

    # 自动换行
    mr_wrapped = wrap_text_by_pixel_width(f"MR : {mr_text_clean}", font, 300, draw_dummy)
    mr_wrapped = mr_wrapped[:max_mr_lines]

    # 计算注释高度
    line_spacing = int(font_size * 0.6)
    padding = 10
    line_heights = [
        draw_dummy.textbbox((0, 0), line, font=font)[3] - draw_dummy.textbbox((0, 0), line, font=font)[1]
        for line in mr_wrapped
    ]
    text_height = sum(line_heights) + line_spacing * (len(mr_wrapped) - 1)
    box_height = text_height + 2 * padding

    # 拼接白底区域
    new_image = Image.new("RGB", (320, 160 + box_height), "white")
    new_image.paste(image, (0, 0))

    # 绘制文字
    draw = ImageDraw.Draw(new_image)
    y = 160 + padding
    for line in mr_wrapped:
        draw.text((padding, y), line, fill="black", font=font)
        line_h = draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
        y += line_h + line_spacing

    # 保存图像
    save_path = os.path.join(save_dir, filename)
    try:
        new_image.save(save_path)  # 支持 .png 或 .jpg
    except Exception as e:
        print(f"保存图像失败: {save_path}, 错误: {e}")




class Pix2PixProcessor:
    def __init__(self,strength=0):
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
            num_inference_steps=50,
            image_guidance_scale=1.5,
            generator = self.generator,
            disable_progress_bar = True  #
        ).images
        return images[0]
def extract_unique_road_types(rules,type=0):
    road_types = set()
    if type ==0:
        pattern = r"Given the ego-vehicle approaches to ([^,]+),"
    else:
        pattern = r"Given the ego-vehicle is ([^,]+),"

    for rule in rules:
        match = re.search(pattern, rule)
        if match:
            road_types.add(match.group(1).strip())

    return sorted(road_types)



def process_path(cav_Path, mr_LLM):
    sheet_data = pd.read_csv(cav_Path)
    print(cav_Path)
    images = sheet_data['frames'].tolist()
    MRs = sheet_data['matched_result'].tolist()
    results = 0
    source_images = "Data/test/RAW"
    match_results = []
    for cont in range(len(images)):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        image = os.path.join(source_images, normalized_path)
        match =re.search(r"Given the ego-vehicle is ([^,]+),", MRs[cont])
        if match:
            MR = match.group(1).strip()
        else:
            continue
        result = mr_LLM.match_scene_to_media(image,MR)
        match_results.append(result)
        results+=result
    print(results/384)
    sheet_data["Test1_results"] = match_results
    sheet_data.to_csv(cav_Path, index=False)
def process_path_ITMT(cav_Path, mr_LLM,type=1):
    print(cav_Path)
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type==1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    results = 0
    source_images = "Data/test/RAW"
    match_results = []
    for cont in range(len(images)):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        image = os.path.join(source_images, normalized_path)
        match =re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
        if match:
            MR = match.group(1).strip()
        else:
            continue
        MR = match.group(1).strip()
        result = mr_LLM.match_scene_to_media(image,MR)
        match_results.append(result)
        results+=result
    sheet_data["Test1_results"]=match_results
    sheet_data.to_csv(cav_Path, index=False)
    print(results/384)


def process_path_ITMT(cav_Path, mr_LLM,type=1):
    print(cav_Path)
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type==1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    results = 0
    source_images = "Data/test/RAW"
    match_results = []
    for cont in range(len(images)):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        image = os.path.join(source_images, normalized_path)
        match =re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
        if match:
            MR = match.group(1).strip()
        else:
            continue
        MR = match.group(1).strip()
        result = mr_LLM.match_scene_to_media(image,MR)
        match_results.append(result)
        results+=result
    sheet_data["Test1_results"]=match_results
    sheet_data.to_csv(cav_Path, index=False)
    print(results/384)


def Generate_AUTOMT(cav_Path,Savedir, inpainter,editor, type=1,control=1):
    print(cav_Path)
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()

    results = 0
    source_images = "Data/test/RAW"
    mask_images = "Data/test/OneFormer"
    for cont in range(len(images)):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        image = os.path.join(source_images, normalized_path)
        #if os.path.exists((os.path.join(Savedir, normalized_path))):
        #    continue  # 如果是在循环里，跳过当前迭代
        mask  = os.path.join(mask_images, normalized_path)
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        print(manipulation)
        if os.path.exists(os.path.join(Savedir, normalized_path)):
            continue

        if control==1:
            if "replace" in manipulation:
                processed_image = editor(manipulation, image)
            else:
                processed_image = inpainter(image, mask, prompt=manipulation)
        if control == 2:
            processed_image = editor.edit_image(image, prompt=manipulation)
        processed_image.save(os.path.join(Savedir, normalized_path))
           # path_1 = os.path.join(source_images, normalized_path)
           # path_2 = os.path.join(Savedir, normalized_path)
           # result = mr_LLM.verify_manipulation_added(path_1,path_2,manipulation)
           # match_results.append(result)
           # results += result
   # print(results)
   # sheet_data["Test2_results"] = match_results
   # sheet_data.to_csv(cav_Path, index=False)
   # print(results / 384)

def Check_results(cav_Path,Savedir, LLM,inpainter,editor, type=1):

    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    source_images = "Data/test/RAW"
    mask_images = "Data/test/OneFormer"
    for cont in range(len(images)):
        if cont>166:
            continue
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        image = os.path.join(source_images, normalized_path)
        #if os.path.exists((os.path.join(Savedir, normalized_path))):
        #    continue  # 如果是在循环里，跳过当前迭代
        mask  = os.path.join(mask_images, normalized_path)
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        print(manipulation)
        results = LLM.verify_manipulation_added(os.path.join(source_images, normalized_path),os.path.join(Savedir, normalized_path),manipulation)
        print(results)
        if results==True and "heavy traffic" not in manipulation:
            continue
        else:
            absolute_image_path = os.path.abspath(image)
            if "replace" in manipulation and "heavy traffic" not in manipulation:
                processed_image = editor(manipulation, image)
            else:
                processed_image = inpainter(image, mask, prompt=manipulation)
            processed_image.save(os.path.join(Savedir, normalized_path))
           # path_1 = os.path.join(source_images, normalized_path)
           # path_2 = os.path.join(Savedir, normalized_path)
           # result = mr_LLM.verify_manipulation_added(path_1,path_2,manipulation)
           # match_results.append(result)
           # results += result


def Eval_results(cav_Path,Savedir, LLM, type=1,ranges=0.01):
    Savedir =  os.path.join(Savedir, "video_1")
    Udacity_Ads = ADSs("Udacity")
    A2D2_Ads = ADSs("A2D2")
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    source_images = "Data/test/data"
    mask_images = "Data/test/OneFormer"
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")
    image_files = matched_data["Image File"]
    total_1 = torch.zeros(6, dtype=torch.int32)
    total_2 = torch.zeros(6, dtype=torch.int32)
    for cont in tqdm(range(len(images))):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        try:
            match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle is ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle approaches to ([^,]+)\nWhen", MRs[cont])
            road = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            road = "no road"
        R1 = LLM.match_scene_to_media(os.path.join(source_images, normalized_path),os.path.join(Savedir, normalized_path), road)
        R2 = LLM.verify_manipulation_added(os.path.join(source_images, normalized_path),
                                           os.path.join(Savedir, normalized_path), manipulation)
        cont_1 += R1
        cont_2 += R2
        cont_3 += R1*R2
        o_paths = []
        f_paths = []
        for k in range(10):
            image_path_ = os.path.basename(image_files[cont*10 + k].replace("\\", "/"))
            o_paths.append(os.path.join(source_images, image_path_))
            f_paths.append(os.path.join(Savedir, image_path_))
        if cont<167:
            violation_matrix = Udacity_Ads.process_images_with_violation(o_paths,f_paths,MRs[cont],ranges)
        else:
            violation_matrix = A2D2_Ads.process_images_with_violation(o_paths, f_paths, MRs[cont],ranges)
        violation_matrix = torch.tensor(violation_matrix)
        if R1*R2==1:
            total_1 = total_1+violation_matrix
        total_2 = total_2+violation_matrix

        if cont == 166:
            x = 167
            print(f"Validity Rate 1 (Scene Match): {cont_1 / x:.4f}")
            print(f"Validity Rate 2 (Manipulation Match): {cont_2 / x:.4f}")
            print(f"Validity Rate All (Scene ∩ Manipulation): {cont_3 / x:.5f}")

            print("Violation Rate Without Validity Filter (All MR Applied):")
            print([val / x for val in total_2])

            print("Violation Rate With Validity Filter (Only Valid Scene+Manipulation):")
            average = torch.mean(torch.stack([val / x for val in total_1]))
            print(f"{average.item():.4f}")  # 例如输出: 0.0
            cont_1 = 0
            cont_2 = 0
            cont_3 = 0
            total_1 = torch.zeros(6, dtype=torch.int32)
            total_2 = torch.zeros(6, dtype=torch.int32)
    x = 217
    print(f"Validity Rate 1 (Scene Match): {cont_1 / x:.4f}")
    print(f"Validity Rate 2 (Manipulation Match): {cont_2 / x:.4f}")
    print(f"Validity Rate All (Scene ∩ Manipulation): {cont_3 / x:.5f}")

    print("Violation Rate Without Validity Filter (All MR Applied):")
    print([val / x for val in total_2])

    print("Violation Rate With Validity Filter (Only Valid Scene+Manipulation):")
    average = torch.mean(torch.stack([val / x for val in total_1]))
    print(f"{average.item():.4f}")  # 例如输出: 0.0209

def compute_validity_rates(cont_1, cont_2, cont_3, cont_4, x):
    result = {
        "Validity Rate 1": float(f"{cont_1 / x:.4f}"),
        "Validity Rate 2": float(f"{cont_2 / x:.4f}"),
        "Validity Rate 3": float(f"{cont_3 / x:.4f}"),
        "Validity Rate All": float(f"{cont_4 / x:.4f}")
    }
    print(result)
    return result
def Eval_results_3(cav_Path,Savedir, LLM, type=1,ranges=0.01):

    Savedir =  os.path.join(Savedir, "0")

    Udacity_Ads = ADSs("Udacity")
    A2D2_Ads = ADSs("A2D2")
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    source_images = "Data/test/data"
    mask_images = "Data/test/OneFormer"
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0
    cont_4 = 0
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")
    image_files = matched_data["Image File"]
    vehicle_speeds = matched_data["Vehicle Speed"]
    total_1 = torch.zeros(6, dtype=torch.int32)
    total_2 = torch.zeros(6, dtype=torch.int32)
    valid_flags = []
    csv_path = []
    for cont in tqdm(range(len(images))):
        speed = vehicle_speeds[cont*10]
        csv_path.append(images[cont])
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        try:
            match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle is ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle approaches to ([^,]+)\nWhen", MRs[cont])
            road = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            road = "no road"
        R1 = LLM.match_scene_to_media(os.path.join(source_images, normalized_path),os.path.join(Savedir, normalized_path), road)
        R2 = LLM.verify_manipulation_added(os.path.join(source_images, normalized_path),
                                           os.path.join(Savedir, normalized_path), manipulation)
        R3 =  LLM.selfcheck(MRs[cont],speed)
        cont_1 += R1
        cont_2 += R2
        cont_3 +=R3
        cont_4 +=    R1*R2*R3
        valid_flags.append(R1*R2*R3)
        o_paths = []
        f_paths = []
        for k in range(10):
            image_path_ = os.path.basename(image_files[cont*10 + k].replace("\\", "/"))
            o_paths.append(os.path.join(source_images, image_path_))
            f_paths.append(os.path.join(Savedir, image_path_))
        if cont == 166:
            x = 167
            result1 = compute_validity_rates(cont_1, cont_2, cont_3, cont_4, x)

            cont_1 = 0
            cont_2 = 0
            cont_3 = 0
            cont_4 = 0
    x = 217
    result2 = compute_validity_rates(cont_1, cont_2, cont_3, cont_4, x)
    return result1,result2,csv_path,valid_flags

def Eval_results_v(cav_Path, Savedir,validation_csv_path , type=1, ranges=0.01):
    df = pd.read_csv(validation_csv_path)
    diff_col = df['valid_flag_diff']
    Savedir = os.path.join(Savedir, "video_1")
    Udacity_Ads = ADSs("Udacity")
    A2D2_Ads = ADSs("A2D2")
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    source_images = "Data/test/data"
    mask_images = "Data/test/OneFormer"
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")
    image_files = matched_data["Image File"]
    total_1 = torch.zeros(6, dtype=torch.int32)
    total_2 = torch.zeros(6, dtype=torch.int32)
    for cont in tqdm(range(len(images))):
        if cont == 166:
            x = 167
            average = torch.mean(torch.stack([val / x for val in total_1]))
            print(f"{average.item():.4f}")  # 例如输出: 0.0
            total_1_stage1 = total_1.clone()
            total_2_stage1 = total_2.clone()
            total_1 = torch.zeros(6, dtype=torch.int32)
            total_2 = torch.zeros(6, dtype=torch.int32)
        if diff_col[cont] == 0:
            continue
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))

        o_paths = []
        f_paths = []
        for k in range(10):
            image_path_ = os.path.basename(image_files[cont * 10 + k].replace("\\", "/"))
            o_paths.append(os.path.join(source_images, image_path_))
            f_paths.append(os.path.join(Savedir, image_path_))
        if cont < 167:
            violation_matrix = Udacity_Ads.process_images_with_violation(o_paths, f_paths, MRs[cont], ranges)
        else:
            violation_matrix = A2D2_Ads.process_images_with_violation(o_paths, f_paths, MRs[cont], ranges)
        violation_matrix = torch.tensor(violation_matrix)
        if diff_col[cont] == 1:
            total_1 = total_1 + violation_matrix
        total_2 = total_2 + violation_matrix



    x = 217
    average = torch.mean(torch.stack([val / x for val in total_1]))
    print(f"{average.item():.4f}")  # 例如输出: 0.0209
    return total_1_stage1 ,total_2_stage1 ,total_1,total_2

def Eval_results_4(cav_Path,Savedir, LLM=0, type=1,ranges=0.01):
    Savedir =  os.path.join(Savedir)
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    source_images = "Data/test/data"
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")
    for cont in tqdm(range(len(images))):

        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        sourpath = os.path.join(source_images, normalized_path)
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        try:
            match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle is ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle approaches to ([^,]+)\nWhen", MRs[cont])
            road = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            road = "no road"
        save_annotated_image(
            original_image_path=os.path.join(Savedir, normalized_path),
            save_dir=os.path.join(Savedir, "annotated_results"),
            filename=f"{cont:03d}.jpg",
            mr_text=MRs[cont],
        )

def Eval_results_1(cav_Path,Savedir, LLM, type=1,ranges=0.01):
    Savedir =  os.path.join(Savedir)
    sheet_data = pd.read_csv(cav_Path)
    images = sheet_data['frames'].tolist()
    if type == 1:
        MRs = sheet_data['matched_MR'].tolist()
    else:
        MRs = sheet_data['matched_result'].tolist()
    source_images = "Data/test/data"
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0
    for cont in tqdm(range(len(images))):
        normalized_path = os.path.basename(images[cont].replace("\\", "/"))
        try:
            match = re.search(r"AUTOMT (.+?),", MRs[cont])

            if not match:
                match = re.search(r"AUTOMT (.+?)\nThen", MRs[cont])
            manipulation = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            manipulation = "Same"
        try:
            match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle is ([^,]+),", MRs[cont])
            if not match:
                match = re.search( r"Given the ego-vehicle approaches to ([^,]+)\nWhen", MRs[cont])
            road = match.group(1).strip()
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            road = "no road"
        R1 = LLM.match_scene_to_media(os.path.join(source_images, normalized_path),os.path.join(Savedir, normalized_path), road)
        R2 = LLM.verify_manipulation_added(os.path.join(source_images, normalized_path),
                                           os.path.join(Savedir, normalized_path), manipulation)
        cont_1 += R1
        cont_2 += R2
        cont_3 += R1*R2


        if cont == 166:
            x = 167
            print(f"Validity Rate 1 (Scene Match): {cont_1 / x:.3f}")
            print(f"Validity Rate 2 (Manipulation Match): {cont_2 / x:.3f}")
            print(f"Validity Rate All (Scene ∩ Manipulation): {cont_3 / x:.5f}")
            cont_1 = 0
            cont_2 = 0
            cont_3 = 0
    x = 217
    print(f"Validity Rate 1 (Scene Match): {cont_1 / x:.3f}")
    print(f"Validity Rate 2 (Manipulation Match): {cont_2 / x:.3f}")
    print(f"Validity Rate All (Scene ∩ Manipulation): {cont_3 / x:.5f}")

"""
print(f"Validity Rate 1 (Scene Match): {cont_1 / 384:.3f}")
            print(f"Validity Rate 2 (Manipulation Match): {cont_2 / 384:.3f}")
            print(f"Validity Rate All (Scene ∩ Manipulation): {cont_3 / 384:.3f}")
        
            print("Violation Rate Without Validity Filter (All MR Applied):")
            print([val / 384 for val in total_2])
        
            print("Violation Rate With Validity Filter (Only Valid Scene+Manipulation):")
            print([val / 384 for val in total_1])
        
            print("Used Violation Ranges:")
            print(ranges)


"""
class Matching_MR:
    def __init__(self, VLM="Qwen",cuda_type=0):
        # Two cuda  device

        if cuda_type == 0:
            self.cuda1 = "cuda:0"
            self.cuda2 = "cuda:0"
        else:
            self.cuda1 = "cuda:1"
            self.cuda2 = "cuda:1"
        self.VLM = VLM
        if self.VLM == "Qwen":
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map=self.cuda1
            )
        self.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.selected_mrs = []  # Store previously selected MRs
        self.max_history = 3  # Maximum number of MRs to keep in history
        self.Wrong = 0

    def analyze_image(self, image):
        if self.VLM == "Qwen":
            prompt = (
                f"Analyze this driving scene. Describe the road types (must be one of the following: {', '.join(self.road_types)}), "
                f"'all roads' can be used if no specific type fits). "
                f"Reply format: road network: <road_type>"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "video": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_result = output_text[0].strip()
            # Step 1: Remove prefix if present
            if raw_result.lower().startswith("road network:"):
                road_type = raw_result[len("road network:"):].strip()
            else:
                road_type = raw_result


            # Step 2: Validate road_type
            if road_type not in self.road_types:
                road_type = random.choice(self.road_types)

            matching_MRs = [mr for mr in self.MRs if road_type in mr]

            # Step 4: Randomly select one if there are matches
            if matching_MRs:
                selected_MR = random.choice(matching_MRs)
            else:
                # fallback: randomly select any MR
                selected_MR = random.choice(self.MRs)

            return selected_MR

    def verify_manipulation_added(self, original_image_path, modified_image_path, manipulation):
        if self.VLM == "Qwen":
            prompt = (
                "You are a visual reasoning AI. "
                f"Determine whether the second image has correctly added or modified elements according to the manipulation description: '{manipulation}'. "
                "The first image is the original. The second image should contain the added/modified content. "
                "Focus on key changes relevant to driving scenes: road signs, vehicles, pedestrians, objects, etc. "
                "Be tolerant to small differences, but focus on whether the change matches the manipulation intent.\n\n"
                "Reply only with 'Yes' if the change is correctly made, or 'No' if not."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_image_path},
                        {"type": "image", "image": modified_image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0].strip().lower()

            if "s" in result:  # Yes
                return 1
            elif "o" in result:  # No
                return 0
            else:
                return 0



    def match_scene_to_media(self, original_image_path, modified_image_path, scene_description):
        if self.VLM == "Qwen":
            prompt = (
                "You are a visual reasoning AI specialized in driving scenes. "
                "You will be shown two images. "
                f"The given scenario description is: '{scene_description}'. "
                "Determine whether **both** images match this scenario. "
                "Focus on key driving elements like road types, lane markings, vehicles, traffic signs, and pedestrians. "
                "Minor visual differences are acceptable, but both images must clearly represent the described scenario. "
                "If the description is generic (e.g., 'all roads'), reply 'Yes' unless the image is clearly unrelated.\n\n"
                "Reply only with 'Yes' if both images reasonably match the scenario description, or 'No' otherwise."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_image_path},
                        {"type": "image", "image": modified_image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0].strip().lower()

            if "s" in result: #Yes
                return 1
            elif "o" in result:#No
                return 0
            else:
                return 0

"""
class QwenChatbot:
    def __init__(self, cuda="cuda:0", model_name="Qwen/Qwen2.5-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map={"": cuda}
        )
        self.system_prompt = "You are an AI assistant specialized in evaluating the **validity of metamorphic relations (MRs)** for autonomous driving test generation."

    def generate_response(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        # Decode only the new tokens
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response
"""
class QwenChatbot:
    def __init__(self, cuda="cuda:0", model_name="Qwen/Qwen2.5-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map={"": cuda}
        )
        self.system_prompt = "You are an AI assistant specialized in judging correctness."

    def generate_response(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        # Decode only the new tokens
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response
class Matching_MR_new:
    def __init__(self, VLM="Qwen",cuda_type=0):
        # Two cuda  device
        self.cuda1 = "cuda:0"
        self.cuda2 = "cuda:1"
        self.VLM = VLM
        if self.VLM == "Qwen":
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map=self.cuda2
            )
        self.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.selected_mrs = []  # Store previously selected MRs
        self.max_history = 3  # Maximum number of MRs to keep in history
        self.Wrong = 0
        self.chatbot = QwenChatbot()

    def analyze_image(self, image):
        if self.VLM == "Qwen":
            prompt = (
                f"Analyze this driving scene. Describe the road types (must be one of the following: {', '.join(self.road_types)}), "
                f"'all roads' can be used if no specific type fits). "
                f"Reply format: road network: <road_type>"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "video": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_result = output_text[0].strip()
            # Step 1: Remove prefix if present
            if raw_result.lower().startswith("road network:"):
                road_type = raw_result[len("road network:"):].strip()
            else:
                road_type = raw_result


            # Step 2: Validate road_type
            if road_type not in self.road_types:
                road_type = random.choice(self.road_types)

            matching_MRs = [mr for mr in self.MRs if road_type in mr]

            # Step 4: Randomly select one if there are matches
            if matching_MRs:
                selected_MR = random.choice(matching_MRs)
            else:
                # fallback: randomly select any MR
                selected_MR = random.choice(self.MRs)

            return selected_MR

    def verify_manipulation_added(self, original_image_path, modified_image_path, manipulation):
        if self.VLM == "Qwen":
            prompt = (
                "You are a visual reasoning AI. "
                f"Determine whether the second image has correctly added or modified elements according to the manipulation description: '{manipulation}'. "
                "The first image is the original. The second image should contain the added/modified content. "
                "Only check if the described objects are present in the second image, ignore other differences."
                "Be tolerant to small differences, but focus on whether the change matches the manipulation intent.\n\n"
                "Reply only with 'Yes' if the change is correctly made, or 'No' if not."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_image_path},
                        {"type": "image", "image": modified_image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0].strip().lower()

            if "s" in result:  # Yes
                return 1
            elif "o" in result:  # No
                return 0
            else:
                return 0



    def match_scene_to_media(self, original_image_path, modified_image_path, scene_description):
        if self.VLM == "Qwen":
            prompt = (
                "You are a visual reasoning AI specialized in driving scenes. "
                "You will be shown two images. "
                f"The given scenario description is: '{scene_description}'. "
                "Determine whether **both** images match this scenario. "
                "Focus on key driving elements like road types, lane markings, vehicles, traffic signs, and pedestrians. "
                "Minor visual differences are acceptable, but both images must clearly represent the described scenario. "
                "If the description is generic (e.g., 'all roads'), reply 'Yes' unless the image is clearly unrelated.\n\n"
                "Reply only with 'Yes' if both images reasonably match the scenario description, or 'No' otherwise."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_image_path},
                        {"type": "image", "image": modified_image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0].strip().lower()

            if "s" in result: #Yes
                return 1
            elif "o" in result:#No
                return 0
            else:
                return 0

    def selfcheck(self,MR,speed):
        speed = speed*3.6
        result_1 = "yes"
        if "turn left" in MR or "turn right" in MR:
            prompt_logic = f"""
            Your task is to determine whether the ego-vehicle’s expected behavior is clearly impossible.
            Example:  
            - turn left must turn sign → turn left  -> yes
            - Adding a speed limit sign → "turn left"   -> no
            respond yes or no based on the MR.
            if answer is no please tell me why
            MR: {MR} """
            result_1 = self.chatbot.generate_response(prompt_logic)
            print(MR, speed, result_1)
        result_2 = "yes"
        if "slow down" in MR:
            if "km/h" in MR:

                prompt_behavior = f"""You are given a metamorphic relation (MR) and the current ego-vehicle speed.
                The expected behavior is considered unreasonable if:
                - The MR adds a speed limit lower than the current speed, but the vehicle is already within the limit.
                Only answer "yes" or "no".
                Metamorphic Relation:
                {MR} Current speed of the ego-vehicle: {speed} km/h
                if answer is no please tell me why"""
                result_2 = self.chatbot.generate_response(prompt_behavior)
                print(MR, speed,  result_2)


        if "yes" in result_1 and "yes" in result_2:
            return 1
        else:

            return 0



class ADSs:
    def __init__(self, dataset="Udacity"):
        self.model_classes = {
            'Epoch': auto_models.Epoch,
            'Resnet101': auto_models.Resnet101,
            'Vgg16': auto_models.Vgg16,
            'PilotNet': auto_models.PilotNet,
            'CNN_LSTM': auto_models.CNN_LSTM,
            'CNN_3D': auto_models.CNN_3D
        }
        self.dataset = dataset
        self.device = "cuda:1"
        self.model_names = ['Epoch', 'Resnet101', 'Vgg16', 'PilotNet', 'CNN_3D', 'CNN_LSTM']
        self.single_model_names = ['Epoch', 'Resnet101', 'Vgg16', 'PilotNet']
        self.times_model_names = ['CNN_3D', 'CNN_LSTM']
        self.steering_models = {}  # GPU 0
        self.speed_models = {}  # GPU 1
        self.steering_device = torch.device(self.device)
        self.speed_device = torch.device(self.device)
        self.load_models()
        self.a = torch.zeros(1, 1, 1)
        self.b = torch.zeros(1, 1, 1)

    def load_models(self):
        use_state = 0
        root_dir = os.path.join("Data", self.dataset, "Save")
        # root_dir = "models"
        for model_name in self.model_names:
            steering_model = self.model_classes[model_name](Use_states=False)
            speed_model = self.model_classes[model_name](Use_states=False)

            save_path = os.path.join(root_dir, f"{model_name}_steering_{str(int(use_state))}.pth")
            model_state_dict = torch.load(save_path, map_location=self.steering_device)
            steering_model.load_state_dict(model_state_dict)
            steering_model = steering_model.to(self.steering_device)
            steering_model.eval()

            save_path = os.path.join(root_dir, f"{model_name}_speed_{str(int(use_state))}.pth")
            model_state_dict = torch.load(save_path, map_location=self.speed_device)
            speed_model.load_state_dict(model_state_dict)
            speed_model = speed_model.to(self.speed_device)
            speed_model.eval()

            setattr(self, f"{model_name}_steering", steering_model)
            setattr(self, f"{model_name}_speed", speed_model)
            self.steering_models[model_name] = steering_model
            self.speed_models[model_name] = speed_model

    def get_model(self, model_name, pred_mode):
        if pred_mode == "steering":
            return self.steering_models[model_name], self.steering_device
        else:
            return self.speed_models[model_name], self.speed_device

    def process_one_image(self, image_paths):
        img = cv2.imread(image_paths)
        height, width = img.shape[:2]
        if (width, height) != (320, 160):
            img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_LANCZOS4)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img.astype(np.float32) / 127.5 - 1.0
        img_sequence = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        return img_sequence

    def process_images(self, image_paths):
        img_sequence_time = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"图片不存在: {image_path}")

            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            if (width, height) != (320, 160):
                img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_LANCZOS4)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            img = img.astype(np.float32) / 127.5 - 1.0
            img_sequence_time.append(img)
        img = torch.from_numpy(np.array(img_sequence_time)).float().permute(0, 3, 1, 2)
        return img

    def process(self, image_paths):
        processed_one_images = []
        processed_images = []

        file_one_name = []
        file_name = []
        index = 0
        scenario_cont = 0
        for i in range(len(image_paths)):
            img_sequence = self.process_one_image(image_paths[i])
            processed_one_images.append(img_sequence)
            file_one_name.append(image_paths[i])
        scenario_cont = 0
        i = 0
        while i < len(image_paths) - 5:
            img_sequence = self.process_images(image_paths[i:i + 5])
            processed_images.append(img_sequence)
            file_name.append(image_paths[i + 5])

            scenario_cont += 1
            if scenario_cont == 5:  # 当前场景已处理 5 个序列
                scenario_cont = 0
                i += 5  # 直接跳到下一个场景的起始点
            i += 1
        batch_one_img = torch.stack(processed_one_images, dim=0)
        batch_img = torch.stack(processed_images, dim=0)
        batch_img = torch.tensor(batch_img)
        batch_one_img = torch.tensor(batch_one_img)

        return batch_one_img, file_one_name, batch_img, file_name

    def pridection(self, image_paths):
        batch_one_img, file_one_name, batch_img, file_name = self.process(image_paths)
        batch_size = 10
        single_dataset = TensorDataset(batch_one_img)
        time_dataset = TensorDataset(batch_img)

        single_loader = DataLoader(single_dataset, batch_size=batch_size, shuffle=False)
        time_loader = DataLoader(time_dataset, batch_size=batch_size, shuffle=False)

        steerings = []
        speeds = []
        with torch.no_grad():
            for model_name in self.single_model_names:
                temp_speed = []
                temp_steer = []
                steering_model = self.steering_models[model_name]
                speed_model = self.speed_models[model_name]
                for batch in single_loader:
                    inputs = (batch[0]).to(self.device)

                    steer, _ = steering_model(inputs, self.a, self.b)
                    speed, _ = speed_model(inputs, self.a, self.b)
                    temp_speed.append(speed)
                    temp_steer.append(steer)
                steerings.append(torch.cat(temp_steer, dim=0))
                speeds.append(torch.cat(temp_speed, dim=0))

            for model_name in self.times_model_names:
                steering_model = self.steering_models[model_name]
                speed_model = self.speed_models[model_name]
                for batch in time_loader:
                    inputs = (batch[0]).to(self.device)
                    steer, _ = steering_model(inputs, self.a, self.b)
                    speed, _ = speed_model(inputs, self.a, self.b)
                    temp_speed.append(speed)
                    temp_steer.append(steer)
                steerings.append(torch.cat(temp_steer, dim=0))
                speeds.append(torch.cat(temp_speed, dim=0))
        return steerings,speeds

    def process_images_with_violation(self, original_paths, follow_paths, rule,ranges=0.01):
        violation_matrix = []
        original_steerings, original_speeds = self.pridection(original_paths)
        follow_steerings, follow_speeds = self.pridection(follow_paths)

        for model_index in range(6):
            o_speed = original_speeds[model_index]
            f_speed = follow_speeds[model_index]
            o_steer = original_steerings[model_index]
            f_steer = follow_steerings[model_index]
            violate = self.get_violation(rule, o_speed, f_speed, o_steer, f_steer,ranges)
            violation_matrix.append(violate)
        return violation_matrix


    def get_violation(self, rule, o_speed, f_speed, o_steer, f_steer,ranges=0):
        try:
            if "slow" or "Slow" in rule:
                alpha, beta = -1, -1 * ranges
                violate = self.count_violations_by_average(o_speed, f_speed, alpha, beta)
            elif "turn right" or "Turn right" in rule:
                alpha, beta = ranges, 5
                violate = self.count_violations_by_average(o_steer, f_steer, alpha, beta)
            elif "turn left" in rule:
                alpha, beta = -5, -1 * ranges
                violate = self.count_violations_by_average(o_steer, f_steer, alpha, beta)
            elif "keep" in rule:
                if ranges==0:
                    ranges = 0.001
                alpha, beta = -1 * ranges, ranges
                violate = self.count_violations_by_average(o_steer, f_steer, alpha, beta)
                violate_ = self.count_violations_by_average(o_speed, f_speed, alpha, beta)
                if violate + violate_ != 0:
                    violate = 1
            else:
                violate = 0
        except (TypeError, AttributeError) as e:  # 处理可能的异常
            violate = 0

        return violate


    def count_violations_by_average(self,A, B, alpha, beta):
        avg_A = torch.mean(torch.tensor([x.item() if isinstance(x, torch.Tensor) else float(x) for x in A]))
        avg_B = torch.mean(torch.tensor([x.item() if isinstance(x, torch.Tensor) else float(x) for x in B]))
        if avg_A != 0:
            change = (avg_B - avg_A) / avg_A
            if alpha < change and change < beta:
                violate = 0
            else:
                violate = 1
        else:
            violate = 1 if (alpha + beta) * avg_B < 0 else 0
        return violate


#Baseline3()
#Baseline(Excel_name="Baseline2",baseline_num=2)
csvpaths = ["mr_results.csv","Baseline1.csv","Baseline2.csv","Baseline3.csv"]
savedirs = ["Data/Results/0","Data/Results/1","Data/Results/2","Data/Results/3"]
#inpainter = FluxInpainting()
#ii = 3
#if ii==1:

#    mr_LLM = Matching_MR()
#    process_path_ITMT(csvpaths[0],mr_LLM)
#    process_path(csvpaths[1],mr_LLM)
#elif ii==2:
#    mr_LLM = Matching_MR(cuda_type=1)
#    process_path(csvpaths[2],mr_LLM)
#    process_path_ITMT(csvpaths[3],mr_LLM,2)
import sys
sys.path.append(os.path.join("Other_tools", "Step1X-Edit"))

from inference import ImageEditor
control=1
if control==1:
    inpainter = FluxInpainting()
    editor = Pix2PixProcessor(strength=0)
elif control==2:
    editor = ImageEditor(model_path="Other_tools/Step1X-Edit/model", size_level=512, quantized=True, offload=True)
    #inpainter = FluxInpainting()
    #editor = Pix2PixProcessor(strength=0)
    mr_LLM = Matching_MR(cuda_type=1)

print(1)


def resize_from_video_to_video1(savedirs):
    src_dir = os.path.join(savedirs, "video")
    out_dir = os.path.join(savedirs, "video_1")
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".png"):
            img_path = os.path.join(src_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, (320, 160))
                cv2.imwrite(os.path.join(out_dir, f), resized)

#for i in range(4):
for Num in range(5):
    for i in range(4):
        cavpath = "Save/"+str(Num)+"/"+csvpaths[i]
        check_and_create = lambda path: os.makedirs(path, exist_ok=True)
        Temp = savedirs[i]+"/"+str(Num)
        check_and_create(Temp)
        if i==0:
            Generate_AUTOMT(cavpath,Temp,inpainter,editor,control=control)
        else:
            Generate_AUTOMT(cavpath, Temp, inpainter, editor,type=2, control=control)


    #
def resize_from_video_to_video2(savedirs):
    # 创建目标目录 savedirs + "/0"（如果不存在）
    output_dir = os.path.join(savedirs, "0")
    os.makedirs(output_dir, exist_ok=True)  # 如果目录已存在则不会报错

    for f in sorted(os.listdir(savedirs)):
        if f.endswith(".png"):
            img_path = os.path.join(savedirs, f)
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, (320, 160))
                # 保存到 savedirs + "/0" 目录下
                output_path = os.path.join(output_dir, f)
                cv2.imwrite(output_path, resized)
import json
def append_result_to_json(new_result, json_path="results.json"):
    # 如果文件已存在，先读取原有内容
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # 追加新的 result
    data.append(new_result)

    # 写回文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if control ==3:
    mr_LLM = Matching_MR_new(cuda_type=1)
    #for num in range(5):
    r=0
    for Num in range(4,5):
        for i in range(4):
            cavpath = "Save/" + str(Num) + "/" + csvpaths[i]
            resize_from_video_to_video2(savedirs[i]+"/"+str(Num))
            if i == 0:
                print("___________________________________", cavpath, "___________________________________")

                result1,result2,csv_path,valid_flags = Eval_results_3(cavpath, savedirs[i]+"/"+str(Num),mr_LLM, ranges=r)
            else:
                print("___________________________________", cavpath, "___________________________________")
                result1,result2,csv_path,valid_flags = Eval_results_3(cavpath, savedirs[i]+"/"+str(Num),mr_LLM, type=2,ranges=r)

            df = pd.DataFrame({
                "csv_path": csv_path,
                "valid_flag_diff": valid_flags
            })
            #df.to_csv(savedirs[i]+"/"+str(Num)+"/"+"Validition.csv", index=False)
            append_result_to_json({
                "type": "California",  # California 模型
                "round": Num,
                "model_id": i,
                "result": result1
            })

            append_result_to_json({
                "type": "Germany",  # Germany 模型
                "round": Num,
                "model_id": i,
                "result": result2
            })


    #valid_matrix = torch.stack([v0, v1, v2, v3], dim=1)

    # 同时为 1 的行（即所有模型都 valid）
    #common_valid = (valid_matrix.sum(dim=1) == 4).nonzero(as_tuple=True)[0]

   # print("所有模型都 valid 的编号：", common_valid.tolist())

    """
    mr_LLM = Matching_MR(cuda_type=1)
    #i=0
    for Num in range(1):

        r=0
        for i in range(4):
            i= 1
            Temp = savedirs[i] #+ "/" + str(Num)
            resize_from_video_to_video2(Temp)
            if i == 0:
                print("___________________________________",Temp,"___________________________________")
                #Eval_results(csvpaths[i], Temp, mr_LLM,ranges=r)
                Eval_results_3(csvpaths[i], Temp, mr_LLM,ranges=r)
            else:
                print("___________________________________", Temp, "___________________________________")
                Eval_results_3(csvpaths[i], Temp, mr_LLM, type=2,ranges=r)
                #print("___________________________________", savedirs[i], "___________________________________")
                #Eval_results(csvpaths[i],  savedirs[i], mr_LLM, type=2, ranges=r)
               # Eval_results_1(csvpaths[i], Temp, mr_LLM, type=2,ranges=r)
"""




def resize_from_video_to_video1(savedirs):
    src_dir = os.path.join(savedirs, "video")
    out_dir = os.path.join(savedirs, "video_1")
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".png"):
            img_path = os.path.join(src_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, (320, 160))
                cv2.imwrite(os.path.join(out_dir, f), resized)
def resize_from_video_to_videoB(savedirs, brightness_factor=1.1):
    src_dir = os.path.join(savedirs, "video")
    out_dir = os.path.join(savedirs, "video_1")
    os.makedirs(out_dir, exist_ok=True)
    num =0
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")
    image_files = matched_data["Image File"]
    for i in range(len(image_files)):
        if i<1670:
            continue
        base = os.path.basename(image_files[i].replace("\\", "/"))

        img_path = os.path.join(src_dir, base)
        img = cv2.imread(img_path)
        if img is not None:
            # 调整亮度
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
            # 调整大小
            resized = cv2.resize(img, (320, 160))
            cv2.imwrite(os.path.join(out_dir, base), resized)
#resize_from_video_to_videoB(savedirs[0] + "/" + str(0),brightness_factor=0.875)
#resize_from_video_to_videoB(savedirs[0] + "/" + str(1),brightness_factor=0.875)
if control == 4:
    mr_LLM = 0
    r = 0
    for Num in range(2):
        for i in range(4):
            validation_csv_path = f"{savedirs[i]}/{Num}/Validition.csv"
            cavpath = "Save/" + str(Num) + "/" + csvpaths[i]
            import warnings

            resize_from_video_to_video1(savedirs[i] + "/" + str(Num),)
            warnings.filterwarnings("ignore")  # 忽略所有 Python 警告
            if i == 0:
                #for k in range(10):
                    #brightness_factor = 2-k*0.2
                   # print(brightness_factor)

                print("___________________________________", cavpath, "___________________________________")

                total_1_stage, total_2_stage, total_1, total_2 = Eval_results_v(cavpath,
                                                                                savedirs[i] + "/" + str(
                                                                                    Num),
                                                                                validation_csv_path,
                                                                                ranges=r)
            else:
                #resize_from_video_to_video1(savedirs[i] + "/" + str(Num))
                print("___________________________________", cavpath, "___________________________________")
                total_1_stage, total_2_stage, total_1, total_2 = Eval_results_v(cavpath,
                                                                                savedirs[i] + "/" + str(
                                                                                    Num),
                                                                                validation_csv_path, type=2,
                                                                                ranges=r)

            result1 = {
                "type": "California",  # California 模型
                "round": Num,
                "model_id": i,
                "total_1": total_1_stage.tolist()
            }

            result2 = {
                "type": "Germany",  # Germany 模型
                "round": Num,
                "model_id": i,
                "total_1": total_1.tolist()
            }

            append_result_to_json(result1, json_path="result1.json")

            append_result_to_json(result2, json_path="result1.json")


    #valid_matrix = torch.stack([v0, v1, v2, v3], dim=1)

    # 同时为 1 的行（即所有模型都 valid）
    #common_valid = (valid_matrix.sum(dim=1) == 4).nonzero(as_tuple=True)[0]

   # print("所有模型都 valid 的编号：", common_valid.tolist())

    """
    mr_LLM = Matching_MR(cuda_type=1)
    #i=0
    for Num in range(1):

        r=0
        for i in range(4):
            i= 1
            Temp = savedirs[i] #+ "/" + str(Num)
            resize_from_video_to_video2(Temp)
            if i == 0:
                print("___________________________________",Temp,"___________________________________")
                #Eval_results(csvpaths[i], Temp, mr_LLM,ranges=r)
                Eval_results_3(csvpaths[i], Temp, mr_LLM,ranges=r)
            else:
                print("___________________________________", Temp, "___________________________________")
                Eval_results_3(csvpaths[i], Temp, mr_LLM, type=2,ranges=r)
                #print("___________________________________", savedirs[i], "___________________________________")
                #Eval_results(csvpaths[i],  savedirs[i], mr_LLM, type=2, ranges=r)
               # Eval_results_1(csvpaths[i], Temp, mr_LLM, type=2,ranges=r)
"""

#print(generate_random_mr(California_Components))