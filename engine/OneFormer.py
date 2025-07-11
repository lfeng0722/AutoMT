from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import engine.data_process as data_process
import cv2
import os
import matplotlib.pyplot as plt
import torch
from transformers import OneFormerProcessor, OneFormerModel
from transformers import AutoModelForCausalLM
import tqdm
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation,DPTImageProcessor
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import engine.Paramenters as Paramenters

# load Mask2Former fine-tuned on Cityscapes semantic segmentation

def OneFormer(args):
    check = True
    if args.dataset == "udacity":
        dir_1 = os.path.join(args.data_file, "Raw", "udacity")
        dir_2 = os.path.join(args.data_file, "ADS_data", "udacity")
        dir_3 = os.path.join(args.data_file, "ADS_data", "udacity", "OneFormer")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    else:
        dir_1 = os.path.join(args.data_file, "Raw", "A2D2")
        dir_2 = os.path.join(args.data_file, "ADS_data", "A2D2")
        dir_3 = os.path.join(args.data_file, "ADS_data", "A2D2", "OneFormer")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
    if args.device !="cuda":
        pass
    else:
        model = model.to("cuda")
    #model = AutoModelForCausalLM.from_pretrained("bert-base-uncased", device_map='cuda')

    #processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    #model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    for i in range(len(datasets)):
        if args.dataset == "udacity":
            file_1 = os.path.join(dir_1, datasets[i], "center")
        else:
            file_1 = os.path.join(dir_1, datasets[i], "cam_front_center")
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")

        data_process.Check_file(file_3)
        print(datasets[i])
        if args.dataset == "udacity":
            Source_images = data_process.get_all_image_paths_Udacity(file_2)
        else:
            Source_images = data_process.get_all_image_paths_A2D2(file_2)

        #pbar = tqdm.tqdm(total=len(Source_images))
        for img_file in tqdm.tqdm(Source_images):
            #pbar.update(100)
            img_name = os.path.basename(img_file)
            file_1_path = os.path.join(file_1, img_name)
            file_2_path = os.path.join(file_2, img_name)
            file_3_path = os.path.join(file_3, img_name)
            file_2_success = False
            file_3_success = False
            try:
                image_2 = Image.open(file_2_path)
                img_resized_2 = image_2.resize((320, 160), resample=Image.LANCZOS)
                file_2_success = True
            except Exception as e:
                pass
            if check == True:
                try:
                    image_3 = Image.open(file_3_path)
                    img_resized_3 = image_3.resize((320, 160), resample=Image.LANCZOS)
                    file_3_success = True
                except Exception as e:
                    pass
            else:
                file_3_success = False
            if file_2_success==True and file_3_success==False:
                image = Image.open(file_2_path)
                if args.device!="cuda":
                    with torch.no_grad():
                        semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                        outputs = model(**semantic_inputs)
                    # you can pass them to processor for semantic postprocessing
                    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                    predicted_semantic_map = predicted_semantic_map.numpy().astype("uint8")
                else:
                    with torch.no_grad():
                        semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                        semantic_inputs = semantic_inputs.to("cuda")
                        outputs = model(**semantic_inputs)
                    # you can pass them to processor for semantic postprocessing
                    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                    predicted_semantic_map = predicted_semantic_map.cpu().numpy().astype("uint8")
                cv2.imwrite(file_3_path, predicted_semantic_map)
            elif not (file_2_success and file_3_success):
                file_1_success =False
                try:
                    image_1 = Image.open(file_1_path)
                    img_resized = image_1.resize((320, 160), resample=Image.LANCZOS)
                    file_1_success = True
                    print(f"open image: {img_name}")
                except Exception as e:
                    print(f"Failed to open image: {img_name}")
                    continue
                if file_1_success==True:

                    image_1.save(file_2_path)
                    image = Image.open(file_2_path)
                    if args.device != "cuda":
                        with torch.no_grad():
                            semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                            outputs = model(**semantic_inputs)
                        # you can pass them to processor for semantic postprocessing
                        predicted_semantic_map = \
                        processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                        predicted_semantic_map = predicted_semantic_map.numpy().astype("uint8")
                    else:
                        with torch.no_grad():
                            semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                            semantic_inputs = semantic_inputs.to("cuda")
                            outputs = model(**semantic_inputs)
                        # you can pass them to processor for semantic postprocessing
                        predicted_semantic_map = \
                        processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                        predicted_semantic_map = predicted_semantic_map.cpu().numpy().astype("uint8")
                    cv2.imwrite(file_3_path, predicted_semantic_map)


def Check_OneFormer(args):
    if args.dataset == "udacity":
        dir_1 = os.path.join(args.data_file, "Raw", "udacity")
        dir_2 = os.path.join(args.data_file, "ADS_data", "udacity")
        dir_3 = os.path.join(args.data_file, "ADS_data", "udacity", "OneFormer")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    else:
        dir_1 = os.path.join(args.data_file, "Raw", "A2D2")
        dir_2 = os.path.join(args.data_file, "ADS_data", "A2D2")
        dir_3 = os.path.join(args.data_file, "ADS_data", "A2D2", "OneFormer")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
    if args.device !="cuda":
        pass
    else:
        model = model.to("cuda")
    for i in range(len(datasets)):
        if args.dataset == "udacity":
            file_1 = os.path.join(dir_1, datasets[i], "center")
        else:
            file_1 = os.path.join(dir_1, datasets[i], "cam_front_center")
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")
        data_process.Check_file(file_3)
        print(datasets[i])
        if args.dataset == "udacity":
            Source_images = data_process.get_all_image_paths_Udacity(file_2)
        else:
            Source_images = data_process.get_all_image_paths_A2D2(file_2)
        for img_file in tqdm.tqdm(Source_images):
            #pbar.update(100)
            img_name = os.path.basename(img_file)
            file_1_path = os.path.join(file_1, img_name)
            file_2_path = os.path.join(file_2, img_name)
            file_3_path = os.path.join(file_3, img_name)
            image = Image.open(file_2_path)
            with torch.no_grad():
                semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                semantic_inputs = semantic_inputs.to("cuda")
                outputs = model(**semantic_inputs)
            # you can pass them to processor for semantic postprocessing
            predicted_semantic_map = \
            processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_semantic_map = predicted_semantic_map.cpu().numpy().astype("uint8")
            cv2.imwrite(file_3_path, predicted_semantic_map)


def Check_OneFormer_resize(args):
    import warnings
    import logging

    logging.getLogger("natten.functional").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore")
    check = True
    if args.dataset == "udacity":
        #dir_1 = os.path.join(args.data_file, "Raw", "udacity")
        dir_2 = os.path.join(args.data_file, "ADS_data","torch", "udacity")
        dir_3 = os.path.join(args.data_file, "ADS_data", "torch","udacity", "OneFormer")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")

    else:
        #dir_1 = os.path.join(args.data_file, "Raw", "A2D2")
        dir_2 = os.path.join(args.data_file, "ADS_data","torch", "A2D2")
        dir_3 = os.path.join(args.data_file, "ADS_data","torch", "A2D2", "OneFormer")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")


    if args.device !="cuda":
        pass
    else:
        model = model.to("cuda")
    for i in range(len(datasets)):
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")


        data_process.Check_file(file_3)
        print(datasets[i])
        if args.dataset == "udacity":
            Source_images = data_process.get_all_image_paths_Udacity(file_2)
        else:
            Source_images = data_process.get_all_image_paths_A2D2(file_2)

        #pbar = tqdm.tqdm(total=len(Source_images))
        for img_file in tqdm.tqdm(Source_images):
            #pbar.update(100)
            img_name = os.path.basename(img_file)
            file_2_path = os.path.join(file_2, img_name)
            file_3_path = os.path.join(file_3, img_name)
            file_2_success = False
            #if os.path.exists(file_3_path):
            #    continue
            image = Image.open(file_2_path)
            with torch.no_grad():
                semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
                semantic_inputs = semantic_inputs.to("cuda")
                outputs = model(**semantic_inputs)
            # you can pass them to processor for semantic postprocessing
            predicted_semantic_map = \
            processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            predicted_semantic_map = predicted_semantic_map.cpu().numpy().astype("uint8")
            cv2.imwrite(file_3_path, predicted_semantic_map)

def Check_OneFormer_1(args):
    check = True
    if args.dataset == "udacity":
        dir_1 = os.path.join(args.data_file, "Raw", "udacity")
        dir_2 = os.path.join(args.data_file, "ADS_data", "udacity")
        dir_3 = os.path.join(args.data_file, "ADS_data", "udacity", "OneFormer")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
    else:
        dir_1 = os.path.join(args.data_file, "Raw", "A2D2")
        dir_2 = os.path.join(args.data_file, "ADS_data", "A2D2")
        dir_3 = os.path.join(args.data_file, "ADS_data", "A2D2", "OneFormer")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
    for i in range(len(datasets)):
        if args.dataset == "udacity":
            file_1 = os.path.join(dir_1, datasets[i], "center")
        else:
            file_1 = os.path.join(dir_1, datasets[i], "cam_front_center")
        file_2 = os.path.join(dir_2, datasets[i], "center")
        file_3 = os.path.join(dir_3, datasets[i], "center")
        data_process.Check_file(file_3)
        print(datasets[i])
        if args.dataset == "udacity":
            Source_images_2 = data_process.get_all_image_paths_Udacity(file_2)
            Source_images_3 = data_process.get_all_image_paths_Udacity(file_3)
        else:
            Source_images_2 = data_process.get_all_image_paths_A2D2(file_2)
            Source_images_3 = data_process.get_all_image_paths_Udacity(file_3)
        print(len(Source_images_2),len(Source_images_3))








if __name__ == "__main__":
    args = Paramenters.parse_args()
    args.seg_process = True
    args.dataset = "A2D2"
    OneFormer(args)