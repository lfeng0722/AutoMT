import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import glob
#from skimage.io import imread
#from skimage.transform import resize
import json
import tqdm
from openpyxl import Workbook
#import pandas as pd
import os
import re
#import pandas as pd
import numpy
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
import pandas as pd
def find_closest_value(array, target_timestep):
    closest_index = np.argmin(np.abs(array[:, 0] - target_timestep))
    return array[closest_index, 1]

def get_all_image_paths_A2D2(folder_path, type='*.png'):
    file_pattern = os.path.join(folder_path, type)  # 假设图片为 png 格式
    all_files = glob.glob(file_pattern)

    # 自定义排序函数
    def sort_key(file_path):
        # 使用正则表达式提取文件名中的数字部分
        match = re.search(r'\d+_camera_\w+_(\d+)\.png', file_path)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # 如果没有找到数字,将其放在最后

    # 使用自定义排序函数对文件路径进行排序
    image_paths = sorted(all_files, key=sort_key)

    return image_paths

def get_all_image_paths_Udacity(folder_path, type='*.png'):
    file_pattern = os.path.join(folder_path, type)  # 假设图片为 png 格式
    all_files = glob.glob(file_pattern)

    # 自定义排序函数
    def sort_key(file_path):
        # 提取文件名部分
        file_name = os.path.basename(file_path)
        # 去掉文件扩展名
        file_name_without_ext = os.path.splitext(file_name)[0]
        # 将文件名转换为整数
        try:
            return int(file_name_without_ext)
        except ValueError:
            return float('inf')  # 如果文件名无法转换为整数,将其放在最后

    # 使用自定义排序函数对文件路径进行排序
    image_paths = sorted(all_files, key=sort_key)

    return image_paths

def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
def process_Udacity_dataset(files,file_new):
    fig_file = os.path.join(files, "center")
    bus_file = files
    fig_file_new = os.path.join(file_new, "center")
    Check_file(fig_file_new)
    Source_images = get_all_image_paths_Udacity(fig_file)
    Source_images = Source_images[::2]  # 20fps - 10fps
    

    ################################################## 控制参数
    steerings = pd.read_csv(os.path.join(bus_file, "steering.csv"))
    matched_data = []
    for img_file in Source_images:
        try:
            img = cv2.imread(img_file)
            img_1 = Image.open(img_file)#左转是正 都是左转为正
            try:
                img_resized = img_1.resize((320, 160), resample=Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing or saving image {img_file}: {str(e)}")
                print(f"Deleting corrupted image: {img_file}")
                os.remove(img_file)
            if img is None:
                continue
            file_name = os.path.basename(img_file)
            new_img_path = os.path.join(fig_file_new, file_name)
            cv2.imwrite(new_img_path, img)
            file_num = int(os.path.splitext(os.path.basename(img_file))[0])
            closest_index = steerings.index[(steerings['timestamp'] - file_num).abs().idxmin()]
            steering = steerings.loc[closest_index, 'angle']* 9.425

            speed = steerings.loc[closest_index, 'speed']
            matched_data.append([file_num, new_img_path, steering, speed])
        except Exception as e:
            print(f"Error processing image {img_file}: {str(e)}")

    # 创建Excel工作簿和工作表
    wb = Workbook()
    ws = wb.active
    ws.title = "Matched Data"

    # 添加表头
    headers = ["Timestamp", "Image File", "Steering Angle", "Vehicle Speed"]
    ws.append(headers)

    # 添加匹配的数据
    for row in matched_data:
        ws.append(row)

    # 保存Excel文件
    excel_filename = os.path.join(file_new, 'matched_data.xlsx')
    wb.save(excel_filename)
    print(f"Matched data saved to {excel_filename}")

from tqdm import tqdm
def process_a2d2_dataset(files,file_new):
    fig_file = os.path.join(files, "cam_front_center")
    bus_file = os.path.join(files, "bus")
    Source_images = get_all_image_paths_A2D2(fig_file)
    Source_images = Source_images[::3]  # 30fps - 10fps
    bus_jsons = os.listdir(bus_file)
    bus_jsons = [file for file in bus_jsons if file.endswith(".json")]
    fig_file_new = os.path.join(file_new, "center")
    Check_file(fig_file_new)
    for bus_json in bus_jsons:
        file_path = os.path.join(bus_file, bus_json)
        with open(file_path, "r") as f:
            bus_data = json.load(f)
    acceleration_x = np.array([sublist[:2] for sublist in bus_data['acceleration_x']['values']])
    accelerator_pedal = np.array([sublist[:2] for sublist in bus_data['accelerator_pedal']['values']])
    angular_velocity_omega_z = np.array([sublist[:2] for sublist in bus_data['angular_velocity_omega_z']['values']])
    steering_angle_calculated = np.array([sublist[:2] for sublist in bus_data['steering_angle_calculated']['values']])
    brake_pressure = np.array([sublist[:2] for sublist in bus_data['brake_pressure']['values']])
    angular_velocity_omega_z[:, 1] = angular_velocity_omega_z[:, 1] / 180 * np.pi
    steering_angle_calculated[:, 1] = steering_angle_calculated[:, 1] / 180 * np.pi
    vehicle_speed = np.array([sublist[:2] for sublist in bus_data['vehicle_speed']['values']])
    vehicle_speed[:, 1] = vehicle_speed[:, 1] / 3.6
    matched_data = []


    for img_file in tqdm(Source_images):
        try:
            img = cv2.imread(img_file)
            img_1 = Image.open(img_file)
            try:
                img_resized = img_1.resize((320,160), resample=Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing or saving image {img_file}: {str(e)}")
                print(f"Deleting corrupted image: {img_file}")
                os.remove(img_file)
            if img is None:
                continue
            json_file = os.path.splitext(img_file)[0] + '.json'
            with open(json_file, "r") as f:
                data = json.load(f)

            file_name = os.path.basename(img_file)
            new_img_path = os.path.join(fig_file_new, file_name)
            cv2.imwrite(new_img_path, img)
            timestamp = data['cam_tstamp']
            acceleration_x_1 = find_closest_value(acceleration_x, timestamp)
            accelerator_pedal_1 = find_closest_value(accelerator_pedal, timestamp)
            angular_velocity_omega_z_1 = find_closest_value(angular_velocity_omega_z, timestamp)
            steering_angle_calculated_1 = find_closest_value(steering_angle_calculated, timestamp)
            vehicle_speed_1 = find_closest_value(vehicle_speed, timestamp)
            brake_pressure_1 = find_closest_value(brake_pressure, timestamp)
            matched_data.append([timestamp, img_file, steering_angle_calculated_1, vehicle_speed_1,angular_velocity_omega_z_1])
        except Exception as e:
            print(f"Error processing image {img_file}: {str(e)}")

    ##brake_pressure_1
    # 创建Excel工作簿和工作表
    wb = Workbook()
    ws = wb.active
    ws.title = "Matched Data"

    # 添加表头
    headers = ["Timestamp", "Image File", "Steering Angle", "Vehicle Speed"]
    ws.append(headers)
    for row in matched_data:
        if row[-1]<0:
            row[2]=row[2]*-1
        ws.append(row[:4])
    excel_filename = file_new + '/matched_data.xlsx'
    wb.save(excel_filename)
    print(f"Matched data saved to {excel_filename}")

def data_process(args):
    if args.data_process==False:
        return 0
    if args.dataset =="udacity":
        root_dir = os.path.join(args.data_file, "Raw", "udacity")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
        save_dir = os.path.join(args.data_file, "ADS_data", "udacity")
        for i in range(len(datasets)):
            file = os.path.join(root_dir, datasets[i])
            file_new = os.path.join(save_dir, datasets[i])
            process_Udacity_dataset(file,file_new)
    if args.dataset =="A2D2":
        root_dir = os.path.join(args.data_file, "Raw", "A2D2")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
        save_dir = os.path.join(args.data_file, "ADS_data", "A2D2")
        for i in range(len(datasets)):
            file = os.path.join(root_dir, datasets[i])
            file_new = os.path.join(save_dir, datasets[i])
            process_a2d2_dataset(file, file_new)

def resize_images(args,type="ORA"):
    if type=="ORA":
        if args.dataset == "udacity":
            dir = os.path.join(args.data_file, "ADS_data", "udacity")
            datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
            save_dir = os.path.join(args.data_file, "ADS_data", "torch", "udacity")

        else:
            dir = os.path.join(args.data_file, "ADS_data", "A2D2")
            datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
            save_dir = os.path.join(args.data_file, "ADS_data", "torch", "A2D2")

        Check_file(save_dir)
        check = False
        if check ==True:
            for i in range(len(datasets)):
                file_0 = os.path.join(dir, datasets[i], "center")
                file_1 = os.path.join(save_dir, datasets[i], "center")
                Check_file(file_1)
                if args.dataset == "udacity":
                    Source_images = get_all_image_paths_Udacity(file_0)
                else:
                    Source_images = get_all_image_paths_A2D2(file_0)

                for img_file in tqdm.tqdm(Source_images):
                    img_name = os.path.basename(img_file)
                    file_0_path = os.path.join(file_0, img_name)
                    file_1_path = os.path.join(file_1, img_name)

                    img = Image.open(file_0_path)
                    try:
                        img_resized = img.resize((320, 160), resample=Image.LANCZOS)
                        img_resized.save(file_1_path)
                    except Exception as e:
                        print(f"Error resizing or saving image {file_0_path}: {str(e)}")
                        print(f"Deleting corrupted image: {file_0_path}")

        for i in range(len(datasets)):
            file_1 = os.path.join(save_dir, datasets[i], "center")
            if args.dataset == "udacity":
                Source_images = get_all_image_paths_Udacity(file_1)
            else:
                Source_images = get_all_image_paths_A2D2(file_1)

            excel_path = os.path.join(dir,datasets[i], "matched_data.xlsx")
            df = pd.read_excel(excel_path)
            for idx, img_file in enumerate(Source_images):
                if idx < len(df):
                    df.loc[idx, 'Image File'] = img_file
            new_excel_path = os.path.join(save_dir, datasets[i], "matched_data.xlsx")
            df.to_excel(new_excel_path, index=False)
    if type=="ONE":
        if args.dataset == "udacity":
            dir = os.path.join(args.data_file, "ADS_data", "udacity", "OneFormer")
            datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
            save_dir = os.path.join(args.data_file, "ADS_data", "torch", "udacity", "OneFormer")

        else:
            dir = os.path.join(args.data_file, "ADS_data", "A2D2", "OneFormer")
            datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
            save_dir = os.path.join(args.data_file, "ADS_data", "torch", "A2D2", "OneFormer")
        Check_file(save_dir)
        for i in range(len(datasets)):
            file_0 = os.path.join(dir, datasets[i], "center")
            file_1 = os.path.join(save_dir, datasets[i], "center")
            Check_file(file_1)
            if args.dataset == "udacity":
                Source_images = get_all_image_paths_Udacity(file_0)
            else:
                Source_images = get_all_image_paths_A2D2(file_0)

            for img_file in Source_images:
                img_name = os.path.basename(img_file)
                file_0_path = os.path.join(file_0, img_name)
                file_1_path = os.path.join(file_1, img_name)

                img = Image.open(file_0_path)
                try:
                    img_resized = img.resize((320, 160), resample=Image.LANCZOS)
                    img_resized.save(file_1_path)
                except Exception as e:
                    print(f"Error resizing or saving image {file_0_path}: {str(e)}")
                    print(f"Deleting corrupted image: {file_0_path}")



check_and_create = lambda path: os.makedirs(path, exist_ok=True)
def prepare_data(args,Type="Udacity"):
    #datasets = ["udacity", "A2D2"]
    use_time_series = [0, 1]
    #for dataset in datasets:
    train_excel = os.path.join("Data",Type, "train_dataset.xlsx")
    val_excel = os.path.join("Data",Type, "val_dataset.xlsx")
    test_excel = os.path.join("Data",Type, "test_dataset.xlsx")
    train_file =  os.path.join("Data",Type,"train","data")
    val_file = os.path.join("Data",Type, "val", "data")
    test_file = os.path.join("Data",Type, "test", "data")
    root_dir =  os.path.join("Data",Type, "Save")
    check_and_create(root_dir)
    for time_series in use_time_series:
        args.Use_time_series =time_series
        if args.Use_time_series == False:
            train_dataset = Get_Dataset(train_file,train_excel)
            val_dataset = Get_Dataset(val_file, val_excel)
            test_dataset = Get_Dataset(test_file, test_excel)
        else:
            train_dataset = Get_Dataset_series(train_file, train_excel)
            val_dataset = Get_Dataset_series(val_file, val_excel)
            test_dataset = Get_Dataset_series(test_file, test_excel)

        if args.Use_time_series == False:
            torch.save(train_dataset, os.path.join(root_dir, 'train.pt'))
            torch.save(val_dataset, os.path.join(root_dir, 'val.pt'))
            torch.save(test_dataset, os.path.join(root_dir, 'test.pt'))
        if args.Use_time_series == True:
            torch.save(train_dataset, os.path.join(root_dir, 'train_series.pt'))
            torch.save(val_dataset, os.path.join(root_dir, 'val_series.pt'))
            torch.save(test_dataset, os.path.join(root_dir, 'test_series.pt'))

class Get_Dataset(Dataset):
    def __init__(self, data_dir,excel):
        self.data_dir = data_dir
        self.matched_data = pd.read_excel(excel, sheet_name="Sheet1")

    def __len__(self):
        return len(self.matched_data)-2+1

    def __getitem__(self, idx):
        matched = self.matched_data.iloc[idx:idx + 2]
        sequence = []
        prev_sequence = []
        cont = 0
        for _, row in matched.iterrows():
            data = [ row['Steering Angle'],row['Vehicle Speed']]

            if cont < 1:
                old_path = row["Image File"]
                filename = os.path.basename(old_path)
                #image = os.path.join(self.data_dir ,os.path.basename(row['Image File']))
                image = os.path.join(self.data_dir, os.path.basename(row['Image File']))
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                # 归一化到 [-1, 1]
                img = img.astype(numpy.float32) / 127.5 - 1.0
                prev_sequence = data
            else:
                sequence= data
            cont = cont + 1

        img_sequence = torch.from_numpy(numpy.array(img)).float().permute(2, 0, 1)
        sequence = torch.tensor(sequence).float()
        prev_sequence = torch.tensor(prev_sequence).float()
        return img_sequence, prev_sequence, sequence


class Get_Dataset_series(Dataset):
    def __init__(self, data_dir,excel):
        self.dir = data_dir
        self.matched_data = pd.read_excel(excel, sheet_name="Sheet1")

    def __len__(self):
        return len(self.matched_data) - 5 + 1

    def __getitem__(self, idx):
        matched = self.matched_data.iloc[idx:idx + 5]

        img_sequence = []
        sequence = []
        prev_sequence = []
        cont = 0
        for _, row in matched.iterrows():
            data = [ row['Steering Angle'],row['Vehicle Speed']]

            if cont < 4:
                image = os.path.join(self.dir, os.path.basename(row['Image File']))
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                # 归一化到 [-1, 1]
                img = img.astype(numpy.float32) / 127.5 - 1.0
                img_sequence.append(img)
                prev_sequence.append(data)
            else:
                sequence.append(data)
            cont = cont + 1

        img_sequence = torch.from_numpy(numpy.array(img_sequence)).float().permute(0, 3, 1, 2)
        sequence = torch.tensor(sequence).float()
        prev_sequence = torch.tensor(prev_sequence).float()
        return img_sequence, prev_sequence, sequence