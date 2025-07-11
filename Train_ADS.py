import os

import torch.cuda

os.getcwd()
import engine.Paramenters as Paramenters
import os
import engine.train_ADS as trian_ADS
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import os
import pandas as pd
from PIL import Image

check_and_create = lambda path: os.makedirs(path, exist_ok=True)
def collect_datasets(Type="Udacity"):
    data_file = "Data"
    dir_0 = os.path.join(data_file, "ADS_data", "udacity")
    datasets_0 = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
    dir_1 = os.path.join(data_file, "ADS_data", "A2D2")
    datasets_1 = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
    all_train_data = []
    all_val_data = []
    all_test_data = []
    if Type=="Udacity":
        for dataset in datasets_0:
            data_dir = os.path.join(dir_0, dataset)
            matched_data = pd.read_excel(os.path.join(data_dir, "matched_data.xlsx"), sheet_name="Matched Data")

            total_samples = len(matched_data)
            train_size = int(total_samples * 0.8) - int(total_samples * 0.8) % 10
            val_size = int(total_samples * 0.1) - int(total_samples * 0.1) % 10
            test_size = int(total_samples * 0.1) - int(total_samples * 0.1) % 10

            if train_size + val_size + test_size > total_samples:
                test_size -= 10

            train_data = matched_data.iloc[:train_size]
            val_data = matched_data.iloc[train_size:train_size + val_size]
            test_data = matched_data.iloc[train_size + val_size:train_size + val_size + test_size]

            all_train_data.append(train_data)
            all_val_data.append(val_data)
            all_test_data.append(test_data)
    else:
        for dataset in datasets_1:
            data_dir = os.path.join(dir_1, dataset)
            matched_data = pd.read_excel(os.path.join(data_dir, "matched_data.xlsx"), sheet_name="Matched Data")

            total_samples = len(matched_data)
            train_size = int(total_samples * 0.8) - int(total_samples * 0.8) % 10
            val_size = int(total_samples * 0.1) - int(total_samples * 0.1) % 10
            test_size = int(total_samples * 0.1) - int(total_samples * 0.1) % 10

            if train_size + val_size + test_size > total_samples:
                test_size -= 10

            train_data = matched_data.iloc[:train_size]
            val_data = matched_data.iloc[train_size:train_size + val_size]
            test_data = matched_data.iloc[train_size + val_size:train_size + val_size + test_size]

            all_train_data.append(train_data)
            all_val_data.append(val_data)
            all_test_data.append(test_data)

    final_train_data = pd.concat(all_train_data, ignore_index=True)
    final_val_data = pd.concat(all_val_data, ignore_index=True)
    final_test_data = pd.concat(all_test_data, ignore_index=True)
    check_and_create(os.path.join(data_file,Type))
    final_train_data.to_excel(os.path.join(data_file,Type, "train_dataset.xlsx"), index=False)
    final_val_data.to_excel(os.path.join(data_file,Type,  "val_dataset.xlsx"), index=False)
    final_test_data.to_excel(os.path.join(data_file,Type,  "test_dataset.xlsx"), index=False)

    return final_train_data, final_val_data, final_test_data


def copy_images(Type="Udacity"):
    data_file = "Data"
    if Type!="Udacity":
        Type = "A2D2"

    # 创建目标文件夹
    os.makedirs(os.path.join(data_file,Type, "train", "data"), exist_ok=True)
    os.makedirs(os.path.join(data_file,Type, "val", "data"), exist_ok=True)
    os.makedirs(os.path.join(data_file,Type, "test", "data"), exist_ok=True)
    os.makedirs(os.path.join(data_file,Type, "test", "OneFormer"), exist_ok=True)
    os.makedirs(os.path.join(data_file,Type, "test", "RAW"), exist_ok=True)

    # 读取已经划分好的数据集
    train_data = pd.read_excel(os.path.join(data_file,Type, "train_dataset.xlsx"))
    val_data = pd.read_excel(os.path.join(data_file,Type, "val_dataset.xlsx"))
    test_data = pd.read_excel(os.path.join(data_file,Type, "test_dataset.xlsx"))

    # 复制并调整测试集图片大小，并复制OneFormer图片
    for _, row in test_data.iterrows():
        img_path = row["Image File"]
        img_filename = os.path.basename(img_path)

        # 复制并调整原始图片
        img = Image.open(img_path)
        img_resized = img.resize((320, 160), Image.LANCZOS)
        dst_img = os.path.join(data_file,Type, "test", "data", img_filename)
        img_resized.save(dst_img)



    # 复制并调整训练集图片大小
    for _, row in train_data.iterrows():
        img_path = row["Image File"]
        img_filename = os.path.basename(img_path)
        img = Image.open(img_path)
        img_resized = img.resize((320, 160), Image.LANCZOS)
        dst_img = os.path.join(data_file,Type, "train", "data", img_filename)
        img_resized.save(dst_img)
    for _, row in val_data.iterrows():
        img_path = row["Image File"]
        img_filename = os.path.basename(img_path)
        img = Image.open(img_path)
        img_resized = img.resize((320, 160), Image.LANCZOS)
        dst_img = os.path.join(data_file,Type, "val", "data", img_filename)
        img_resized.save(dst_img)
    for _, row in test_data.iterrows():
        img_path = row["Image File"]
        img_filename = os.path.basename(img_path)
        img = Image.open(img_path)
        dst_img = os.path.join(data_file,Type, "test", "RAW", img_filename)
        img.save(dst_img)


if __name__ == "__main__":
    datasets = ["Udacity","A2D2"]
    args = Paramenters.parse_args()
    data_process.data_process(args)
    cuda = "cuda:1"
    for dataset in datasets:
        collect_datasets(Type=dataset)
        copy_images(Type=dataset)

        data_process.prepare_data(args,dataset)
        #torch.cuda.empty_cache()
        trian_ADS.Train(args,dataset,cuda)
        trian_ADS.Val(args,dataset,cuda)
    # Test_datasets = ["A2D2", "udacity"]  # ["A2D2", "udacity"]
    # for dataset in Test_datasets:
    #    args.dataset = dataset
    #    OneFormer.Check_OneFormer(args) #生成道路分割
    
    
    

    #args.Use_time_series=1
    #args.data_process = True
    ##########################################
    #data_process.data_process(args) #下采样图像并且配对传感器数据
    #Test_datasets = ["A2D2", "udacity"]  # ["A2D2", "udacity"]
    #for dataset in Test_datasets:
    #    args.dataset = dataset
    #    OneFormer.Check_OneFormer(args) #生成道路分割
    #d#ata_process.resize_images(args,"ONE")#把图像变为 320 160 方便训练
    ## 加载为torch结构
    #trian_ADS.Train(args)#训练自动驾驶
    #1479424944177116069.png


    #LangSAM.Check_langsam(args)
    #OneFormer.Check_OneFormer_resize(args)  # 生成道路分割
    #############################################
    #Test_datasets = "A2D2"#["A2D2", "udacity"]
    #MT.find_test_images(args)
    #args.dataset =random.choice(Test_datasets)
    #MT.find_test_images(args)

    #Equality_MRs(args)
    #
    #Test_1(args,"diffusion",25) #add_object,add_object_1,Pix2Pix,diffusion
    #Test_1(args, "Pix2Pix", 1)
    #Test_ADS(args, type="Pix2Pix")
   # Test_ADS(args, type="diffusion")

