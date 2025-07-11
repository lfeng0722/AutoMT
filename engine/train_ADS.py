
import os
import glob
#from skimage.io import imread
#from skimage.transform import resize
import json
import os
import re
import pandas as pd
import numpy
import torch
import engine.ADS_model as auto_models
from tqdm import tqdm
import itertools
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import ConcatDataset
def get_model(model_name, device,Use_states,Pred_mode):
    model_classes = {
        'Epoch': auto_models.Epoch,
        'Resnet101': auto_models.Resnet101,
        'Vgg16': auto_models.Vgg16,
        'EgoStatusMLPAgent':auto_models.EgoStatusMLPAgent,
        'PilotNet':auto_models.PilotNet,
        'CNN_LSTM': auto_models.CNN_LSTM,
        'Weiss_CNN_LSTM': auto_models.Weiss_CNN_LSTM,
        'CNN_3D': auto_models.CNN_3D
    }
    if model_name in model_classes:
        model = model_classes[model_name](Use_states=Use_states,Pred_mode=Pred_mode).to(device)
        return model
    else:
        raise ValueError(f"Model '{model_name}' not found.")

def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)

def pre_load(dataset, batch_size,shuffle):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_images = []
    all_states = []
    all_labels = []
    for images, states, labels in loader:
        all_images.append(images)
        all_states.append(states)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_states = torch.cat(all_states, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    new_dataset = TensorDataset(all_images, all_states, all_labels)

    loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

import torch.nn as nn
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        l1 = self.l1_loss(predictions, targets)
        mse = self.mse_loss(predictions, targets)
        return self.alpha * l1 + (1 - self.alpha) * mse


def trian_ADS(args,dataset,cuda):
    # 训练参数
    num_epochs = 30
    batch_size = 128
    # 定义损失函数和优化器
   # criterion = CombinedLoss(alpha=0.5) #torch.nn.L1Loss(reduction='sum')  # 使用平均绝对误差损失
    criterion = torch.nn.L1Loss(reduction='sum')
    # criterion = torch.nn.MSELoss()
    device =cuda
    root_dir = os.path.join("Data",dataset, "Save")
    if args.Use_time_series == False:
        train_dataset_1 = torch.load(os.path.join(root_dir, 'train.pt'),weights_only=0)
        val_dataset_1 = torch.load(os.path.join(root_dir, 'val.pt'),weights_only=0)

    if args.Use_time_series == True:
        train_dataset_1 = torch.load(os.path.join(root_dir, 'train_series.pt'),weights_only=0)
        val_dataset_1 = torch.load(os.path.join(root_dir, 'val_series.pt'),weights_only=0)
    train_dataset = train_dataset_1
    val_dataset = val_dataset_1
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#pre_load(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = pre_load(val_dataset, batch_size=batch_size, shuffle=False)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)#pre_load(val_dataset, batch_size=batch_size, shuffle=False)
    if args.Use_time_series == False:
        models = ["Resnet101", "Vgg16", "Epoch", "PilotNet"]  # "Epoch",
    else:
        models = ["CNN_LSTM",  "CNN_3D"]#"Weiss_CNN_LSTM",

    #dataset = args.dataset
    #use_state = args.Use_vehicle_states
    use_state =0
    pred_mode = args.pre_model
    for model_name in models:
        model = get_model(model_name, device,  use_state, pred_mode)
        torch.cuda.empty_cache()
        best_loss = float('inf')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f"models/{model_name}_{pred_mode}_{str(int(use_state))}")
        save_path = os.path.join(root_dir,f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
        #model.load_state_dict(torch.load(save_path))
        #Check_file(os.path.join("models"))
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            total_samples = 0
            for images, states, labels in train_loader:
                images = images.to(device)
                states = states.to(device)
                labels = labels.to(device)
                outputs, label_out = model(images, states, labels)
                loss = criterion(outputs, label_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_samples += labels.size(0)
            epoch_loss = running_loss /total_samples
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for images, states, labels in val_loader:
                    images = images.to(device)
                    states = states.to(device)
                    labels = labels.to(device)
                    outputs, label_out = model(images, states, labels)
                    loss = criterion(outputs, label_out)
                    val_loss += loss.item()
                    total_samples += labels.size(0)
            val_loss /= total_samples
            print(f"Validation Loss: {val_loss:.4f}")
            # 如果验证损失是目前最好的,就保存模型

            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)

def Train(args,dataset,cuda):
    use_time_series = [ 0,1]#1,
    pre_models = ["speed","steering"]#,]#["speed","steering"]#,"steering"]  # [0, 1] ,"steering" ,"steering"
    args.Use_vehicle_states = [0]
    for pre_model in pre_models:
        for time_series in use_time_series:
            args.Use_time_series = time_series
            args.pre_model = pre_model
            trian_ADS(args,dataset,cuda)

def Val(args,dataset,cuda):
    use_time_series = [ 0,1]#1,
    pre_models = ["speed","steering"]#,]#["speed","steering"]#,"steering"]  # [0, 1] ,"steering" ,"steering"
    args.Use_vehicle_states = [0]
    for pre_model in pre_models:
        for time_series in use_time_series:
            args.Use_time_series = time_series
            args.pre_model = pre_model
            test_ADS(args,dataset,cuda)


def evaluate_mae(model, test_loader, device):
    model.eval()
    total_absolute_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, states, labels in test_loader:
            images = images.to(device)
            states = states.to(device)
            labels = labels.to(device)
            outputs, label_out = model(images, states, labels)
            absolute_error = torch.sum(torch.abs(outputs - label_out)).item()
            total_absolute_error += absolute_error
            total_samples += labels.size(0)
    mean_mae = total_absolute_error / total_samples
    return mean_mae

def test_ADS(args, dataset, cuda):
    batch_size = 128
    device = cuda
    root_dir = os.path.join("Data", dataset, "Save")

    # 加载测试集（根据是否使用时间序列）
    if args.Use_time_series:
        test_dataset = torch.load(os.path.join(root_dir, 'test_series.pt'), weights_only=0)
        models = ["CNN_LSTM", "CNN_3D"]
    else:
        test_dataset = torch.load(os.path.join(root_dir, 'test.pt'), weights_only=0)
        models = ["Resnet101", "Vgg16", "Epoch", "PilotNet"]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    use_state = 0
    pred_mode = args.pre_model

    #print("\n=== Begin Testing ===")
    for model_name in models:
        model = get_model(model_name, device, use_state, pred_mode)
        save_path = os.path.join(root_dir, f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
        model.load_state_dict(torch.load(save_path))
        model.to(device)

        mae_score = evaluate_mae(model, test_loader, device)
        print(f"Dataset: {dataset},[{model_name}], Prediction:{args.pre_model} , Test MAE: {mae_score:.4f}")

"""
import os
import glob
#from skimage.io import imread
#from skimage.transform import resize
import json
import os
import re
import pandas as pd
import numpy
import torch
import engine.ADS_model as auto_models
from tqdm import tqdm
import itertools

def get_model(model_name, device,Use_states,Pred_mode):
    model_classes = {
        'Epoch': auto_models.Epoch,
        'Resnet101': auto_models.Resnet101,
        'Vgg16': auto_models.Vgg16,
        'EgoStatusMLPAgent':auto_models.EgoStatusMLPAgent,
        'PilotNet':auto_models.PilotNet,
        'CNN_LSTM': auto_models.CNN_LSTM,
        'Weiss_CNN_LSTM': auto_models.Weiss_CNN_LSTM,
        'CNN_3D': auto_models.CNN_3D
    }
    if model_name in model_classes:
        model = model_classes[model_name](Use_states=Use_states,Pred_mode=Pred_mode).to(device)
        return model
    else:
        raise ValueError(f"Model '{model_name}' not found.")

def Check_file(save_dir):
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
def trian_ADS(args):
    # 训练参数
    num_epochs = 70
    batch_size = 64
    # 定义损失函数和优化器
    criterion = torch.nn.L1Loss()  # 使用平均绝对误差损失
    # criterion = torch.nn.MSELoss()
    device = args.device
    if args.dataset == "udacity":
        save_dir = os.path.join(args.data_file, "ADS_data", "torch", "udacity")
    else:
        save_dir = os.path.join(args.data_file, "ADS_data", "torch", "A2D2")
    if args.Use_time_series == False:
        train_dataset = torch.load(os.path.join(save_dir, 'train.pt'))
        val_dataset = torch.load(os.path.join(save_dir, 'val.pt'))

    if args.Use_time_series == True:
        train_dataset = torch.load(os.path.join(save_dir, 'train_series.pt'))
        val_dataset = torch.load(os.path.join(save_dir, 'val_series.pt'))
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if args.Use_time_series == False:
        models = ["Resnet101", "Vgg16", "Epoch", "PilotNet"]  # "Epoch",
    else:
        models = ["CNN_LSTM", "Weiss_CNN_LSTM", "CNN_3D"]
    dataset = args.dataset
    use_state = args.Use_vehicle_states
    pred_mode = args.pre_model
    for model_name in models:
        best_loss = float('inf')
        model = get_model(model_name, device,  use_state, pred_mode)
        torch.cuda.empty_cache()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print(f"models/{model_name}_{dataset}_{pred_mode}_{str(int(use_state))}")
        save_path = os.path.join("models", dataset,f"{model_name}_{pred_mode}_{str(int(use_state))}.pth")
        Check_file(os.path.join("models", dataset))
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            for images, states, labels in train_loader:
                images = images.to(device)
                states = states.to(device)
                labels = labels.to(device)
                outputs, label_out = model(images, states, labels)
                loss = criterion(outputs, label_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            val_loss = 0.0

            with torch.no_grad():
                for images, states, labels in val_loader:
                    images = images.to(device)
                    states = states.to(device)
                    labels = labels.to(device)
                    outputs, label_out = model(images, states, labels)
                    loss = criterion(outputs, label_out)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            # 如果验证损失是目前最好的,就保存模型

            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)

def Train(args):
    datasets = ["A2D2"]#"udacity",
    use_time_series = [0,1]#1,
    pre_models = ["speed","steering"]  # [0, 1] ,"steering" ,"steering"
    args.Use_vehicle_states = 0
    for dataset in datasets:
        for pre_model in pre_models:
            for time_series in use_time_series:
                args.dataset = dataset
                args.Use_time_series = time_series
                args.pre_model = pre_model
                trian_ADS(args)

"""