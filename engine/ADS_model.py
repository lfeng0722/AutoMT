import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import argparse
import os
#import matplotlib.image as mpimg
import numpy as np


def select_labels(pred_mode, labels):
    if pred_mode == "speed":
        return labels[:, 1:2]
    elif pred_mode == "steering":
        return labels[:, 0:1]
    else:
        raise ValueError(f"Unknown prediction mode: {pred_mode}")

def select_feature(dataset,Pred_mode):
    if dataset == "Udacity":
        input_feature = 2
    elif dataset == "a2d2":
        input_feature = 2
    if Pred_mode == "speed":
        output_feature = 1
    elif Pred_mode == "steering":
        output_feature = 1
    return  input_feature,output_feature
class CNN_LSTM(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=False,Pred_mode="speed"):
        super(CNN_LSTM, self).__init__()#这个模型可能要使用mse损失
        self.sequence_len = 4
        self.Pred_mode = Pred_mode
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.dropout = nn.Dropout(0.2)
        # 定义池化层
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        # 定义LSTM层
        self.Use_states = Use_states
        input_feature, output_feature = select_feature(dataset, Pred_mode)

        if Use_states == True:
            self.Encoder = nn.Sequential(
                nn.Linear(self.sequence_len * input_feature, 256),
                nn.ReLU(), )
            self.lstm = nn.LSTM(256, 256, batch_first=False)
            self.layer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ELU(),
                nn.Linear(128, 32),
                nn.ELU(),
            )
            self.de = nn.Linear(32, output_feature)
        else:
            self.lstm = nn.LSTM(256, 100, batch_first=False)
            self.layer = nn.Sequential(
                nn.Linear(100, 50),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.ELU(),
            )
            self.de = nn.Linear(10, output_feature)


    def forward(self, x,states,labels):
        # 对输入进行归一化
        lstm_inputs = []
        for t in range(self.sequence_len):
            xt = x[:, t, :, :, :]
            xt = self.conv(xt)
            xt = self.pool(xt)
            xt = self.dropout(xt)
            xt = torch.flatten(xt, 1)#torch.Size([32, 256])
            xt = xt.unsqueeze(0)
            lstm_inputs.append(xt)
        lstm_inputs = torch.cat(lstm_inputs,dim=0)
        output,_ = self.lstm(lstm_inputs) #torch.Size([160, 100])
        output = output[-1,:,:]
        #######################################是否输入过去的车辆状态，这些状态不输入lstm，而是通过一个NN来学习
        if self.Use_states == True:
            states = torch.flatten(states, 1)
            Prev_states = self.Encoder(states)
            output = torch.cat((output,Prev_states),dim=1)
        output = self.layer(output)
        output = self.de(output)
        if self.Pred_mode=="speed":
            labels = labels[:,-1, 1:2]
        elif self.Pred_mode == "steering":
            labels = labels[:,-1,0:1]
        return output,labels

class Weiss_CNN_LSTM(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=False,Pred_mode="speed"):
        super(Weiss_CNN_LSTM, self).__init__()
        self.sequence_len = 4
        self.Pred_mode = Pred_mode
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Sequential(
            nn.Linear(27456, 256),
            nn.ReLU())
        # 定义池化层
        # 定义LSTM层
        self.Use_states = Use_states
        input_feature, output_feature = select_feature(dataset, Pred_mode)

        if Use_states==True:
            self.Encoder = nn.Sequential(
                nn.Linear(self.sequence_len * input_feature, 256),
                nn.ReLU(), )
            self.lstm = nn.LSTM(256, 256, batch_first=False)
            self.layer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ELU(),
                nn.Linear(128, 32),
                nn.ELU(),
            )
            self.de = nn.Linear(32, output_feature)
        else:
            self.lstm = nn.LSTM(256, 100, batch_first=False)
            self.layer = nn.Sequential(
                nn.Linear(100, 50),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.ELU(),
            )
            self.de = nn.Linear(10, output_feature)



    def forward(self, x,states,labels):
        # 对输入进行归一化
        lstm_inputs = []
        for t in range(self.sequence_len):
            xt = x[:, t, :, :, :]
            xt = self.conv(xt)
            xt = self.dropout(xt)
            xt = torch.flatten(xt, 1)
            xt = self.linear(xt)#由于和原模型图片大小不一样，这里必须要线性层
            xt = xt.unsqueeze(0)
            lstm_inputs.append(xt)
        lstm_inputs = torch.cat(lstm_inputs,dim=0)
        output,_ = self.lstm(lstm_inputs)
        output = output[-1,:,:]

        #######################################是否输入过去的车辆状态，这些状态不输入lstm，而是通过一个NN来学习
        if self.Use_states == True:
            states = torch.flatten(states, 1)
            Prev_states = self.Encoder(states)
            output = torch.cat((output,Prev_states),dim=1)
        output = self.layer(output)

        output = self.de(output)
        if self.Pred_mode == "speed":
            labels = labels[:,-1, 1:2]
        elif self.Pred_mode == "steering":
            labels = labels[:, -1,0:1]
        return output,labels

class CNN_3D(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=False,Pred_mode="speed"):
        super(CNN_3D, self).__init__()
        self.sequence_len = 4
        self.Pred_mode = Pred_mode
        self.Use_states = Use_states
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(24),
            nn.ELU(),
            nn.Conv3d(24, 36, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(36),
            nn.ELU(),
            nn.Conv3d(36, 48, kernel_size=(2, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(48),
            nn.ELU(),
            nn.Conv3d(48, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
        self.pool = nn.AdaptiveAvgPool3d(3)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Sequential(
            nn.Linear(1728, 256),
            nn.ReLU())
        # 定义池化层
        # 定义LSTM层
        input_feature, output_feature = select_feature(dataset, Pred_mode)

        if Use_states==True:
            self.Encoder = nn.Sequential(
                nn.Linear(self.sequence_len * input_feature, 256),
                nn.ReLU(), )
            self.layer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ELU(),
                nn.Linear(128, 32),
                nn.ELU(),
            )
            self.de = nn.Linear(32, output_feature)
        else:
            self.layer = nn.Sequential(
                nn.Linear(256, 50),
                nn.ELU(),
                nn.Linear(50, 10),
                nn.ELU(),
            )
            self.de = nn.Linear(10, output_feature)


    def forward(self, x,states,labels):
        # 对输入进行归一化
        output = self.conv( x.permute(0, 2, 1, 3, 4))#B C D H W
        output = self.pool(output)
        output = self.dropout(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        #######################################是否输入过去的车辆状态，这些状态不输入lstm，而是通过一个NN来学习
        if self.Use_states == True:
            states = torch.flatten(states, 1)
            Prev_states = self.Encoder(states)
            output = torch.cat((output,Prev_states),dim=1)
        output = self.layer(output)

        output = self.de(output)
        if self.Pred_mode == "speed":
            labels = labels[:,-1, 1:2]
        elif self.Pred_mode == "steering":
            labels = labels[:,-1, 0:1]
        return output,labels




class PilotNet(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=False,Pred_mode="speed"):
        super(PilotNet, self).__init__()
        self.dataset = dataset
        self.Pred_mode = Pred_mode
        self.Use_states = Use_states
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.network_2 = nn.Sequential(
            nn.Linear(27456, 1200),
            nn.Linear(1200, 100),
            nn.Linear(100, 50))
        input_feature, output_feature = select_feature(dataset, Pred_mode)

        if Use_states==True:
            self.Encoder = nn.Sequential(
                nn.Linear(input_feature, 50),
                nn.ReLU(), )
            self.layer = nn.Sequential(
                nn.Linear(100, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
            )
            self.de = nn.Linear(32, output_feature)
        else:
            self.de = nn.Sequential(
                nn.Linear(50, 10),
                nn.Linear(10, output_feature),
            )


    def forward(self, x,states,labels):
        output = self.network(x)#torch.Size([32, 3, 160, 320])->torch.Size([32, 27456])
        output = self.network_2(output)
        if self.Use_states == True:
            states = torch.flatten(states, 1)
            Prev_states = self.Encoder(states)
            output = torch.cat((output,Prev_states),dim=1)
            output = self.layer(output)
        output = self.de(output)
        labels =select_labels(self.Pred_mode,labels)
        return output, labels
class EgoStatusMLPAgent(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=True,Pred_mode="speed"):
        super(EgoStatusMLPAgent, self).__init__()
        self.dataset = dataset
        self.Pred_mode = Pred_mode
        input_feature, output_feature = select_feature(dataset,Pred_mode)
        if Use_states==False:
            print("Must use vehicle state to get EgoStatusMLPAgent")
        hidden_layer_dim = 128
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_feature, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, output_feature),
        )

    def forward(self, x,states,labels):
        out = self.mlp(states)
        labels =select_labels(self.Pred_mode,labels)
        return out,labels


class Epoch(nn.Module):
    def __init__(self,dataset="Udacity",Use_states=True,Pred_mode="speed"):
        super(Epoch, self).__init__()
        self.dataset = dataset
        self.Pred_mode = Pred_mode
        self.Use_states = Use_states
        input_feature, output_feature = select_feature(dataset, Pred_mode)
        self.conv_layers = nn.Sequential(
            self.create_conv_layer(3, 32),
            self.create_conv_layer(32, 64),
            self.create_conv_layer(64, 128),
            self.create_conv_layer(128, 256)
        )
        self.layer = nn.Sequential(
            nn.Linear(51200, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if Use_states == True:
            self.Encoder = nn.Sequential(
                nn.Linear(input_feature, 256),
                nn.ReLU(),
            )
            self.de = nn.Linear(512, output_feature)
        else:
            self.de = nn.Linear(256, output_feature)

    def create_conv_layer(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x,states,labels):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.layer(out)
        if self.Use_states==True:
            Prev_states = self.Encoder(states)
            out = torch.cat((out,Prev_states),dim=1)
        out = self.de(out)
        labels =select_labels(self.Pred_mode,labels)
        return out,labels
class Resnet101(nn.Module):
    def __init__(self, pretrained=False,dataset="Udacity",Use_states=True,Pred_mode="speed"):
        super(Resnet101, self).__init__()
        self.dataset = dataset
        self.Pred_mode = Pred_mode
        self.Use_states = Use_states
        input_feature, output_feature = select_feature(dataset, Pred_mode)
        self.model = models.resnet101(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU()
        )
        if Use_states == True:
            self.Encoder = nn.Sequential(
                nn.Linear(input_feature, 256),
                nn.ReLU(),
            )
            self.de = nn.Linear(512, output_feature)
        else:
            self.de = nn.Linear(256, output_feature)

    def forward(self, x,states,labels):
        out = self.model(x)
        if self.Use_states == True:
            Prev_states = self.Encoder(states)
            out = torch.cat((out, Prev_states), dim=1)
        out = self.de(out)
        labels =select_labels(self.Pred_mode,labels)
        return out, labels

class Vgg16(nn.Module):
    def __init__(self, pretrained=False,dataset="Udacity",Use_states=True,Pred_mode="speed"):
        super(Vgg16, self).__init__()
        self.dataset = dataset
        self.Pred_mode = Pred_mode
        self.Use_states = Use_states
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512 * 7 * 7),
            nn.Linear(512 * 7 * 7, 256)
        )
        input_feature, output_feature = select_feature(dataset, Pred_mode)
        if Use_states == True:
            self.Encoder = nn.Sequential(
                nn.Linear(input_feature, 256),
                nn.ReLU(),
            )
            self.de = nn.Linear(512, output_feature)
        else:
            self.de = nn.Linear(256, output_feature)
    def forward(self, x,states,labels):
        out = self.model.features(x)#torch.Size([32, 3, 160, 320])->torch.Size([32, 512, 5, 10])
        out = self.conv_new(out)#torch.Size([32, 512, 5, 10])->torch.Size([32, 512, 5, 10])
        out = self.model.avgpool(out)#torch.Size([32, 512, 5, 10])->torch.Size([32, 512, 7, 7])
        out = torch.flatten(out, 1)#torch.Size([32, 25088])
        out = self.model.classifier(out)#torch.Size([32, 25088])-> torch.Size([32, 256])
        if self.Use_states == True:
            Prev_states = self.Encoder(states)
            out = torch.cat((out, Prev_states), dim=1)
        out = self.de(out)
        labels =select_labels(self.Pred_mode,labels)
        return out, labels


#https://github.com/sksq96/pytorch-vae/blob/master/vae.py
#and https://github.com/araffin/aae-train-donkeycar/blob/live-twitch-2/ae/autoencoder.py

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 2, 2)#128*144 8*16 2 72 4 36 8 18
class VAE(nn.Module):
    def __init__(self, device="cpu", image_channels=3, h_dim=1024, z_dim=256):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        esp = esp.to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x,states,labels):
        h = self.encoder(x)#torch.Size([32, 3, 160, 320])->torch.Size([32, 18432])
        z, mu, logvar = self.bottleneck(h)#torch.Size([32, 18432])->torch.Size([32, 256])X3
        z = self.fc3(z)#torch.Size([32, 18432])
        z_resize = self.decoder(z) #orch.Size([32, 3, 160, 320])
        return z_resize, mu, logvar,z



import torch
import torch.nn as nn

"""MODEL"""

import torch.nn as nn
import torch.nn.functional as F
import math


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.num_channels = config["num_channels"]
        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]

        self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        self.projection = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size,
                                    stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    # Patch Embeddings + (CLS Token + Positional Embeddings )
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.embed_dim = config["embed_dim"]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.embed_dim))
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embeddings[:, :x.shape[1], :]
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, attention_head_size, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.attention_head_size = attention_head_size
        self.bias = config["bias"]

        self.query = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)
        self.key = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)
        self.value = nn.Linear(self.embed_dim, self.attention_head_size, bias=self.bias)

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        attention_out = torch.matmul(attention_scores, v)

        return attention_out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.head_size = self.embed_dim // self.num_heads
        self.all_head_size = self.head_size * self.num_heads
        self.dropout = config["dropout"]
        self.qkv_bias = config["bias"]

        self.heads = nn.ModuleList([
            AttentionHead(
                self.head_size,
                config
            ) for _ in range(self.num_heads)
        ])

        self.attention_mlp = nn.Linear(self.all_head_size, self.embed_dim)
        self.out_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output in attention_outputs],
                                     dim=-1)  # concat attention for each head
        attention_output = self.attention_mlp(attention_output)
        attention_output = self.out_dropout(attention_output)

        return attention_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
        # self.act = nn.GELU()
        self.act = NewGELUActivation()
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.hidden_dim = config["hidden_dim"]
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        attention_output = self.attention(self.layer_norm1(x))
        x = x + attention_output
        mlp_out = self.mlp(self.layer_norm2(x))
        x = x + mlp_out
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x):
        all_attentions = []
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.embed_dim = config["embed_dim"]
        self.num_classes = config["num_classes"]

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoder_output = self.encoder(embedding_output)
        classification = self.classifier(encoder_output[:, 0, :])
        return classification

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.positional_embeddings.data = nn.init.trunc_normal_(
                module.positional_embeddings.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.positional_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.cls_token.dtype)


def train_ADS(args):
    if args.dataset == "udacity":
        dir = os.path.join(args.data_file, "ADS_data", "udacity")
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
    else:
        dir = os.path.join(args.data_file, "ADS_data", "A2D2")
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]