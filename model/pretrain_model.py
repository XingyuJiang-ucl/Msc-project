import math

import timm
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer
from typing import Tuple, List

from torch.xpu import device

from helper import dotdict
import numpy as np
import torchmetrics
from model.tiny_vit import tiny_vit_21m_224


class CLIPLoss(torch.nn.Module):
    """
    Loss function for multimodal contrastive learning based off of the CLIP paper.

    Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
    similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
    Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
    """

    def __init__(self,
                 temperature: float,
                 lambda_0: float = 0.5) -> None:
        super(CLIPLoss, self).__init__()

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError('lambda_0 must be a float between 0 and 1.')
        self.lambda_0 = lambda_0
        self.lambda_1 = 1 - lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
        # normalize the embedding onto the unit hypersphere
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
        logits = torch.matmul(out0, out1.T) / self.temperature
        if indices is None:
            # labels = torch.arange(len(out0), device=out0.device)
            labels = torch.eye(len(out0),device = out0.device)
        else:
            # Convert the caseid list to tensor
            indices_tensor = torch.tensor(indices, device=out0.device)
            _, inverse_indices = torch.unique(indices_tensor, sorted=True, return_inverse=True)
            labels = inverse_indices
        loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
        loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
        loss = loss_0 + loss_1

        labels = torch.arange(len(out0), device=out0.device)

        return loss, logits, labels

class SiglipLoss(torch.nn.Module):
    """
    Loss function for multimodal contrastive learning based off of the SigLip paper.
    """

    def __init__(self) -> None:
        super(SiglipLoss, self).__init__()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias  = nn.Parameter(torch.tensor(-10.0))

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
        # normalize the embedding onto the unit hypersphere
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        logits = (torch.matmul(out0, out1.T) * self.logit_scale) + self.logit_bias

        # labels = torch.arange(len(out0), device=out0.device)
        labels = (2 * torch.eye(len(out0)) - torch.ones(len(out0))).to(out0.device)
        loss = - torch.sum(F.logsigmoid(labels * logits),dim=1).mean()

        labels = torch.arange(len(out0), device=out0.device)

        return loss, logits, labels


class SimCLRProjectionHead(nn.Module):
    """
    SimCLR Projection Head.

    According to the SimCLR paper, we construct an MLP with the following structure:
        - First layer: Linear layer -> [Optional BatchNorm1d] -> ReLU activation
        - Intermediate layers (if num_layers > 2): Linear layer -> [Optional BatchNorm1d] -> ReLU activation
        - Final layer: Linear layer -> [Optional BatchNorm1d] (no activation function)

    Args:
        input_dim: Input feature dimension (default 2048)
        hidden_dim: Hidden layer feature dimension (default 2048)
        output_dim: Output feature dimension (default 128)
        num_layers: Number of MLP layers (typically 2 for SimCLR v1, can be greater than 2 for SimCLR v2)
        batch_norm: Whether to use BatchNorm1d (default True)
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers = []
        # When using batch_norm, no bias is needed for the linear layer
        use_bias = not batch_norm

        # First block: Linear layer -> [BatchNorm1d] -> ReLU
        layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())

        # If there are more than 2 layers, add intermediate blocks: Linear layer -> [BatchNorm1d] -> ReLU
        for _ in range(2, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # Final block: Linear layer -> [BatchNorm1d], no activation function
        layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        self.arch = 'resnet50d'
        if cfg.arch is not None:
            self.arch = cfg.arch

        if cfg.arch != 'vision transformer':
            # out dim = [batch_size, 2048, x, x]  x vary with image shape
            self.decoder_image =  timm.create_model(self.arch, pretrained=pretrained,
                                                    in_chans=1, num_classes=0, global_pool='')
        else:
            self.decoder_image = tiny_vit_21m_224(pretrained=pretrained)

        self.decoder_tab = FTTransformer( # out dim = [2, d_block]
        n_cont_features= cfg.n_cont_features,
        cat_cardinalities=cfg.cat_cardinalities,
        d_out= None, # neglect final linear layer
        n_blocks=3,
        d_block=cfg.d_block,
        attention_n_heads=8,
        attention_dropout=0,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0,
        residual_dropout=0.0,
    )
        self.emb_dim = 512
        # paper code:
        self.projector_image = SimCLRProjectionHead(1000, self.emb_dim, 128)
        self.projector_tab = SimCLRProjectionHead(cfg.d_block, self.emb_dim, 128)

    def forward_image(self, batch):
        device = self.D.device
        # print(device)
        image = batch.to(device)
        emb_image = self.decoder_image(image)
        # print(f'image:{emb_image.shape}')
        if self.arch != 'vision transformer':
            emb_image = F.adaptive_avg_pool2d(emb_image, (1, 1))
            print(f'image after pool:{emb_image.shape}')
            emb_image = emb_image.view(emb_image.size(0), -1)
        return emb_image

    def forward_tab(self, batch):
        device = self.D.device
        # print(device)
        cont_tab = batch['cont_tab'].to(device)
        cat_tab = batch['cat_tab'].to(device)
        emb_tab = self.decoder_tab(cont_tab, cat_tab)
        return emb_tab

    def forward_Cliploss(self, emb_image,emb_tab,indices: List[int] = None):
        device = self.D.device
        pro_image = self.projector_image(emb_image)
        pro_tab = self.projector_tab(emb_tab.to(device))
        #lossfunction = CLIPLoss(temperature=0.1, lambda_0=0.5)
        lossfunction = SiglipLoss()
        loss,logits,labels =lossfunction(pro_image,pro_tab,indices)
        return loss,logits,labels

def run_check_net():
    def initialize_classifier_and_metrics(nclasses_train, nclasses_val,device):
        """
        Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
        """

        # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
        top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train).to(device)
        top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val).to(device)

        top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_train).to(device)
        top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_val).to(device)
        return top1_acc_train, top1_acc_val, top5_acc_train, top5_acc_val

    # 定义批次大小
    batch_size = 6

    # 连续型特征
    n_cont_features = 3
    x_cont = torch.randn(batch_size, n_cont_features)  # 生成随机连续特征

    # 类别型特征及其基数(可能的类别数)
    cat_cardinalities = [4, 7, 5]  # 各类别特征的可能取值数
    n_cat_features = len(cat_cardinalities)

    # 生成类别特征
    x_cat = torch.stack([torch.randint(0, c, (batch_size,)) for c in cat_cardinalities], dim=1)
    print(x_cat)

    # 断言确保数据正确性
    assert x_cat.dtype == torch.int64
    assert x_cat.shape == (batch_size, n_cat_features)

    # 独热编码类别特征
    x_cat_ohe = [F.one_hot(x_cat[:, i], num_classes=cat_cardinalities[i]) for i in range(n_cat_features)]

    # 将连续特征和独热编码的类别特征合并
    x = torch.cat([x_cont] + x_cat_ohe, dim=1)

    # 确认最终数据维度
    assert x.shape == (batch_size, n_cont_features + sum(cat_cardinalities))
    d_out = None  # For example, a single regression task.

    # tab_decoder = FTTransformer( # out dim = [2, 192]
    # n_cont_features=n_cont_features,
    # cat_cardinalities=cat_cardinalities,
    # d_out= None, # neglect final linear layer
    # n_blocks=3,
    # d_block=2048,
    # attention_n_heads=8,
    # attention_dropout=0.2,
    # ffn_d_hidden=None,
    # ffn_d_hidden_multiplier=4 / 3,
    # ffn_dropout=0.1,
    # residual_dropout=0.0,
    # )

    # o = tab_decoder(x_cont, x_cat)
    # print(o.shape)

    cfg = dotdict(
        n_cont_features = 3,
        cat_cardinalities=cat_cardinalities,
        arch = 'vision transformer',
        d_block = 1000
    )
    #image = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 1, 256, 256))).float()
    image = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 3, 224, 224))).float()
    batch = {
        'image': image,
        'cont_tab' : x_cont,
        'cat_tab' : x_cat
    }

    model = Net(True,cfg = cfg).to('cuda')
    zi = model.forward_image(image)
    zt = model.forward_tab(batch)
    # print(zi.shape, zt.shape)
    # loss = CLIPLoss(temperature= 0.5)
    lossv, logits, labels = model.forward_Cliploss(zi,zt)
    print(lossv)


    image = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 3, 224, 224))).float()
    # model = tiny_vit_21m_224(pretrained=True)
    # output = model(image)
    # print(output.shape)

    # device = 'cuda'
    # top1_acc_train, top1_acc_val, top5_acc_train, top5_acc_val = initialize_classifier_and_metrics(batch_size,
    #                                                                                                batch_size,device)
    # acctop1 = top1_acc_train(logits,labels)
    # acctop5 = top5_acc_train(logits,labels)
    # print(f"acctop1:{acctop1}acctop5:{acctop5}")

# main #################################################################
if __name__ == '__main__':
    run_check_net()