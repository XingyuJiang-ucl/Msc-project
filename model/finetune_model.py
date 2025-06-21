import timm
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer
from typing import Tuple, List
from helper import dotdict
import numpy as np
from model.tiny_vit import tiny_vit_21m_224, tiny_vit_11m_224
from model.pretrain_model import SimCLRProjectionHead

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
                                                    in_chans=3, num_classes=0, global_pool='')
        else:
            self.decoder_image = tiny_vit_11m_224(pretrained=pretrained)

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
        self.num_classes = cfg.num_classes
        in_dim = cfg.d_block + cfg.img_dim
        self.head = nn.Linear(in_dim, cfg.num_classes)
        if cfg.num_classes ==1:
            self.scale_factor = 7000
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward_image(self, batch):
        device = self.D.device
        # print(device)
        image = batch.to(device)
        emb_image = self.decoder_image(image)
        if self.arch != 'vision transformer':
            emb_image = F.adaptive_avg_pool2d(emb_image, (1, 1))
            # print(f'image after pool:{emb_image.shape}')
            emb_image = emb_image.view(emb_image.size(0), -1)
        return emb_image

    def forward_tab(self, batch):
        device = self.D.device
        cont_tab = batch['cont_tab'].to(device)
        cat_tab = batch['cat_tab'].to(device)
        emb_tab = self.decoder_tab(cont_tab, cat_tab)
        return emb_tab

    def forward_loss(self, emb_image,emb_tab,targets,output_type = 'loss'):
        device = self.D.device
        emb_tab = emb_tab.to(device)
        # print(emb_tab.shape,emb_image.shape)
        x = torch.cat([emb_image, emb_tab], dim=1)
        x = self.head(x)
        x = torch.squeeze(x)
        if output_type == 'loss':
            if self.num_classes ==1:
                loss = self.loss(x,targets) * self.scale_factor
            else:
                loss = self.loss(x, targets)
            return x, loss
        return x

def run_check_net():
    batch_size = 2

    # continuous features
    n_cont_features = 15
    x_cont = torch.randn(batch_size, n_cont_features)  # Generate random continuous features

    # category features
    cat_cardinalities = [3, 2, 2, 2, 2, 2, 4, 3, 4, 7, 2, 2, 4, 4, 5, 2, 2, 7, 4, 4, 14, 2, 5, 2, 5]
    n_cat_features = len(cat_cardinalities)
    x_cat = torch.stack([torch.randint(0, c, (batch_size,)) for c in cat_cardinalities], dim=1)
    print(x_cat)

    # One-hot encoding category features
    x_cat_ohe = [F.one_hot(x_cat[:, i], num_classes=cat_cardinalities[i]) for i in range(n_cat_features)]

    # Merge continuous features and one-hot category features
    x = torch.cat([x_cont] + x_cat_ohe, dim=1)

    assert x.shape == (batch_size, n_cont_features + sum(cat_cardinalities))

    cfg = dotdict(
        n_cont_features = n_cont_features,
        cat_cardinalities=cat_cardinalities,
        arch = 'resnet50d',
        d_block = 512,
        num_classes = 1,
        img_dim=2048  # vit 22M:576,vit 1M:448,resnet:2048
    )
    image = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 1, 256, 256))).float()
    batch = {
        'image': image,
        'cont_tab' : x_cont,
        'cat_tab' : x_cat
    }

    model = Net(False,cfg = cfg).to('cuda')
    zi = model.forward_image(image)
    zt = model.forward_tab(batch)
    # print(zi.shape, zt.shape)
    # loss = CLIPLoss(temperature= 0.5)
    labels = torch.tensor([10,20], dtype=torch.float, device='cuda')
    lossv= model.forward_loss(zi,zt, labels)
    print(lossv)

# main #################################################################
if __name__ == '__main__':
    run_check_net()