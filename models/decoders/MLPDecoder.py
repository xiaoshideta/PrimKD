import numpy as np
import torch.nn as nn
import torch
import time
from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
    
def load_dualpath_model(model, model_file):
# load raw state_dict
# t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        #raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file



    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('decode_head.linear') >= 0:
            state_dict[k] = v
            state_dict[k.replace('decode_head.linear', 'linear')] = v
    

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    # breakpoint()
    del state_dict

class DecoderHead_hint(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False,
                 losses='',
                 lambda_mgd=0.75):
        
        super(DecoderHead_hint, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.losses = losses
        self.in_channels = in_channels
        self.lambda_mask = lambda_mgd
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
         # 根据 losses 的内容动态创建 conv 层
        for loss_name in self.losses:
            setattr(self, f"align_{loss_name}", nn.Conv2d(eval(f"c{loss_name[-1]}_in_channels"), eval(f"c{loss_name[-1]}_in_channels"), kernel_size=1))
        # self.conv_c4 = nn.Conv2d(c4_in_channels, c4_in_channels, kernel_size=1)
        # self.conv_c3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1)
        # self.conv_c2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=1)
        # self.conv_c1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=1)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        outs = []
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        # 根据 losses 的内容动态使用 conv 层
        for loss_name in self.losses:
            conv_layer = getattr(self, f"align_{loss_name}")
            b = conv_layer(eval(f"c{loss_name[-1]}"))
            # print(b.shape)
            outs.append(b)

        return x, outs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

# t_end = time.time()
# logger.info(
#     "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
#         t_ioend - t_start, t_end - t_ioend))

class DecoderHead_mask(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.75,
                 losses=''):
        
        super(DecoderHead_mask, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.losses = losses
        self.in_channels = in_channels

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        # print("lambda_mask", self.lambda_mgd)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
         # 根据 losses 的内容动态创建 conv 层
        for loss_name in self.losses:
            # print(loss_name)
            setattr(self, f"align_{loss_name}", nn.Conv2d(eval(f"c{loss_name[-1]}_in_channels"), eval(f"c{loss_name[-1]}_in_channels"), kernel_size=1))
            setattr(self, f"conv_{loss_name}", nn.Sequential(
            nn.Conv2d(eval(f"c{loss_name[-1]}_in_channels"), eval(f"c{loss_name[-1]}_in_channels"), kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(eval(f"c{loss_name[-1]}_in_channels"), eval(f"c{loss_name[-1]}_in_channels"), kernel_size=3, padding=1)))
            # nn.Conv2d(eval(f"c{loss_name[-1]}_in_channels"), eval(f"c{loss_name[-1]}_in_channels"), kernel_size=1))
        
        # nn.Sequential(
        #     nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True), 
        #     nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))
        # self.conv_c4 = nn.Conv2d(c4_in_channels, c4_in_channels, kernel_size=1)
        # self.conv_c3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1)
        # self.conv_c2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=1)
        # self.conv_c1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=1)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        outs = []
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        # 根据 losses 的内容动态使用 conv 层
        for loss_name in self.losses:
            # breakpoint()
            align_layer = getattr(self, f"align_{loss_name}")
            conv_layer = getattr(self, f"conv_{loss_name}")

            preds_S = eval(f"c{loss_name[-1]}")
            N, C, H, W = preds_S.shape
            device = preds_S.device
            # align
            preds = align_layer(preds_S)

            mat = torch.rand((N,1,H,W)).to(device)
            mat = torch.where(mat>self.lambda_mgd, 0, 1).to(device)
            masked_fea = torch.mul(preds, mat)
            # print("masked_fea", masked_fea.shape)
            # print("masked_fea", masked_fea)
            # print("preds_S", preds_S)

            b = conv_layer(masked_fea)

            # b = conv_layer(eval(f"c{loss_name[-1]}"))
            # print(b.shape)
            outs.append(b)

        return x, outs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
