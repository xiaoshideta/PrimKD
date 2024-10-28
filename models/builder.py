import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, load=True, decode_init=1, losses="loss1", lambda_mask=0.75):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead_mask
            self.decode_head = DecoderHead_mask(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim, losses=losses, lambda_mgd=lambda_mask)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion

        self.load = load
        if self.load:
            print(self.load)
            self.init_weights(cfg, pretrained=cfg.pretrained_model, decode_init=decode_init)
    
    # def init_weights(self, cfg, pretrained=None, pretrained1=None, decode_init=1):
    #     if pretrained:
    #         logger.info('Loading pretrained model: {}'.format(pretrained))
    #         if pretrained1:
    #             logger.info('Loading pretrained model: {}'.format(pretrained1))
    #             self.backbone.init_weights(pretrained=pretrained, pretrained1=pretrained1)
    #     # logger.info('Initing weights ...')
    #     logger.info('Initing weights ...')
    #     if decode_init:
    #         print("decode_init")
    #         self.decode_head.init_weights(pretrained=pretrained)
    #     else:
    #         init_weight(self.decode_head, nn.init.kaiming_normal_,
    #                 self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
    #                 mode='fan_in', nonlinearity='relu')
    #     if self.aux_head:
    #         init_weight(self.aux_head, nn.init.kaiming_normal_,
    #             self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
    #             mode='fan_in', nonlinearity='relu')

    def init_weights(self, cfg, pretrained=None, decode_init=1):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        # logger.info('Initing weights ...')
        logger.info('Initing weights ...')
        if decode_init:
            print("decode_init")
            self.decode_head.init_weights(pretrained=pretrained)
        else:
            init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out, out2 = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, out2, aux_fm
        return out, out2

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, out2, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out, out2= self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return out, out2, loss
        return out, out2



class EncoderDecoder2(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, load=True, decode_init=1):
        super(EncoderDecoder2, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        elif cfg.backbone == 'single_mit_b5':
            logger.info('Using backbone: sigle-Segformer-B5')
            from .encoders.single_segformer import single_mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'single_mit_b4':
            logger.info('Using backbone: sigle-Segformer-B4')
            from .encoders.single_segformer import single_mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'single_mit_b2':
            logger.info('Using backbone: sigle-Segformer-B2')
            from .encoders.single_segformer import single_mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'single_mit_b1':
            logger.info('Using backbone: sigle-Segformer-B1')
            from .encoders.single_segformer import single_mit_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'single_mit_b0':
            logger.info('Using backbone: sigle-Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.single_segformer import single_mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        else:
            logger.info('Using backbone: single-Segformer-B2')
            from .encoders.single_segformer import single_mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion

        self.load = load
        if self.load:
            print(self.load)
            self.init_weights(cfg, pretrained=cfg.pretrained_model1, decode_init=decode_init)
    
    def init_weights(self, cfg, pretrained=None, decode_init=0):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        if decode_init:
            print("decode_init")
            self.decode_head.init_weights(pretrained=pretrained)
        else:
            init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        if modal_x is not None:
            x = self.backbone(rgb, modal_x)
            out = self.decode_head.forward(x)
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, x, aux_fm
        else:
            x = self.backbone(rgb)
            out = self.decode_head.forward(x)
            out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
            if self.aux_head:
                aux_fm = self.aux_head(x[self.aux_index])
                aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
                return out, x, aux_fm
        return out, x

    # def encode_decode2(self, rgb):
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     orisize = rgb.shape
    #     x = self.backbone(rgb)
    #     out = self.decode_head.forward(x)
    #     out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
    #     if self.aux_head:
    #         aux_fm = self.aux_head(x[self.aux_index])
    #         aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
    #         return out, aux_fm
    #     return out

    def forward(self, rgb, modal_x, label=None):
        
        if self.aux_head:
            out, x, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out, x = self.encode_decode(rgb, modal_x)

        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return out, x, loss
        return out, x


class EncoderDecoder3(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, load=True, decode_init=1, losses="loss1", lambda_mask=0.75):
        super(EncoderDecoder3, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead_hint
            self.decode_head = DecoderHead_hint(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim, losses=losses, lambda_mgd=lambda_mask)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion

        self.load = load
        if self.load:
            print(self.load)
            self.init_weights(cfg, pretrained=cfg.pretrained_model, decode_init=decode_init)

    def init_weights(self, cfg, pretrained=None, decode_init=1):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        # logger.info('Initing weights ...')
        logger.info('Initing weights ...')
        if decode_init:
            print("decode_init")
            self.decode_head.init_weights(pretrained=pretrained)
        else:
            init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out, out2 = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, out2, aux_fm
        return out, out2

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, out2, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out, out2= self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return out, out2, loss
        return out, out2
