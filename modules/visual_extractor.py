import torch
import torch.nn as nn
import torchvision.models as models
from modules.vits import create_vit
from torchvision import models

class VisualExtractorEmbed(nn.Module):
    def __init__(self, args):
        super(VisualExtractorEmbed, self).__init__()

        # Visual extractor setup
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        
        if 'vit' in self.visual_extractor:
            self.model, self.feature_dim = self._setup_vit_model()
            self.projection_head = nn.Linear(768, 512, bias=False)
        else:
            self.model, self.feature_dim = self._setup_resnet_model()
            self.projection_head = nn.Linear(2048, 512, bias=False)

    def _setup_vit_model(self):
        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        image_size = 224

        vit_name = self.visual_extractor[4:]
        model, feature_dim = create_vit(
            vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        if self.pretrained:
            self._load_pretrained_vit_weights(model)

        return model, feature_dim

    def _load_pretrained_vit_weights(self, model):
        print('################ loading pretrained weights for vit')
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    def _setup_resnet_model(self):
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]  # Remove last FC layer
        feature_extractor = nn.Sequential(*modules)
        return feature_extractor, 2048  # ResNet50 output feature dimension

    def forward(self, images):
        if 'vit' in self.visual_extractor:
            img_feat = self.model(images)
            img_embeds = self.projection_head(img_feat[:, 0].contiguous())
            return img_feat[:, 1:].contiguous(), img_feat[:, 0].contiguous(), img_embeds
        else:
            patch_feats = self.model(images)
            avg_feats = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            img_embeds = self.projection_head(avg_feats)
            return patch_feats, avg_feats, img_embeds


class VisualExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(VisualExtractor, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)  # Flatten
args = {
    'visual_extractor': 'vit',  # Can be 'resnet50' or 'vit' (from your available options)
    'visual_extractor_pretrained': True,
}

model = VisualExtractorEmbed(args)
# Assume images is a tensor of shape [batch_size, 3, 224, 224]
images = torch.randn(32, 3, 224, 224)  # Example input
patch_feats, avg_feats, img_embeds = model(images)
