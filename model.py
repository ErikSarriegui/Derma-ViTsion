import torchvision
from torch import nn

def loadViT(
    out_features : int,
    weights : torchvision.models
):
  # Load the model
  vit = torchvision.models.vit_b_16(weights = weights)

  # Freeze the features extractor
  for block in vit.conv_proj.parameters():
    block.requires_grad = False

  for block in vit.encoder.parameters():
    block.requires_grad = False

  # Create de classifier layer
  vit.heads = nn.Sequential(
      nn.Linear(in_features=768, out_features = 256),
      nn.Hardswish(),
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features = 256, out_features = out_features, bias=True)
  )
  return vit