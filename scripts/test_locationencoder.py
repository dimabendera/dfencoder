"""
EXAMPLE RUN:
 python3 scripts/test_locationencoder.py
 https://github.com/MarcCoru/locationencoder
"""
from locationencoder import LocationEncoder

hparams = dict(
    legendre_polys=10,
    dim_hidden=64,
    num_layers=2,
    optimizer=dict(lr=1e-4, wd=1e-3),
    num_classes=1
)

# Pytorch Lightning Model
model = LocationEncoder("sphericalharmonics", "siren", hparams)

# input longitude latitude in degrees
import torch
lonlat = torch.tensor([[51.9, 5.6]], dtype=torch.float32)

# forward pass
output = model(lonlat)
