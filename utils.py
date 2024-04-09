import math

import numpy as np
import torch

INPUT_SIZE = 3


def export_to_ONNX(model, filename, device='cpu'):
    model.to(device)
    model.eval()
    batch_size = 1  # just a random number
    x = torch.randn(batch_size, INPUT_SIZE, requires_grad=True, device=device)
    torch.onnx.export(model, x, filename, export_params=True, do_constant_folding=True)


def normalize_distance(distance):
    # ρ range: 56000  mean: 11424
    return (distance - 11424) / 56000


def normalize_angle(angle):
    # angle range: 2pi  mean: 0
    return angle / (math.pi * 2)


def to_grid(polar_ro, polar_theta):
    x = polar_ro * math.cos(polar_theta)
    y = polar_ro * math.sin(polar_theta)
    return x, y
