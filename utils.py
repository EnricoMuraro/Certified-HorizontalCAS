import torch

INPUT_SIZE = 3


def export_to_ONNX(model, filename, device):
    model.eval()
    batch_size = 1  # just a random number
    x = torch.randn(batch_size, INPUT_SIZE, requires_grad=True, device=device)
    torch.onnx.export(model, x, filename, export_params=True, do_constant_folding=True)
