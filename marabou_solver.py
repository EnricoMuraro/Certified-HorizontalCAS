import torch
from maraboupy import Marabou, MarabouCore
import numpy as np
import onnx


def marabou_solve(model_filename, property_filename):
    network = Marabou.read_onnx(model_filename)

    if P == 1:
        return property1.get_training_examples(network)

    return [], []


if __name__ == "__main__":
    pass
