import os

import torch
from maraboupy import Marabou, MarabouCore
from MarabouProperties import marabou_properties
import numpy as np
import onnx
import utils


def network_from_pytorch(model):
    utils.export_to_ONNX(model, "model-tmp.onnx")
    network = Marabou.read_onnx(f"model-tmp.onnx")

    if os.path.exists("model-tmp.onnx"):
        os.remove("model-tmp.onnx")

    return network


def network_from_file(model_filename):
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    utils.export_to_ONNX(model, f"{model_filename}-tmp.onnx")
    network = Marabou.read_onnx(f"{model_filename}-tmp.onnx")

    if os.path.exists(f"{model_filename}-tmp.onnx"):
        os.remove(f"{model_filename}-tmp.onnx")

    return network


def marabou_solve(model, property_number):
    network = network_from_pytorch(model)
    network = marabou_properties.add_constraints(network, property_number)

    options = Marabou.createOptions(numWorkers=8, verbosity=0)
    # SnC_options = Marabou.createOptions(snc=True, numWorkers=12, onlineSplits=2, initialSplits=4,
    #                                    initialTimeout=5, timeoutFactor=3, verbosity=0)

    sat, vals, stats = network.solve(options=options)
    # counterexample found
    sample_x, sample_y = [], []
    if sat == "sat":
        # vals.values() contains all the internal variable values for the counterexample
        counterexample_x = list(vals.values())[:3]
        counterexample_y = list(vals.values())[3:8]
        sample_x, sample_y = marabou_properties.generate_samples(counterexample_x,
                                                                 counterexample_y, property_number)
    return sample_x, sample_y


def marabou_solve_f(model_filename, property_number):
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    return marabou_solve(model, property_number)


if __name__ == "__main__":
    test_model = "Checkpoints/HCAS_TrainedNetwork_pra0_tau00.pt"
    new_x, new_y = marabou_solve_f(test_model, 1)
    print(new_x)
    print(new_y)
    """
    for filename in os.listdir("Checkpoints"):
        model_path = os.path.join("Checkpoints", filename)
        print(f"Solving {model_path}")
        marabou_solve(model_path, 1)
    """

