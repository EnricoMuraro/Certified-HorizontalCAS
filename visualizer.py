# imports Pil module
import PIL
import numpy as np
import onnx
import torch
from matplotlib import pyplot as plt
from onnx2pytorch import ConvertModel

import main


# filename = "TrainedNetworks/HCAS_TrainedNetwork_pra0_tau00.onnx"
# onnx_model = onnx.load(filename)
# pytorch_model = ConvertModel(onnx_model)

def test():

    rho = np.linspace(-0.5, 0.5, 1000)
    theta = np.linspace(-0.5, 0.5, 1000)

    xy = np.array(np.meshgrid(rho, theta)).T.reshape(-1, 2)
    psi = np.ones(1000000) * (-0.25)
    inputs = np.stack((xy[:, 0], xy[:, 1], psi), axis=-1)
    # Compute means, ranges
    # means = np.mean(inputs, axis=0)
    # ranges = np.max(inputs, axis=0) - np.min(inputs, axis=0)

    # Normalize each dimension of inputs to have 0 mean, unit range
    # If only one value, then range is 0. Just divide by 1 instead of range
    # ranges = np.where(ranges == 0.0, 1.0, ranges)
    # inputs = (inputs - means) / ranges

    print(inputs)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device='cuda:0')

    model = torch.load('best-model.pt')
    model.eval()
    outputs = model(inputs_tensor).cpu().detach().numpy()

    advisories = np.argmax(outputs, axis=1)
    advisories = advisories.reshape(1000, 1000)

    levels = [0, 1, 2, 3, 4, 5]

    x = np.linspace(0, 1000, 1000)
    y = np.linspace(0, 1000, 1000)
    h = plt.contourf(x, y, advisories, levels)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()


def test2():
    inputs, outputs = main.load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")

    inputs = inputs[inputs[:, 2] == -0.25]
    x = inputs[:, 0]*np.cos(inputs[:, 1])


    xy = [inputs[0]*np.cos(inputs[1]), inputs[0]*np.sin(inputs[1])]

    print(xy)

if __name__ == '__main__':
    test2()