# imports Pil module
import math
from scipy.interpolate import griddata
import PIL
import numpy as np
import onnx
import torch
from matplotlib import pyplot as plt
from onnx2pytorch import ConvertModel

import main
import utils


# filename = "TrainedNetworks/HCAS_TrainedNetwork_pra0_tau00.onnx"
# onnx_model = onnx.load(filename)
# pytorch_model = ConvertModel(onnx_model)


def denormalize_data(inputs, outputs, psi):
    advisories = np.argmax(outputs, axis=1)

    rows = np.where(inputs[:, 2] == psi)
    inputs = inputs[rows]
    advisories = advisories[rows]

    x = []
    y = []
    for input in inputs:
        x.append((input[0]*56000+11424) * np.cos(input[1]*math.pi))
        y.append((input[0]*56000+11424) * np.sin(input[1]*math.pi))
    y = np.array(y)
    x = np.array(x)

    return x, y, advisories


def denormalized_model_output(model_filename, inputs, psi):

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device='cpu')
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    model.eval()
    outputs = model(inputs_tensor).cpu().detach().numpy()

    return denormalize_data(inputs, outputs, psi)


def test():

    rho = np.linspace(-0.2, 0.8, 1000)
    theta = np.linspace(-0.5, 0.5, 1000)

    xy = np.array(np.meshgrid(rho, theta)).T.reshape(-1, 2)
    psi = np.ones(1000000) * (-math.pi)
    inputs = np.stack((xy[:, 0], xy[:, 1], psi), axis=-1)

    print(inputs)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device='cuda:0')

    model = torch.load('best-model.pt')
    model.eval()
    outputs = model(inputs_tensor).cpu().detach().numpy()

    advisories = np.argmax(outputs, axis=1)
    advisories = advisories.reshape(1000, 1000)

    levels = [0, 1, 2, 3, 4, 5]

    x = (rho*56000 + 11424) * np.cos(theta*math.pi)
    y = (rho*56000 + 11424) * np.sin(theta*math.pi)

    fig, ax = plt.subplots()
    contour = ax.contourf(x, y, advisories, levels)
    legend1 = ax.legend(*contour.legend_elements(),
                        loc="best", title="Classes")
    ax.add_artist(legend1)
    plt.axis('scaled')
    plt.show()


def test2():
    inputs, outputs = main.load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")
    advisories = np.argmax(outputs, axis=1)
    rows = np.where(inputs[:, 2] == 0.5)
    inputs = inputs[rows]
    advisories = advisories[rows]

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device='cpu')
    model = torch.load('best-model.pt', map_location=torch.device('cpu'))
    model.eval()
    # outputs = model(inputs_tensor).cpu().detach().numpy()
    # advisories = np.argmax(outputs, axis=1)

    x = []
    y = []
    for input in inputs:
        x.append((input[0]*56000+11424) * np.cos(input[1]*math.pi))
        y.append((input[0]*56000+11424) * np.sin(input[1]*math.pi))

    y = np.array(y)
    x = np.array(x)

    crop = np.where((y < 15000) & (y > -15000))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    plt.tricontourf(x, y, advisories, levels=[0,1,2,3,4,5])
    plt.colorbar()

    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x, y, s=1, c=advisories)
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="best", title="Classes")
    # ax.add_artist(legend1)

    plt.axis('scaled')
    plt.show()
    print(x, y)


def contour_plot():
    inputs, outputs = main.load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")

    # x, y, advisories = denormalized_model_output('best-model-certified.pt', inputs, 0.5)
    x, y, advisories = denormalize_data(inputs, outputs, 0.5)

    x_lower = 0
    x_higher = 3000
    y_lower = -1500
    y_higher = 1500

    crop = np.where((y < y_higher) & (y > y_lower))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    crop = np.where((x < x_higher) & (x > x_lower))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    xi = np.linspace(x_lower, x_higher, 1000)
    yi = np.linspace(y_lower, y_higher, 1000)

    zi = griddata((x, y), advisories, (xi[None, :], yi[:, None]), method='nearest')

    fig, ax = plt.subplots()
    cs = ax.contourf(xi, yi, zi, 5)
    handles, labels = cs.legend_elements()
    ax.legend(handles, ["COC", "WL", "WR", "SL", "SR"])
    plt.show()


def scatter_plot():
    inputs, outputs = main.load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")

    x, y, advisories = denormalized_model_output('best-model-certified.pt', inputs, 0.5)
    # x, y, advisories = denormalize_data(inputs, outputs, 0.5)

    crop = np.where((y < 15000) & (y > -15000))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    fig, ax = plt.subplots()
    cs = ax.scatter(x, y, s=1, c=advisories)
    handles, labels = cs.legend_elements()
    ax.legend(handles, ["COC", "WL", "WR", "SL", "SR"])
    plt.show()


def contourf_all_points():
    rho = np.linspace(-0.2, 0.8, 1000)
    theta = np.linspace(-0.5, 0.5, 1000)

    xy = np.array(np.meshgrid(rho, theta)).T.reshape(-1, 2)
    psi = np.ones(1000000) * 0.5
    inputs = np.stack((xy[:, 0], xy[:, 1], psi), axis=-1)

    x, y, advisories = denormalized_model_output('best-model.pt', inputs, 0.5)

    xi = np.linspace(0, 50000, 1000)
    yi = np.linspace(-15000, 15000, 1000)

    zi = griddata((x, y), advisories, (xi[None, :], yi[:, None]), method='nearest')

    fig, ax = plt.subplots()
    cs = ax.contourf(xi, yi, zi, 5)
    handles, labels = cs.legend_elements()
    ax.legend(handles, ["COC", "WL", "WR", "SL", "SR"])
    plt.show()


if __name__ == '__main__':
    contour_plot()
    # scatter_plot()
    # contourf_all_points()

