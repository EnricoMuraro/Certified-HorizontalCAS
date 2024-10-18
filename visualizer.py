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


def rect_outline(rho1, theta1, rho2, theta2):
    points_per_side = 100
    rhos = np.linspace(rho1, rho2, points_per_side)
    thetas = np.linspace(theta1, theta2, points_per_side)

    outline_theta = np.ones(points_per_side)*theta1
    outline_rho = rhos

    outline_theta = np.concatenate((outline_theta, np.ones(points_per_side)*theta2))
    outline_rho = np.concatenate((outline_rho, rhos))

    outline_theta = np.concatenate((outline_theta, thetas))
    outline_rho = np.concatenate((outline_rho, np.ones(points_per_side)*rho1))

    outline_theta = np.concatenate((outline_theta, thetas))
    outline_rho = np.concatenate((outline_rho, np.ones(points_per_side) * rho2))

    x = (outline_rho[i] * math.cos(outline_theta[i]) for i in range(len(outline_theta)))
    y = (outline_rho[i] * math.sin(outline_theta[i]) for i in range(len(outline_theta)))

    return np.array(list(x)), np.array(list(y))

def property_outline(p):
    if p == 1:
        return rect_outline(250, 0.2, 400, 0.4)
    if p == 2:
        return rect_outline(1500, -0.06, 1800, 0.06)
    if p == 3:
        return rect_outline(36000, 0.7, 60760, math.pi)
    if p == 4:
        return rect_outline(2000, -0.7, 5000, -math.pi)
    if p == 5:
        return rect_outline(43000, -1, 50000, 1)
    if p == 6:
        return rect_outline(250, -0.4, 400, -0.2)
    if p == 7:
        return rect_outline(36000, -math.pi, 60760, -0.7)
    if p == 8:
        return rect_outline(2000, 0.7, 5000, math.pi)
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


def crop_coords(x, y, advisories, x_lower, x_higher, y_lower, y_higher):
    crop = np.where((y < y_higher) & (y > y_lower))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    crop = np.where((x < x_higher) & (x > x_lower))
    x = x[crop]
    y = y[crop]
    advisories = advisories[crop]

    return x, y, advisories


def contour_plot(model_filename):
    inputs, outputs = main.load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")

    x, y, advisories = denormalized_model_output(model_filename, inputs, 0.5)
    # x, y, advisories = denormalize_data(inputs, outputs, 0.5)

    x_lower = 0
    x_higher = 2500
    y_lower = -1000
    y_higher = 1000
    x, y, advisories = crop_coords(x, y, advisories, x_lower, x_higher, y_lower, y_higher)
    xi = np.linspace(x_lower, x_higher, 1000)
    yi = np.linspace(y_lower, y_higher, 1000)

    zi = griddata((x, y), advisories, (xi[None, :], yi[:, None]), method='nearest')

    #rectangle outline
    outline_x, outline_y = property_outline(property_number)
    outline_x, outline_y, _ = crop_coords(outline_x, outline_y, [], x_lower, x_higher, y_lower, y_higher)

    fig, ax = plt.subplots()
    cs = ax.contourf(xi, yi, zi, 5)
    handles, labels = cs.legend_elements()
    ax.legend(handles, ["COC", "WL", "WR", "SL", "SR"])

    ax.scatter(outline_x, outline_y, color="r", s=1, alpha=0.8)
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


def contourf_all_points(model_filename):
    points = 1000
    rho = np.linspace(-0.2, 0.8, points)
    theta = np.linspace(-0.5, 0.5, points)
    xy = np.array(np.meshgrid(rho, theta)).T.reshape(-1, 2)
    psi = np.ones(points*points) * -0.5
    inputs = np.stack((xy[:, 0], xy[:, 1], psi), axis=-1)

    x, y, advisories = denormalized_model_output(model_filename, inputs, -0.5)
    x_lower = 0
    x_higher = 50000
    y_lower = -30000
    y_higher = 30000
    x, y, advisories = crop_coords(x, y, advisories, x_lower, x_higher, y_lower, y_higher)
    xi = np.linspace(x_lower, x_higher, points)
    yi = np.linspace(y_lower, y_higher, points)

    zi = griddata((x, y), advisories, (xi[None, :], yi[:, None]), method='nearest')

    #rectangle outline
    outline_x, outline_y = property_outline(property_number)
    outline_x, outline_y, _ = crop_coords(outline_x, outline_y, outline_y, x_lower, x_higher, y_lower, y_higher)

    unique_values = list(set(advisories))
    legend_labels = ["COC", "WL", "WR", "SL", "SR"]
    legend_labels = list(legend_labels[value] for value in unique_values)

    fig, ax = plt.subplots()
    cs = ax.contourf(xi, yi, zi, levels=len(unique_values))
    handles, labels = cs.legend_elements()
    legend_handles = list(handles[value] for value in unique_values)
    ax.legend(legend_handles, legend_labels)
    ax.scatter(outline_x, outline_y, color="r", s=1, alpha=0.5)
    plt.show()


property_number = 4
if __name__ == '__main__':
    pra = "2"
    tau = "00"
    # scatter_plot()
    model = f"Checkpoints/HCAS_TrainedNetwork_pra{pra}_tau{tau}.pt"
    model_certified = f"CertifiedNetworks/p{property_number}/HCAS_CertifiedNetwork_pra{pra}_tau{tau}_p{property_number}.pt"
    # contour_plot(model)
    contourf_all_points(model_certified)

