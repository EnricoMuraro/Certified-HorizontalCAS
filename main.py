import copy
from multiprocessing import Pool

import marabou_solver
import utils
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# OPTIONS
EPOCHS = 1000
TRAINING_DATA_FOLDER = "TrainingData"
TRAINED_NETWORKS_FOLDER = "TrainedNetworks"
CHECKPOINTS_FOLDER = "Checkpoints"


def load_dataset(filename):
    f = h5py.File(filename, "r")

    X = f["X"][()]
    y = f["y"][()]

    return X, y


def neural_network():
    layer_size = 32
    network = nn.Sequential(
        (nn.Linear(3, 128)),
        (nn.ReLU()),
        (nn.Linear(128, 64)),
        (nn.ReLU()),
        (nn.Linear(64, 32)),
        (nn.ReLU()),
        (nn.Linear(32, 16)),
        (nn.ReLU()),
        (nn.Linear(16, 8)),
        (nn.ReLU()),
        (nn.Linear(8, 5)))
    return network


def accuracy(y_pred, y_true):
    # the largest value's index represents the output advisory
    advisory_pred = np.argmax(y_pred, axis=1)
    advisory_true = np.argmax(y_true, axis=1)

    # array that contains 1 if the prediction for the i-th example is correct, 0 otherwise
    # then average the values in the array to get the accuracy
    compared_outputs = [1 if advisory_pred[i] == advisory_true[i] else 0 for i in range(len(advisory_true))]
    return np.mean(compared_outputs)


def accuracy_of_network(network, dataloader):
    X, y_true = dataloader.dataset.tensors
    y_pred = network(X)
    y_pred = y_pred.numpy(force=True)
    y_true = y_true.numpy(force=True)
    accuracy(y_pred, y_true)


def custom_MSELoss(y_pred, y_true):
    loss_factor = 30.0
    num_outputs = 5

    difference = y_true - y_pred
    # the index of the highest value in y_true is the optimal advisory
    advisory_true = torch.argmax(y_true, dim=1)

    # matrix where there's a 1 in the index of the optimal advisory, 0 for the other 4 suboptimal advisories
    advisory_onehot = nn.functional.one_hot(advisory_true, num_outputs)

    # matrix where there's a 0 in the index of the optimal advisory, -1 for the other 4 suboptimal advisories
    others_onehot = advisory_onehot - 1

    # matrix that contains the differences between y_true and y_pred only for the optimal advisory
    d_optimal = difference * advisory_onehot
    # matrix that contains the differences between y_true and y_pred for the suboptimal advisories
    d_suboptimal = difference * others_onehot

    # penalized loss to be applied when the optimal advisory is underestimated
    penalized_optimal_loss = loss_factor * (num_outputs - 1) * (torch.square(d_optimal) + torch.abs(d_optimal))
    # normal loss to be applied when the optimal advisory is overestimated
    optimal_loss = torch.square(d_optimal)

    # penalized loss to be applied when a suboptimal advisory is overestimated
    penalized_suboptimal_loss = loss_factor * (torch.square(d_suboptimal) + torch.abs(d_suboptimal))
    # normal loss to be applied when a suboptimal advisory is underestimated
    suboptimal_loss = torch.square(d_suboptimal)

    # apply the losses
    optimal_advisory_loss = torch.where(d_optimal > 0, penalized_optimal_loss, optimal_loss)
    suboptimal_advisory_loss = torch.where(d_suboptimal > 0, penalized_suboptimal_loss, suboptimal_loss)

    # average out the combined losses
    mean_loss = torch.mean(optimal_advisory_loss + suboptimal_advisory_loss)
    return mean_loss


"""
def custom_MSELoss(y_pred, y_true):
    y_pred = y_pred.numpy(force=True)
    y_true = y_true.numpy(force=True)
    # optimal advisories for each row
    advisory_true = np.argmax(y_true, axis=1)

    rows, columns = y_true.shape
    loss = np.empty_like(y_pred)
    diff = y_pred-y_true
    penalization_factor = 30.0
    for i in range(rows):
        for j in range(columns):
            if j == advisory_true[i]:
                # the current element is the expected optimal advisory
                if diff[i][j] < 0:
                    # the optimal advisory is underestimated
                    # heavy penalization
                    loss[i][j] = penalization_factor*(rows-1)*np.square(diff[i][j])
                else:
                    # the optimal advisory is overestimated
                    # normal penalization
                    loss[i][j] = np.square(diff[i][j])
            else:
                # the current element is a suboptimal advisory
                if diff[i][j] < 0:
                    # the current suboptimal advisory is underestimated
                    # normal penalization
                    loss[i][j] = np.square(diff[i][j])
                else:
                    # the current suboptimal advisory is overestimated
                    # heavy penalization
                    loss[i][j] = penalization_factor*(rows-1)*np.square(diff[i][j])

    return np.mean(loss, axis=1)
"""


def train(network, dataloader, device="cpu"):
    network.to(device, non_blocking=True)

    optimizer = optim.Adam(network.parameters(), lr=0.001)
    best_acc = 0
    best_model = network

    print("training started")
    for epoch in range(EPOCHS):
        for X_batch, y_batch in dataloader:
            # loss_fn = nn.MSELoss()
            # loss_fn = nn.L1Loss()
            y_pred = network(X_batch)
            loss = custom_MSELoss(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = accuracy_of_network(network, dataloader)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(network)

        print(f"epoch {epoch}")
        print(f"best accuracy: {best_acc}")

    return best_model


def train_ACAS_network(pra_tau):
    previous_advisory, tau = pra_tau
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset_filename = f"{TRAINING_DATA_FOLDER}/HCAS_rect_TrainingData_v6_pra{previous_advisory}_tau{tau}.h5"
    trained_filename = f"{TRAINED_NETWORKS_FOLDER}/HCAS_TrainedNetwork_pra{previous_advisory}_tau{tau}.onnx"
    checkpoint_filename = f"{CHECKPOINTS_FOLDER}/HCAS_TrainedNetwork_pra{previous_advisory}_tau{tau}.pt"
    X, y = load_dataset(dataset_filename)

    X_train_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    network = neural_network()
    best_model = train(network, train_dataloader, device)

    # utils.export_to_ONNX(best_model, trained_filename, device)
    torch.save(best_model, checkpoint_filename)

    return best_model


def certify_model(model, X, y, p, max_attempts=100):
    X = X.tolist()
    y = y.tolist()
    attempts = max_attempts
    sample_x, sample_y = marabou_solver.marabou_solve(model, property_number=p)
    if not sample_x:
        print(f"Certified after {max_attempts-attempts} attempts")
        return model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    Inputs = X + sample_x
    Outputs = y + sample_y
    while attempts > 0:

        X_train_tensor = torch.tensor(Inputs, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(Outputs, dtype=torch.float32, device=device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            loss = custom_MSELoss(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sample_x, sample_y = marabou_solver.marabou_solve(model, property_number=p)

        if not sample_x:
            print(f"Certified after {max_attempts - attempts} attempts")
            return model

        Inputs += sample_x
        Outputs += sample_y
        attempts -= 1


if __name__ == '__main__':
    previous_advisories = ["0", "1", "2", "3", "4"]
    taus = ["00", "05", "10", "15", "20", "30", "40", "60"]

    combinations = [(pra, tau) for pra in previous_advisories for tau in taus]

    # model = train_ACAS_network(("0", "00"))
    model = torch.load("Checkpoints/HCAS_TrainedNetwork_pra0_tau00.pt", map_location=torch.device('cpu'))
    X, y = load_dataset("TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5")
    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_pred = model(X_tensor)
    y_pred = y_pred.numpy(force=True)
    acc_before = accuracy(y_pred, y)

    model = certify_model(model, X, y, 1)
    y_pred = model(X_tensor)
    y_pred = y_pred.numpy(force=True)
    acc_after = accuracy(y_pred, y)

    print(f"accuracy before certification: {acc_before}")
    print(f"accuracy after certification: {acc_after}")
    torch.save(model, 'best-model-certified.pt')

    # with Pool(12) as p:
    #     p.map(train_ACAS_network, combinations)

    # model = torch.load('best-model.pt')
