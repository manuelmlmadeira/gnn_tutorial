import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
from torch import nn, optim
from torch_geometric.datasets import GitHub
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric
import torch_geometric as pyg


def fetch_and_preprocess_data():
    # Download and process the dataset
    dataset = GitHub(".")
    data = dataset._data

    print("Design matrix")
    n_nodes, n_feats = data.x.shape
    print(f"Num. nodes: {n_nodes}; num features: {n_feats}")

    # Follow the original implementation
    data.y = 1 - data.y

    print("Target vector")
    print("First five elements:", data.y[:5])
    print("Number of samples:", data.y.shape[0])
    print("Number of nodes in class 1:", data.y.sum().item())
    print("Edge index shape:", data.edge_index.shape)
    print(
        "Edge index first 5 elements:",
    )
    data.edge_index[:, :5]

    # generate training set mask
    rng = torch.Generator().manual_seed(452)
    train_mask = torch.randn(n_nodes, generator=rng) < 0.8

    n_nodes_tr = train_mask.sum().item()
    print(f"Training set size: {n_nodes_tr} ({n_nodes_tr / n_nodes:.2%})")
    print(f"Test set size: {n_nodes - n_nodes_tr} ({1 - n_nodes_tr / n_nodes:.2%})")
    print(
        f"Ratio of class 1 in training: {torch.sum(train_mask * data.y).item() / n_nodes_tr:.2%}"
    )

    return data, train_mask


def rf_train_and_test(x_train, y_train, x_test, y_test):
    print("Initalizing the Random Forest Classifier!")

    print("Starting training...")
    rf_classifier = RandomForestClassifier().fit(x_train, y_train)

    print("Training complete! Starting testing...")
    y_rf = rf_classifier.predict(x_test)
    print("Testing complete!")
    report = classification_report(y_test, y_rf, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    return macro_f1


##### Deep Learning #####


class MLP(nn.Module):

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=1)

    def forward(self, x):
        x = self.linear1(x).relu()
        return self.linear2(x)


def train_nn_step(
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.train()  # Used to ensure that relevant blocks are in training mode

    optimizer.zero_grad()

    loss = loss_fn(model(x).squeeze(), y)
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_nn(model: nn.Module, loader: DataLoader, metric_fn, device) -> float:

    model.eval()  # Used to ensure that relevant block are in evaluation model

    metric_fn.reset()

    for x, y_true in loader:
        y_pred = model(x.to(device)).sigmoid().squeeze()
        metric_fn(y_pred, y_true.to(device))

    return metric_fn.compute().item()


def train_nn(
    loader_train,
    model,
    optimizer,
    loss_fn,
    n_epochs,
    device,
):
    loss_list = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for x, y in loader_train:
            loss = train_nn_step(
                optimizer, loss_fn, model, x.to(device), y.to(device).double()
            )

            epoch_losses.append(loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}, Av. Training Loss: {avg_loss:.4f}")

    plt.plot(range(1, n_epochs + 1), loss_list, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss per Epoch")
    plt.show()


#### Graph Deep Learning ####
def train_gnn_batched(loader_train, model, loss_fn, optimizer, n_epochs, device):
    epoch_loss_list = []
    # Your solution here #######################################################
    for epoch in range(n_epochs):
        batch_losses = []
        for batch in loader_train:
            batch = batch.to(device)
            batch.y = batch.y.to(device).double()
            loss = train_gnn_step(optimizer, loss_fn, model, batch, batch.y.double())
            batch_losses.append(loss)

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}: Average Loss = {avg_loss:.4f}")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return epoch_loss_list


def train_gnn_step(
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    model: nn.Module,
    data: pyg.data.Data,
    y: torch.Tensor,
) -> float:
    model.train()  # Used to ensure that relevant blocks are in training mode
    optimizer.zero_grad()
    loss = loss_fn(model(data.x, data.edge_index).squeeze(), y)
    loss.backward()
    optimizer.step()

    return loss.item()
