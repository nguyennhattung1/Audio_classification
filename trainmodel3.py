import pandas as pd
import torch
import argparse
from pathlib import Path
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import SoundDS
from model import AudioClassifier
from model2 import Residual
from d2l import torch as d2l

# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
def prepara_data():
    path = "UrbanSound8K"
    data_path = path + "/metadata/UrbanSound8K.csv"
    # Read metadata file

    df = pd.read_csv(Path(data_path))

    # Construct file path by concatenating fold and file name
    df["relative_path"] = (
        "/fold" + df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
    )
    myds = SoundDS(df, path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)

    return (train_dl, val_dl)



def create_resnet():
    """
    Create a ResNet-like neural network model.

    """
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                layers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                layers.append(Residual(num_channels))
        return layers

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    model = nn.Sequential(b1, b2, b3, b4, b5,
                          nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten(), nn.Linear(512, 10))

    return model

# ----------------------------
# Training Loop
# ----------------------------
# Inside the training function
def training(train_dl,test_dl, num_epochs):

    net = create_resnet()

    lr = 0.05
    device = d2l.try_gpu()
    # Repeat for each epoch
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_dl)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_dl):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                test_acc = evaluate_accuracy_gpu(net, test_dl)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
        
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    torch.save(net.state_dict(), "net3.pt")
    print("Finished Training")



def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def inference(net, data_loader, device=None):
    """
    Evaluate the model on a given dataset.

    Args:
    - net (nn.Module): The trained neural network model.
    - data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - device (str or torch.device, optional): Device to perform evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
    - accuracy (float): Accuracy of the model on the dataset.
    """

    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    return accuracy
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True)
    args = vars(ap.parse_args())  # Corrected this line

    train_dl, test_dl = prepara_data()
    device = d2l.try_gpu()

    if args["mode"] == "train":
        # Run training model
        training(train_dl,test_dl, num_epochs=100)
    else:
        # Load trained/saved model
        net = create_resnet()
        net.load_state_dict(torch.load("net2.pt"))
        net.to(device)

        # Evaluate the model on the test set
        test_accuracy = inference(net, test_dl, device)
        print(f"Test Accuracy: {test_accuracy:.4f}")
