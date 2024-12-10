#!/usr/bin/env python

# Deep Learning Homework 1

#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class LogisticRegression(nn.Module):
    def __init__(self, n_classes, n_features, **kwargs):
        """
        Initializes weights and bias for logistic regression.
        """
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)  # Linear layer Wx + b

    def forward(self, x, **kwargs):
        """
        Computes the logits y = Wx + b for a batch of inputs.
        """
        return self.linear(x)


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super().__init__()
        # Implement me!
        raise NotImplementedError

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        raise NotImplementedError


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    Trains the model on a batch of data.
    """
    optimizer.zero_grad()          # Clear previous gradients
    logits = model(X)              # Forward pass
    loss = criterion(logits, y)    # Compute loss
    loss.backward()                # Backward pass (gradient computation)
    optimizer.step()               # Update parameters
    return loss.item()             # Return the loss value


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    logits = model(X)
    loss = criterion(logits, y)
    loss = loss.item()
    y_hat = logits.argmax(dim=-1)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return loss, n_correct / n_possible


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=100, type=int,
                        help="Number of epochs to train for.")
    parser.add_argument('-batch_size', default=32, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=200)
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-learning_rate', type=float, default=0.002)
    parser.add_argument('-l2_decay', type=float, default=0.01)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-momentum', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 10
    n_feats = dataset.X.shape[1]

    # Initialize the model
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            opt.hidden_size,
            opt.layers,
            opt.activation,
            opt.dropout
        )

    # Get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # Get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    print('Initial val acc: {:.4f}'.format(evaluate(model, dev_X, dev_y, criterion)[1]))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('Train loss: {:.4f} | Val loss: {:.4f} | Val acc: {:.4f}'.format(
            epoch_train_loss, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # Plot
    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }
    plot(epochs, losses, filename=f'{opt.model}-training-loss.pdf')
    accuracy = {"Valid Accuracy": valid_accs}
    plot(epochs, accuracy, filename=f'{opt.model}-validation-accuracy.pdf')


if __name__ == '__main__':
    main()