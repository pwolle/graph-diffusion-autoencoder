import time
import wandb
import random

import matplotlib.pyplot as plt

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="graph-diffusion-autoencoder_test",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "layers": 5,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# example plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("test plot")

# simulate training
epochs = wandb.config.epochs
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    # Wait 5 seconds
    # time.sleep(5)

# [optional] finish the wandb run, necessary in notebooks
image = wandb.Image(plt)
wandb.log({"test_plot": image})
wandb.finish()
