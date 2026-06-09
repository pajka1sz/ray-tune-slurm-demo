import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback

def train_cifar(config):
    # 1. Data preparation (CIFAR-10 will be downloaded automatically)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2
    )

    # 2. Load the model (ResNet18)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"]
    )

    # 3. Main training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # 4. Validation (evaluate how well the model is learning)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 5. Report metrics to Ray Tune (and automatically to WandB)
        train.report({
            "loss": val_loss / len(testloader),
            "accuracy": correct / total
        })

if __name__ == "__main__":
    # A. Define the Optuna hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "momentum": tune.uniform(0.8, 0.99)
    }

    # B. Configure logging to Weights & Biases
    wandb_callback = WandbLoggerCallback(
        project="ray-tune-slurm-cifar",
        # Make sure to set the WANDB_API_KEY environment variable before running!
        api_key="YOUR_WANDB_API_KEY"
    )

    # C. Launch Ray Tune with Optuna search
    tuner = tune.Tuner(
        # Resources allocated to a SINGLE trial:
        # 2 CPU cores and half of a GPU
        tune.with_resources(
            train_cifar,
            resources={"cpu": 2, "gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=10,  # Run 10 experiments with different hyperparameters
        ),
        param_space=search_space,
        run_config=train.RunConfig(
            callbacks=[wandb_callback]
        )
    )
    
    results = tuner.fit()
    print("Best hyperparameters:", results.get_best_result().config)