import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from filelock import FileLock

import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch

# 1. WandB import
from ray.air.integrations.wandb import setup_wandb

def train_cifar(config):
    wandb_run = setup_wandb(
        config,
        project="ray-tune-slurm-cifar",
        api_key=os.environ.get("WANDB_API_KEY", "YOUR_API_KEY") 
    )

    # 1. Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    scratch_dir = os.environ.get("SCRATCH", "./")
    data_dir = os.path.join(scratch_dir, "cifar_data")
    lock_path = os.path.join(scratch_dir, "cifar.lock")

    # Block: data can be downloaded and unpacked by only one process at the time
    with FileLock(lock_path):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=False, num_workers=2
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
        
        # 5. Report metrics
        metrics = {
            "loss": val_loss / len(testloader),
            "accuracy": correct / total
        }
        
        # Submit metrics to Ray Tune i and log them in WandB
        tune.report(**metrics) 
        wandb_run.log(metrics)

if __name__ == "__main__":
    ip_head = os.environ.get("ip_head")
    if ip_head:
        ray.init(address=ip_head)
    else:
        ray.init() 

    # A. Define the Optuna hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "momentum": tune.uniform(0.8, 0.99)
    }

    scratch_dir = os.environ.get("SCRATCH", "./")
    ray_results_dir = os.path.join(scratch_dir, "ray_results")

    # C. Launch Ray Tune with Optuna search
    tuner = tune.Tuner(
        tune.with_resources(
            train_cifar,
            resources={"cpu": 2, "gpu": 0} # 2 CPU per worker
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=10,
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            verbose=1,
            storage_path=ray_results_dir
        )
    )
    
    results = tuner.fit()
    print("Best hyperparameters:", results.get_best_result().config)
