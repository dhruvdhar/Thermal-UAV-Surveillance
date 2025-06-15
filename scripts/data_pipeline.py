# Data loading and training
import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from scripts.5_mgo_optimizer import MGO, evaluate

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

if os.path.exists("cropped_dataset/train"):
    train_ds = datasets.ImageFolder("cropped_dataset/train", transform=transform)

    val_size = int(0.2 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds_split, val_ds_split = random_split(train_ds, [train_size, val_size])
    train_dl_split = DataLoader(train_ds_split, batch_size=32, shuffle=True)
    val_dl_split = DataLoader(val_ds_split, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    best_params = MGO(train_dl_split, val_dl_split, device, population_size=5, max_iter=5)
    print("Best Parameters from MGO:")
    print(f"Dropout: {best_params[0]:.4f}, Filters: {int(best_params[1])}, LR: {best_params[2]:.6f}")

    final_acc = evaluate(best_params, train_dl_split, val_dl_split, device)
    print(f"Final Validation Accuracy: {final_acc*100:.2f}%")
