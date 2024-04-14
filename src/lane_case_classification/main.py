import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm


def train(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    classes = train_dataset.classes

    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-5)

    for epoch in range(args.epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, data in enumerate(pbar, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} Loss: {running_loss / (i+1):.3f}")

        if (epoch + 1) % args.checkpoint_epoch == 0:
            torch.save(net.state_dict(), os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint at epoch {epoch+1}")

    print("Finished Training")


def main():
    parser = argparse.ArgumentParser(description="Lane-case classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save trained models")
    parser.add_argument(
        "--checkpoint_epoch", type=int, default=5, help="Epoch interval to save checkpoints (default: 5)"
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
