import argparse
import os
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from glob import glob
import shutil

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def predict(args):
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    # 画像ファイルのみをリストアップするフィルタリング処理を追加
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # サポートされる画像拡張子
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob(os.path.join(args.infer_dir, f"**/*{ext}"), recursive=True))

    dataset = CustomImageDataset(image_paths=image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    net = models.resnet18(weights=True)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, args.num_classes)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    with open(args.output_file, 'w') as f:
        with torch.no_grad():
            for idx, images in enumerate(loader):
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                f.write(f"{image_paths[idx]} {predicted[0].item()}\n")
                
                output_dir = os.path.join(args.output_dir, str(predicted[0].item()))
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(image_paths[idx])
                shutil.copyfile(image_paths[idx], os.path.join(output_dir, filename))
                
                print(f"copy {image_paths[idx]} to {os.path.join(output_dir, filename)}")
                

def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--infer_dir", type=str, required=True, help="Path to inference images folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output text file for inference results")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    predict(args)

if __name__ == "__main__":
    main()
    