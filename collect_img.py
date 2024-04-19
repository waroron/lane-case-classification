import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import shutil
import argparse

class ImageFolderWithPaths(Dataset):
    """Custom dataset that includes image file paths. Extends the functionality for image loading."""
    def __init__(self, folder_path, transform=None):
        self.file_paths = []
        self.transform = transform
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path

def process_images(data_loader, model, target_class_index, save_base_path, gt_base_path, file_index):
    for images, paths in data_loader:
        with torch.no_grad():
            predictions = model(images)  # 'images' are already tensors
            class_indices = predictions.argmax(1)

        for image, path, class_index in zip(images, paths, class_indices):
            if class_index == target_class_index:
                new_img_name = f'{file_index:04d}.jpg'
                img_path = os.path.join(save_base_path, new_img_name)
                transforms.functional.to_pil_image(image).save(img_path)  # Save filtered image

                # Copy associated text and GT files if they exist
                txt_name = os.path.splitext(os.path.basename(path))[0] + '.lines.txt'
                txt_path = os.path.join(os.path.dirname(path), txt_name)
                if os.path.exists(txt_path):
                    new_txt_name = f'{file_index:04d}.lines.txt'
                    shutil.copy(txt_path, os.path.join(save_base_path, new_txt_name))
                
                gt_name = os.path.splitext(os.path.basename(path))[0] + '.png'
                gt_path = os.path.join(gt_base_path, os.path.relpath(os.path.dirname(path), start=os.path.split(gt_base_path)[0]), gt_name)
                if os.path.exists(gt_path):
                    new_gt_name = f'{file_index:04d}.png'
                    shutil.copy(gt_path, os.path.join(save_base_path, new_gt_name))

                file_index += 1
    return file_index


def main():
    parser = argparse.ArgumentParser(description="Process images with a model")
    parser.add_argument("--folder_paths", type=str, nargs='+', required=True, help="Paths to the folders containing images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--save_base_path", type=str, required=True, help="Base path for saving processed images")
    parser.add_argument("--gt_base_path", type=str, required=True, help="Base path for ground truth files")
    parser.add_argument("--target_class_index", type=int, required=True, help="Target class index to filter")
    args = parser.parse_args()

    # Set up the model and transformations
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 9)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    file_index = 0  # Initialize the file index
    for folder_path in args.folder_paths:
        dataset = ImageFolderWithPaths(folder_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
        file_index = process_images(data_loader, model, args.target_class_index, args.save_base_path, args.gt_base_path, file_index)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
