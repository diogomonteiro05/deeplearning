import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SimpleDeepFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.data = []
        
        # Load real images (label 0)
        for folder in os.listdir(real_dir):
            folder_path = os.path.join(real_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(folder_path, img_name)
                        self.data.append((img_path, 0))
        
        # Load fake images (label 1)
        for folder in os.listdir(fake_dir):
            folder_path = os.path.join(fake_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(folder_path, img_name)
                        self.data.append((img_path, 1))
        
        print(f"Found {sum(1 for x in self.data if x[1] == 0)} real images")
        print(f"Found {sum(1 for x in self.data if x[1] == 1)} fake images")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((128, 128))
        return image, label

# Create dataset with absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
real_path = os.path.join(current_dir, "data", "wiki")
fake_path = os.path.join(current_dir, "data", "inpainting")

dataset = SimpleDeepFakeDataset(real_path, fake_path)

print(f"Total images loaded: {len(dataset)}")

# Display first few samples
def display_samples(dataset, num_samples=6):
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        img, label = dataset[i]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"{'Real' if label == 0 else 'Fake'}")
        plt.axis('off')
    plt.show()

if len(dataset) > 0:
    display_samples(dataset)