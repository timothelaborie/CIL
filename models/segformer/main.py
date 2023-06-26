# %%
import os
cwd = os.getcwd().replace("\\", "/") + "/models/segformer"
print(cwd)

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import requests
import wandb
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from glob import glob


BATCH_SIZE = 1

kaggle = True if cwd == "/kaggle/working" else False
data_path = "/kaggle/input/" if kaggle else cwd + "/../../data/"

# if kaggle:
#     main_path = data_path+"ethz-cil-road-segmentation-2023/"
#     pretrain_path = data_path+"massachusetts-roads-dataset/"
# else:
#     main_path = data_path+"official_roads/"
#     pretrain_path = data_path+"massachusetts_roads/"

# main_x_path = main_path + "training/images/"
# main_y_path =  main_path + "training/groundtruth/"

# pretrain_x_path =  pretrain_path + "tiff/train/"
# pretrain_y_path =  pretrain_path + "tiff/train_labels/"

#takes path of x and returns x and y as images
def get_label(x_path):
    if x_path.__contains__("massachusetts"):
        y_path = x_path.replace("tiff/train/", "tiff/train_labels/").replace(".tiff", ".tif")

    if x_path.__contains__("cil"):
        y_path = x_path.replace("images/", "groundtruth/")

    if x_path.__contains__("deepglobe"):
        y_path = x_path.replace("sat.jpg", "mask.png")

    return Image.open(x_path), Image.open(y_path)


def save(model, name):
    torch.save(model.state_dict(), ("/kaggle/working/" if kaggle else "") + name + ".pth")

def load(model, name):
    model.load_state_dict(torch.load(("/kaggle/input/" if kaggle else "") + name + ".pth"))
    model.eval()
    

# %%
# Load the model and setup the classifier head for binary classification
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model.decode_head.classifier = nn.Conv2d(768, 1, kernel_size=(1, 1), stride=(1, 1))
model = model.cuda()

# Instantiate the feature extractor
feature_extractor:SegformerImageProcessor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", size=800)

# Do a forward pass with random data to initialize the model
# x = torch.randn(1, 3, 800, 800).cuda()
# y = model(x).logits
# print(y.shape)


print("model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# %%
class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        x_orig, y_orig = get_label(self.image_files[idx])

        x_orig:Image = x_orig.convert("RGB")
        x_augmented = self.transform(x_orig) if self.transform else x_orig
        x = feature_extractor(images=x_augmented, return_tensors="pt").pixel_values.squeeze(0).cuda()

        y_orig:Image = y_orig.convert("RGB")
        #if the image is larger than 400x400, downscale it
        if y_orig.size[0] > 400 or y_orig.size[1] > 400:
            y_orig = y_orig.resize((400, 400))
        
        y_augmented = self.transform(y_orig) if self.transform else y_orig
        y = np.array(y_augmented, dtype=np.float32)/255
        y = torch.tensor(y, dtype=torch.float32)
        y = y[:, :, 0]
        y = y.unsqueeze(0).cuda()


        return x, y, self.image_files[idx], np.array(x_orig, dtype=np.float32)/255, np.array(y_orig, dtype=np.float32)/255, np.array(x_augmented, dtype=np.float32)/255, np.array(y_augmented, dtype=np.float32)/255


transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.autoaugment.AutoAugment(),
])

# %%
massachusetts_dataset = CustomDataset(glob(data_path + "massachusetts-roads-dataset/tiff/train/*.tiff"))
massachusetts_loader = DataLoader(massachusetts_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
deepglobe_dataset = CustomDataset(glob(data_path + "deepglobe-road-extraction-dataset/train/*.jpg"))
deepglobe_loader = DataLoader(deepglobe_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
main_dataset = CustomDataset(glob(data_path + "ethz-cil-road-segmentation-2023/training/images/*.png"))

# Split the dataset
val_size = int(len(main_dataset) * 0.2)
train_size = len(main_dataset) - val_size
torch.manual_seed(0)
train_dataset, val_dataset = random_split(main_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %%
def visualize_sample(model, loader):
    with torch.no_grad():
        rows = 2
        fig, ax = plt.subplots(rows, 5, figsize=(20, 30))
        for i, (x, y, name, x_orig, y_orig, x_augmented, y_augmented) in enumerate(loader):
            x = x[0]
            y = y[0]
            name = name[0]
            x_orig = x_orig[0]
            y_orig = y_orig[0]
            x_augmented = x_augmented[0]
            y_augmented = y_augmented[0]

            print(name)

            # print(pred.shape)
            # print(y.shape)
            # print(x.shape)

            pred = model(x.unsqueeze(0)).logits.squeeze(0)
            # print(pred.shape)

            pred = F.sigmoid(pred).permute(1, 2, 0).cpu().numpy()
            y = y.permute(1, 2, 0).cpu().numpy()
            x = x.permute(1, 2, 0).cpu().numpy()

            # print(pred.shape)
            # print(y.shape)
            # print(x.shape)

            # print(pred)
            # print(y)
            # print(x)

            ax[i][0].imshow(x_orig)
            ax[i][1].imshow(y_orig)
            ax[i][2].imshow(x_augmented)
            ax[i][3].imshow(y_augmented)
            ax[i][4].imshow(pred, cmap='gray')
            

            if i == rows - 1:
                break


# %%
visualize_sample(model, massachusetts_loader)
