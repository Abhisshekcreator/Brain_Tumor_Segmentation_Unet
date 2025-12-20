# python train.py
import os
import glob
import time
import csv
import pandas as pd
import torch
import numpy as np
import wandb
from tqdm import tqdm
import nibabel as nib
import traceback 
import sys

# DEBUG: Resolve OMP: Error #15 (Deadlock on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, MetaTensor
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet 
from monai.transforms import (
    Compose, CropForegroundd, LoadImaged, NormalizeIntensityd,
    RandSpatialCropd, RandFlipd, RandRotate90d, ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd, AsDiscrete,  
    SpatialPadd, Activations, Lambda 
)
from monai.utils import set_determinism

# Initialize environment
torch.cuda.empty_cache()
print_config()

# %% [markdown]
# ### 1. Experiment Configuration
# %%
wandb.init(
    project="BraTS2020-Segmentation",
    entity="your_wandb_entity",
    name="unet_brain_tumor_baseline", 
    resume="allow"
)

config = wandb.config
config.seed = 2024
config.roi_size = (96, 96, 96)
config.infer_roi_size = (96, 96, 96)
config.batch_size = 1
config.learning_rate = 5e-4 
config.num_workers = 0 
config.save_interval = 5

# Path Configuration - Replace these with your local paths
config.base_results_dir = r"your_path/to/model_result"
config.experiment_name = "unet_brain_tumor" 
config.experiment_dir = os.path.join(config.base_results_dir, config.experiment_name)
config.csv_filename = os.path.join(config.experiment_dir, "training_log.csv")
config.nifti_output_dir = os.path.join(config.experiment_dir, "nifti_outputs")
config.best_model_path = os.path.join(config.experiment_dir, "best_metric_model.pth")

set_determinism(seed=config.seed)
os.makedirs(config.experiment_dir, exist_ok=True)
os.makedirs(config.nifti_output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### 2. Data Preparation
# %%
def format_dataset(base_dir, file_suffix=".nii"):
    """ Scans directories for BraTS formatted NIfTI files. """
    patient_folders = sorted([d for d in glob.glob(os.path.join(base_dir, "BraTS20_Training_*")) if os.path.isdir(d)])
    dataset_list = []
    
    for folder_path in tqdm(patient_folders, desc="Formatting dataset"):
        ctid = os.path.basename(folder_path)
        def get_nii_path(name):
            p1, p2 = os.path.join(folder_path, f"{name}{file_suffix}"), os.path.join(folder_path, f"{name}{file_suffix}.gz")
            return p1 if os.path.isfile(p1) else (p2 if os.path.isfile(p2) else None)

        input_paths = [get_nii_path(n) for n in [f"{ctid}_flair", f"{ctid}_t1", f"{ctid}_t1ce", f"{ctid}_t2"]]
        label_path = get_nii_path(f"{ctid}_seg")
        
        if all(input_paths) and label_path:
            dataset_list.append({"image": input_paths, "label": label_path, "name": ctid})
    return dataset_list

# Load splits created by format.py
train_base_dir = r"your_path/to/train_set"
val_base_dir = r"your_path/to/val_set"

train_files = format_dataset(train_base_dir)
val_files = format_dataset(val_base_dir)

# %% [markdown]
# ### 3. Pipeline & Transforms
# %%
# Transform description: Converts BraTS labels to 3 Multi-channels (TC, WT, ET)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size, mode="constant"),
    NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True), 
    RandSpatialCropd(keys=["image", "label"], roi_size=config.roi_size, random_size=False),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size, mode="constant"),
    NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)

# %% [markdown]
# ### 4. Network and Utilities
# %%
model = UNet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch",
).to(device)

loss_function = DiceLoss(to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5) 
dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

def reconstruct_brats_labels(tensor_3channel):
    """ Post-processing: Converts TC, WT, ET channels back to original BraTS labels (1, 2, 4) """
    tc, wt, et = tensor_3channel[0] > 0.5, tensor_3channel[1] > 0.5, tensor_3channel[2] > 0.5
    output = torch.zeros_like(et, dtype=torch.int8) 
    output[et] = 4
    output[(tc) & (~et)] = 1
    output[(wt) & (~tc)] = 2
    return output.unsqueeze(0)

post_pred_nifti_export = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), Lambda(reconstruct_brats_labels)])

def export_nifti_from_metatensor(metaTensor, outPath, dtype):
    arr_np = metaTensor.detach().cpu().numpy().squeeze().astype(dtype)
    affine = metaTensor.affine if isinstance(metaTensor, MetaTensor) else np.eye(4)
    ni_img = nib.Nifti1Image(arr_np, affine=affine)
    ni_img.header.set_data_dtype(dtype)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    nib.save(ni_img, outPath)

# %% [markdown]
# ### 5. Training Loop
# %%
best_metric, best_metric_epoch, epoch = -1, -1, 0

# Result logging setup
csv_file = open(config.csv_filename, 'a' if os.path.exists(config.csv_filename) else 'w', newline='')
csv_writer = csv.writer(csv_file)
if csv_file.tell() == 0:
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice_mean", "val_dice_tc", "val_dice_wt", "val_dice_et"])

try:
    while True: 
        epoch += 1
        model.train()
        epoch_loss = 0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)

        # Validation phase
        model.eval()
        dice_metric.reset()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_step, val_data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} Validation"), 1):
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # Sliding window inference for memory-efficient 3D segmentation
                val_outputs = sliding_window_inference(val_inputs, config.infer_roi_size, config.batch_size, model)
                epoch_val_loss += loss_function(val_outputs, val_labels).item()

                dice_metric(y_pred=[AsDiscrete(sigmoid=True, threshold=0.5)(i) for i in decollate_batch(val_outputs)], 
                            y=decollate_batch(val_labels))
                
                # Export segmentations for visual inspection
                if epoch % config.save_interval == 0:
                    case_name = val_data["name"][0]
                    reconstructed = post_pred_nifti_export(val_outputs[0]) 
                    out_path = os.path.join(config.nifti_output_dir, f"epoch{epoch}", f"{case_name}_seg.nii.gz")
                    export_nifti_from_metatensor(reconstructed, out_path, np.uint8)

        epoch_val_loss /= len(val_loader)
        metric_ch = dice_metric.aggregate()
        m_mean, m_tc, m_wt, m_et = metric_ch.mean().item(), metric_ch[0].item(), metric_ch[1].item(), metric_ch[2].item()
        dice_metric.reset()

        print(f"Epoch {epoch} Summary: Loss {epoch_loss:.4f} | Val Dice {m_mean:.4f}")
        wandb.log({"epoch": epoch, "train/loss": epoch_loss, "val/loss": epoch_val_loss, "val/dice_mean": m_mean})
        csv_writer.writerow([epoch, epoch_loss, epoch_val_loss, m_mean, m_tc, m_wt, m_et])
        csv_file.flush()

        if m_mean > best_metric:
            best_metric, best_metric_epoch = m_mean, epoch
            torch.save(model.state_dict(), config.best_model_path)
            print(f"*** New Best Model Saved: {best_metric:.4f} ***")

except KeyboardInterrupt:
    print("Training interrupted.")
finally:
    csv_file.close()
    wandb.finish()