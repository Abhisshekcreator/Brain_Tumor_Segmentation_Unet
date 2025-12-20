# python test.py
import os
import glob
import csv
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import sys

# DEBUG: Resolve OMP: Error #15 (Deadlock on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

from monai.data import DataLoader, Dataset, decollate_batch, MetaTensor
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet 
from monai.transforms import (
    Compose, CropForegroundd, LoadImaged, NormalizeIntensityd,
    ToTensord, ConvertToMultiChannelBasedOnBratsClassesd,
    AsDiscrete, SpatialPadd, Activations, Lambda 
)
from monai.utils import set_determinism

# Initialize device and clear cache
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ### 1. Test Configuration and Path Setup
# %%
config = {
    "seed": 2024,
    "roi_size": (96, 96, 96),       # Must match training patch size
    "infer_roi_size": (96, 96, 96), # Sliding window patch size
    "batch_size": 1,
    "base_results_dir": r"your_path/to/model_result",
    "experiment_name": "unet_brain_tumor",
    "test_base_dir": r"your_path/to/test_set" 
}

# Derived Paths
config["experiment_dir"] = os.path.join(config["base_results_dir"], config["experiment_name"])
config["best_model_path"] = os.path.join(config["experiment_dir"], "best_metric_model.pth")
config["test_output_dir"] = os.path.join(config["test_base_dir"], "output") 
config["test_results_csv"] = os.path.join(config["experiment_dir"], "test_results.csv")

set_determinism(seed=config["seed"])
os.makedirs(config["test_output_dir"], exist_ok=True)

# %% [markdown]
# ### 2. Utility Functions
# %%
def format_dataset(base_dir, file_suffix=".nii"):
    """ Scans directories for BraTS formatted NIfTI files. """
    patient_folders = sorted([d for d in glob.glob(os.path.join(base_dir, "BraTS20_Training_*")) if os.path.isdir(d)])
    dataset_list = []
    
    for folder_path in tqdm(patient_folders, desc="Scanning test cases"):
        ctid = os.path.basename(folder_path)
        def get_nii_path(name):
            p1, p2 = os.path.join(folder_path, f"{name}{file_suffix}"), os.path.join(folder_path, f"{name}{file_suffix}.gz")
            return p1 if os.path.isfile(p1) else (p2 if os.path.isfile(p2) else None)

        input_paths = [get_nii_path(n) for n in [f"{ctid}_flair", f"{ctid}_t1", f"{ctid}_t1ce", f"{ctid}_t2"]]
        label_path = get_nii_path(f"{ctid}_seg")
        
        if all(input_paths) and label_path:
            dataset_list.append({"image": input_paths, "label": label_path, "name": ctid})
    return dataset_list

def reconstruct_brats_labels(tensor_3channel):
    """ Converts 3-channel (TC, WT, ET) sigmoid output back to original BraTS labels (1, 2, 4). """
    tc, wt, et = tensor_3channel[0] > 0.5, tensor_3channel[1] > 0.5, tensor_3channel[2] > 0.5
    output = torch.zeros_like(et, dtype=torch.int8) 
    output[et] = 4
    output[(tc) & (~et)] = 1
    output[(wt) & (~tc)] = 2
    return output.unsqueeze(0)

def export_nifti_from_metatensor(metaTensor, outPath, dtype):
    """ Saves a MetaTensor as a NIfTI file while preserving spatial affine metadata. """
    arr_np = metaTensor.detach().cpu().numpy().squeeze().astype(dtype)
    affine = metaTensor.affine if isinstance(metaTensor, MetaTensor) else np.eye(4)
    ni_img = nib.Nifti1Image(arr_np, affine=affine)
    ni_img.header.set_data_dtype(dtype)
    nib.save(ni_img, outPath)

# %% [markdown]
# ### 3. Pipeline Definitions (Transforms and Model)
# %%
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=config["roi_size"], mode="constant"),
    NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

# Define the processing for NIfTI reconstruction
post_pred_nifti_export = Compose([
    Activations(sigmoid=True), 
    AsDiscrete(threshold=0.5), 
    Lambda(reconstruct_brats_labels)
])

# Dataset and Loader
test_files = format_dataset(config["test_base_dir"])
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# Load UNet Architecture
model = UNet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch",
).to(device)

# Load best weights from training
if not os.path.exists(config["best_model_path"]):
    raise FileNotFoundError(f"Weight file not found: {config['best_model_path']}")
model.load_state_dict(torch.load(config["best_model_path"]))
model.eval()

# %% [markdown]
# ### 4. Evaluation Loop
# %%
results_list = []
dice_metric = DiceMetric(include_background=True, reduction="mean_batch")



with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Evaluation"):
        inputs, labels = test_data["image"].to(device), test_data["label"].to(device)
        case_name = test_data["name"][0]
        
        # Inference using sliding window to handle large volumes
        outputs = sliding_window_inference(inputs, config["infer_roi_size"], config["batch_size"], model)
        
        # Calculate Dice scores for the current case
        dice_metric.reset()
        dice_metric(y_pred=[AsDiscrete(sigmoid=True, threshold=0.5)(i) for i in decollate_batch(outputs)], 
                    y=decollate_batch(labels))
        metrics = dice_metric.aggregate()
        
        results_list.append({
            "case": case_name,
            "dice_mean": metrics.mean().item(),
            "dice_tc": metrics[0].item(),
            "dice_wt": metrics[1].item(),
            "dice_et": metrics[2].item()
        })
        
        # Save predicted segmentation as NIfTI (.nii.gz)
        reconstructed = post_pred_nifti_export(outputs[0])
        out_path = os.path.join(config["test_output_dir"], f"{case_name}_pred_seg.nii.gz")
        export_nifti_from_metatensor(reconstructed, out_path, np.uint8)

# %% [markdown]
# ### 5. Final Results Consolidation
# %%
df_results = pd.DataFrame(results_list)

# Add a row for global averages across all test cases
avg_row = {
    "case": "AVERAGE",
    "dice_mean": df_results["dice_mean"].mean(),
    "dice_tc": df_results["dice_tc"].mean(),
    "dice_wt": df_results["dice_wt"].mean(),
    "dice_et": df_results["dice_et"].mean()
}
df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

df_results.to_csv(config["test_results_csv"], index=False)
print(f"\n--- Testing Complete ---")
print(f"Final Test Mean Dice: {avg_row['dice_mean']:.4f}")
print(f"Detailed CSV results saved to: {config['test_results_csv']}")