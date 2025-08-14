import os
import json
import random
import SimpleITK as sitk

INPUT_DIR    = "/playpen/jesse/HaN-Seg/set_1"  
OUTPUT_BASE  = "./hanseg_data_ct/nnUNet_raw"
TASK_ID      = "100"
TASK_NAME    = f"Dataset{int(TASK_ID):03d}_HaNSeg"
OUTPUT_DIR   = os.path.join(OUTPUT_BASE, TASK_NAME)

os.makedirs(os.path.join(OUTPUT_DIR, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labelsTr"), exist_ok=True)

print(f"Creating nnU-Net dataset: {TASK_NAME} (CT-only, all training data)")

cases = sorted([d for d in os.listdir(INPUT_DIR) if d.startswith("case_")])
if not cases:
    raise RuntimeError(f"No case directories found under {INPUT_DIR}")

print(f"Found {len(cases)} cases: {cases[0]} to {cases[-1]}")

train_cases = cases
print(f"All {len(train_cases)} cases will be used for training")

first_case_dir = os.path.join(INPUT_DIR, cases[0])
label_files = [f for f in os.listdir(first_case_dir) if f.endswith(".seg.nrrd")]
label_names = sorted([f.replace(f"{cases[0]}_OAR_", "").replace(".seg.nrrd", "") for f in label_files])

print(f"Found {len(label_names)} organs/structures")

labels_dict = {}
for i, name in enumerate(label_names, start=1):
    labels_dict[name] = i

labels_dict = {"background": 0, **labels_dict}

print(f"Label mapping created: background=0, organs=1-{len(label_names)}")

def convert_case(case, out_img_folder, out_lbl_folder):
    case_dir = os.path.join(INPUT_DIR, case)
    
    try:
        ct_img = sitk.ReadImage(os.path.join(case_dir, f"{case}_IMG_CT.nrrd"))
        sitk.WriteImage(ct_img, os.path.join(out_img_folder, f"{case}_0000.nii.gz"))
        
        print(f"    CT: {ct_img.GetSize()}")
        
        lbl_img = sitk.Image(ct_img.GetSize(), sitk.sitkUInt8)
        lbl_img.CopyInformation(ct_img)
        
        organs_found = 0
        organs_missing = 0
        
        for name in label_names:
            label_value = labels_dict[name]  
            seg_path = os.path.join(case_dir, f"{case}_OAR_{name}.seg.nrrd")
            
            if os.path.exists(seg_path):
                seg = sitk.ReadImage(seg_path)
                seg = sitk.Cast(sitk.Resample(seg, ct_img), sitk.sitkUInt8)
                lbl_img = lbl_img + (seg > 0) * label_value
                organs_found += 1
            else:
                organs_missing += 1
        
        sitk.WriteImage(lbl_img, os.path.join(out_lbl_folder, f"{case}.nii.gz"))
        
        print(f"  {case}: {organs_found}/{len(label_names)} organs found ({organs_missing} missing)")
        return True
        
    except Exception as e:
        print(f"  ERROR {case}: {e}")
        return False

print(f"\nConverting {len(train_cases)} training cases...")
train_success = 0
for i, case in enumerate(train_cases, 1):
    print(f"[{i}/{len(train_cases)}]", end=" ")
    if convert_case(case, 
                    os.path.join(OUTPUT_DIR, "imagesTr"),
                    os.path.join(OUTPUT_DIR, "labelsTr")):
        train_success += 1


dataset = {
    "channel_names": {
        "0": "CT"
    },
    "labels": labels_dict,  
    "numTraining": train_success,
    "file_ending": ".nii.gz",
    "dataset_name": TASK_NAME,
    "reference": "HaN-Seg Dataset",
    "licence": "",
    "release": "",
    "tensorImageSize": "4D",
    "modality": {
        "0": "CT"
    },
    "numTest": 0,  
    "name": "HaNSeg_CT_OAR_Segmentation",
    "description": "Head and Neck organ-at-risk segmentation using CT only - all training data"
}

json_path = os.path.join(OUTPUT_DIR, "dataset.json")
with open(json_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\n" + "="*60)
print(f"DATASET CREATION COMPLETE")
print(f"="*60)
print(f"Dataset: {TASK_NAME}")
print(f"Location: {OUTPUT_DIR}")
print(f"")
print(f"Cases processed:")
print(f"  Training: {train_success}/{len(train_cases)} successful")
print(f"  Testing:  0 (all data used for training)")
print(f"  Total:    {train_success}/{len(cases)} successful")
print(f"")
print(f"Data structure:")
print(f"  Modalities: CT only (1 channel)")
print(f"  Organs: {len(label_names)} + background")
print(f"  Labels: 0={labels_dict['background']}, 1-{len(label_names)} for organs")
print(f"")
print(f"Files created:")
print(f"  imagesTr/: {train_success} CT files")
print(f"  labelsTr/: {train_success} label files")
print(f"  imagesTs/: 0 files (no test set)")
print(f"  labelsTs/: 0 files (no test set)")
print(f"  dataset.json: 1 file")

# Verify dataset.json contents
with open(json_path, "r") as f:
    loaded_dataset = json.load(f)
    background_ok = loaded_dataset['labels'].get('background') == 0
    print(f"")
    print(f"Dataset verification:")
    print(f"  Background label 0: {'✓' if background_ok else '✗'}")
    print(f"  Total labels: {len(loaded_dataset['labels'])}")
    print(f"  Training cases: {loaded_dataset['numTraining']}")
    print(f"  Test cases: {loaded_dataset['numTest']}")
    print(f"  Modalities: CT only")

if background_ok and train_success > 0:
    print(f"\nCT-only dataset ready for nnU-Net v2! (All training data)")
else:
    print(f"\nIssues detected - please check the errors above")

print(f"="*60)