#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom, center_of_mass, binary_dilation
import json
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import pickle

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

class InteractiveDataset(Dataset):
    def __init__(self, data_root, target_shape=(64, 128, 128), use_preprocessed=True):
        self.data_root = Path(data_root)
        self.target_shape = target_shape
        self.use_preprocessed = use_preprocessed
        
        with open(self.data_root / "dataset.json", 'r') as f:
            self.dataset_info = json.load(f)
        
        nnunet_preprocessed = Path("/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_preprocessed/Dataset100_HaNSeg")
        with open(nnunet_preprocessed / "splits_final.json", 'r') as f:
            self.splits = json.load(f)
        
        self.training_cases = self.splits[0]['train']
        self.organs = list(self.dataset_info['labels'].keys())
        self.organs.remove('background')
        
        if self.use_preprocessed:
            self.preprocessed_path = nnunet_preprocessed / "nnUNetPlans_3d_fullres"
            print(f"Using nnUNet preprocessed data from: {self.preprocessed_path}")
        else:
            print(f"Using nnUNet raw data from: {self.data_root}")
        
        self.samples = []
        for case in self.training_cases:
            for organ in self.organs:
                if self.use_preprocessed:
                    data_file = self.preprocessed_path / f"{case}.b2nd"
                    pkl_file = self.preprocessed_path / f"{case}.pkl"
                    
                    if data_file.exists() and pkl_file.exists():
                        self.samples.append((case, organ))
                else:
                    image_file = self.data_root / "imagesTr" / f"{case}_0000.nii.gz"
                    label_file = self.data_root / "labelsTr" / f"{case}.nii.gz"
                    
                    if image_file.exists() and label_file.exists():
                        self.samples.append((case, organ))
        
        print(f"Dataset: {len(self.samples)} samples, target shape: {self.target_shape}")
        print(f"Using {'preprocessed' if self.use_preprocessed else 'raw'} nnUNet data")

    def load_preprocessed_case(self, case_name):
        try:
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
            
            dataset = nnUNetDatasetBlosc2(
                folder=str(self.preprocessed_path),
                identifiers=[case_name],
                folder_with_segs_from_previous_stage=None
            )
            
            data, seg, seg_prev, properties = dataset.load_case(case_name)
            
            # Print preprocessing properties for debugging
            print("\nnnUNet preprocessing properties:")
            for key, value in properties.items():
                print(f"{key}: {value}")
            
            ct_array = np.array(data[0]).astype(np.float32)
            label_full = np.array(seg[0]).astype(np.uint8)  # Use preprocessed segmentation
            
            # Check if we need to transpose the data based on original orientation
            if 'transpose_forward' in properties:
                transpose_order = properties['transpose_forward']
                print(f"\nTranspose order from nnUNet: {transpose_order}")
                # Apply the inverse transpose to get back to original orientation
                ct_array = np.transpose(ct_array, np.argsort(transpose_order))
                label_full = np.transpose(label_full, np.argsort(transpose_order))
            
            return ct_array[None], label_full, properties 
            
        except Exception as e:
            print(f"Error loading preprocessed case {case_name}: {e}")
            return None, None, None

    def preprocess_ct(self, ct_image):
        if ct_image.GetPixelID() != sitk.sitkFloat32:
            ct_image = sitk.Cast(ct_image, sitk.sitkFloat32)
        
        ct_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        
        ct_array = sitk.GetArrayFromImage(ct_image)[None]
        ct_array = ct_array.astype(np.float32)
        
        ct_min, ct_max = ct_array.min(), ct_array.max()
        if ct_max > ct_min:
            ct_array = (ct_array - ct_min) / (ct_max - ct_min)
        
        original_shape = ct_array.shape[1:]
        zoom_factors = [self.target_shape[i] / original_shape[i] for i in range(3)]
        
        resampled_ct = zoom(ct_array[0], zoom_factors, order=1)
        resampled_ct = resampled_ct[None]
        
        return resampled_ct, zoom_factors

    def generate_interactions(self, ground_truth_mask, num_interactions=5):
        interactions = []
        
        # Get mask dimensions
        D, H, W = ground_truth_mask.shape
        print(f"\nMask shape in generate_interactions: {ground_truth_mask.shape}")
        
        # Find all positive points
        z_coords, y_coords, x_coords = np.where(ground_truth_mask > 0)
        if len(z_coords) == 0:
            print("No positive points found in mask")
            return interactions
            
        # Calculate center of mass
        center_z = int(np.mean(z_coords))
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        print(f"Original center point: ({center_z}, {center_y}, {center_x})")
        print(f"Coordinate ranges: z=[0,{D-1}], y=[0,{H-1}], x=[0,{W-1}]")
        
        # Ensure coordinates are within bounds
        center_z = np.clip(center_z, 0, D-1)
        center_y = np.clip(center_y, 0, H-1)
        center_x = np.clip(center_x, 0, W-1)
        
        # Add the center point
        if ground_truth_mask[center_z, center_y, center_x]:
            interactions.append((center_z, center_y, center_x, True))
            print(f"Added interaction point: ({center_z}, {center_y}, {center_x})")
            
            # Print a small region around the point to verify it's on the mask
            z_min = max(0, center_z - 1)
            z_max = min(D, center_z + 2)
            y_min = max(0, center_y - 1)
            y_max = min(H, center_y + 2)
            x_min = max(0, center_x - 1)
            x_max = min(W, center_x + 2)
            
            region = ground_truth_mask[z_min:z_max, y_min:y_max, x_min:x_max]
            print(f"3x3x3 region around point:\n{region}")
        else:
            print(f"Warning: Center point ({center_z}, {center_y}, {center_x}) is not on the mask")
        
        return interactions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case, organ = self.samples[idx]
        
        if self.use_preprocessed:
            ct_array, label_full, properties = self.load_preprocessed_case(case)
            if ct_array is None:
                return None
            
            organ_id = self.dataset_info['labels'][organ]
            organ_mask = (label_full == organ_id).astype(np.uint8)
            
            if ct_array.shape[1:] != self.target_shape:
                zoom_factors = [self.target_shape[i] / ct_array.shape[i+1] for i in range(3)]
                
                resampled_ct = zoom(ct_array[0], zoom_factors, order=1)
                ct_array = resampled_ct[None]
                
                resampled_mask = zoom(organ_mask.astype(float), zoom_factors, order=0)
                organ_mask = (resampled_mask > 0.5).astype(np.uint8)
            
            interactions = self.generate_interactions(organ_mask)
            
            return {
                'ct_array': ct_array,
                'organ_mask': organ_mask,
                'interactions': interactions,
                'case': case,
                'organ': organ
            }
        else:
            image_file = self.data_root / "imagesTr" / f"{case}_0000.nii.gz"
            label_file = self.data_root / "labelsTr" / f"{case}.nii.gz"
            
            ct_image = sitk.ReadImage(str(image_file))
            ct_array, zoom_factors = self.preprocess_ct(ct_image)
            
            label_image = sitk.ReadImage(str(label_file))
            label_full = sitk.GetArrayFromImage(label_image).astype(np.uint8)
            
            label_full = zoom(label_full.astype(float), zoom_factors, order=0)
            label_full = (label_full > 0.5).astype(np.uint8)
            
            organ_id = self.dataset_info['labels'][organ]
            organ_mask = (label_full == organ_id).astype(np.uint8)
            
            interactions = self.generate_interactions(organ_mask)
            
            return {
                'ct_array': ct_array,
                'organ_mask': organ_mask,
                'interactions': interactions,
                'case': case,
                'organ': organ
            }

class HaNSegEvaluator:
    """Lightweight evaluator for epoch-wise validation"""
    def __init__(self, session, target_shape=(128, 512, 512), use_preprocessed=False):
        self.session = session
        self.target_shape = target_shape
        self.device = session.device
        self.use_preprocessed = use_preprocessed
        
        if self.use_preprocessed:
            self.preprocessed_path = Path("/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_preprocessed/Dataset100_HaNSeg/nnUNetPlans_3d_fullres")
            print(f"Evaluator using nnUNet preprocessed data from: {self.preprocessed_path}")
        else:
            print(f"Evaluator using raw HaN-Seg data (recommended for test set evaluation)")
        
        print(f"Evaluation target shape: {self.target_shape} (same as eval.py configuration)")
    
    def load_preprocessed_case_for_eval(self, case_name):
        """Load nnUNet preprocessed case data for evaluation"""
        try:
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
            
            dataset = nnUNetDatasetBlosc2(
                folder=str(self.preprocessed_path),
                identifiers=[case_name],
                folder_with_segs_from_previous_stage=None
            )
            
            data, seg, seg_prev, properties = dataset.load_case(case_name)
            
            ct_array = np.array(data[0]).astype(np.float32)
            ct_array = ct_array[None]
            
            return ct_array, properties
            
        except Exception as e:
            print(f"Error loading preprocessed case {case_name} for evaluation: {e}")
            return None, None
    
    def preprocess_ct_for_eval(self, ct_image):
        """Preprocess CT image for evaluation (higher resolution)"""
        if ct_image.GetPixelID() != sitk.sitkFloat32:
            ct_image = sitk.Cast(ct_image, sitk.sitkFloat32)
        
        ct_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        
        ct_array = sitk.GetArrayFromImage(ct_image)[None]
        ct_array = ct_array.astype(np.float32)
        
        ct_min, ct_max = ct_array.min(), ct_array.max()
        if ct_max > ct_min:
            ct_array = (ct_array - ct_min) / (ct_max - ct_min)
        
        original_shape = ct_array.shape[1:]
        zoom_factors = [self.target_shape[i] / original_shape[i] for i in range(3)]
        
        resampled_ct = zoom(ct_array[0], zoom_factors, order=1)
        resampled_ct = resampled_ct[None]
        
        return resampled_ct, zoom_factors
    
    def preprocess_mask_for_eval(self, mask_image, zoom_factors):
        """Preprocess mask to match evaluation CT"""
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask_array = (mask_array > 0).astype(np.uint8)
        
        resampled_mask = zoom(mask_array.astype(float), zoom_factors, order=0)
        resampled_mask = (resampled_mask > 0.5).astype(np.uint8)
        
        return resampled_mask
    
    def load_hanseg_case(self, case_path):
        """Load single HaN-Seg case for evaluation"""
        case_path = Path(case_path)
        case_name = case_path.name
        
        if self.use_preprocessed:
            ct_array, properties = self.load_preprocessed_case_for_eval(case_name)
            if ct_array is not None:
                oar_files = list(case_path.glob(f"{case_name}_OAR_*.seg.nrrd"))
                ground_truth_masks = {}
                
                for oar_file in oar_files:
                    oar_name = oar_file.stem.replace(f"{case_name}_OAR_", "").replace(".seg", "")
                    
                    oar_image = sitk.ReadImage(str(oar_file))
                    oar_array = sitk.GetArrayFromImage(oar_image)
                    oar_array = (oar_array > 0).astype(np.uint8)
                    
                    if 'bbox_used_for_cropping' in properties:
                        bbox = properties['bbox_used_for_cropping']
                        oar_array = oar_array[bbox[0][0]:bbox[0][1], 
                                             bbox[1][0]:bbox[1][1], 
                                             bbox[2][0]:bbox[2][1]]
                    
                    if oar_array.shape != ct_array.shape[1:]:
                        zoom_factors = [ct_array.shape[i+1] / oar_array.shape[i] for i in range(3)]
                        oar_array = zoom(oar_array.astype(float), zoom_factors, order=0)
                        oar_array = (oar_array > 0.5).astype(np.uint8)
                    
                    if np.sum(oar_array > 0) > 0:
                        ground_truth_masks[oar_name] = oar_array
                
                return ct_array, ground_truth_masks
        
        ct_path = case_path / f"{case_name}_IMG_CT.nrrd"
        if not ct_path.exists():
            return None, {}
        
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array, zoom_factors = self.preprocess_ct_for_eval(ct_image)
        
        oar_files = list(case_path.glob(f"{case_name}_OAR_*.seg.nrrd"))
        ground_truth_masks = {}
        
        for oar_file in oar_files:
            oar_name = oar_file.stem.replace(f"{case_name}_OAR_", "").replace(".seg", "")
            
            oar_image = sitk.ReadImage(str(oar_file))
            oar_array = self.preprocess_mask_for_eval(oar_image, zoom_factors)
            
            if np.sum(oar_array > 0) > 0:
                ground_truth_masks[oar_name] = oar_array
        
        return ct_array, ground_truth_masks
    
    def generate_interactions_for_eval(self, ground_truth_mask, num_interactions=5):
        """Generate interaction points for evaluation"""
        interactions = []
        
        positive_coords = np.where(ground_truth_mask > 0)
        if len(positive_coords[0]) == 0:
            return interactions
        
        center_z, center_y, center_x = [int(c) for c in center_of_mass(ground_truth_mask)]
        
        if ground_truth_mask[center_z, center_y, center_x] > 0:
            interactions.append((center_z, center_y, center_x, True))
        
        num_additional = min(num_interactions - 1, len(positive_coords[0]) - 1)
        if num_additional > 0:
            indices = np.random.choice(len(positive_coords[0]), num_additional, replace=False)
            for i in indices:
                z, y, x = positive_coords[0][i], positive_coords[1][i], positive_coords[2][i]
                if (z, y, x) != (center_z, center_y, center_x):
                    interactions.append((z, y, x, True))
        
        return interactions
    
    def segment_with_interactions(self, ct_array, ground_truth_mask, num_interactions=5):
        """Perform interactive segmentation"""
        try:
            torch.cuda.empty_cache()
            
            self.session.reset_interactions()
            
            self.session.set_image(ct_array)
            target_tensor = torch.zeros(ct_array.shape[1:], dtype=torch.uint8, device=self.device)
            self.session.set_target_buffer(target_tensor)
            
            interactions = self.generate_interactions_for_eval(ground_truth_mask, num_interactions)
            
            if not interactions:
                return np.zeros_like(ground_truth_mask)
            
            for z, y, x, is_positive in interactions:
                try:
                    self.session.add_point_interaction((z, y, x), include_interaction=is_positive)
                except Exception as e:
                    print(f"Interaction failed: {e}")
                    break
            
            result = self.session.target_buffer.clone().cpu().numpy()
            
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"Segmentation failed: {e}")
            torch.cuda.empty_cache()
            return np.zeros_like(ground_truth_mask)
    
    def calculate_dice(self, predicted, ground_truth):
        """Calculate Dice coefficient"""
        pred_binary = (predicted > 0).astype(np.uint8)
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)
        
        return (2.0 * intersection) / (union + 1e-8)
    
    def quick_eval(self, hanseg_root, eval_cases=None, num_interactions=5):
        """Quick evaluation on subset of cases"""
        if eval_cases is None:
            test_images_dir = Path("/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_raw/Dataset100_HaNSeg/imagesTs")
            
            test_files = list(test_images_dir.glob("*_0000.nii.gz"))
            eval_cases = []
            for f in test_files:
                case_name = f.stem.replace('_0000', '').replace('.nii', '')
                eval_cases.append(case_name)
            
            eval_cases = sorted(eval_cases)
            print(f"Using actual test set cases: {len(eval_cases)} cases")
            print(f"Test cases: {eval_cases}")
        
        hanseg_root = Path(hanseg_root)
        set_1_path = hanseg_root / "set_1"
        
        dice_scores = []
        
        for case_name in eval_cases:
            case_path = set_1_path / case_name
            if not case_path.exists():
                continue
                
            try:
                torch.cuda.empty_cache()
                
                ct_array, ground_truth_masks = self.load_hanseg_case(case_path)
                if ct_array is None:
                    continue
                
                case_dice_scores = []
                for oar_name, gt_mask in ground_truth_masks.items():
                    predicted_mask = self.segment_with_interactions(
                        ct_array, gt_mask, num_interactions
                    )
                    
                    dice = self.calculate_dice(predicted_mask, gt_mask)
                    case_dice_scores.append(dice)
                
                if case_dice_scores:
                    dice_scores.extend(case_dice_scores)
                
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Eval case {case_name} failed: {e}")
                torch.cuda.empty_cache()
                continue
        
        return np.mean(dice_scores) if dice_scores else 0.0

class InteractiveTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_shape = (128, 512, 512)
        
        self.num_epochs = 10
        self.batch_size = 2 
        self.lr = 5e-5
        
        self.eval_every_epoch = True
        self.hanseg_root = "/playpen/jesse/HaN-Seg"
        self.eval_cases = None 
        
        self.patience = 3
        self.min_delta = 0.01
        
        print(f"Interactive Trainer initialized")
        print(f"Training shape: {self.target_shape}, Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr} (reduced for fine-tuning)")
        print(f"Evaluation: {'Enabled' if self.eval_every_epoch else 'Disabled'}")
        print(f"Early stopping: patience={self.patience}, min_delta={self.min_delta}")
    
    def load_nninteractive_session(self):
        """Initialize nnInteractive session"""
        try:
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            
            model_path = "/playpen/jesse/image_seg/nninteract/nnInteractive_v1.0"
            
            session = nnInteractiveInferenceSession(
                device=self.device,
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True,
            )
            
            session.initialize_from_trained_model_folder(model_path)
            print("nnInteractive session loaded successfully")
            
            return session
            
        except Exception as e:
            print(f"Failed to load nnInteractive session: {e}")
            return None
    
    def train(self):
        """Execute interactive fine-tuning with evaluation"""
        print("Starting interactive fine-tuning with evaluation...")

        session = self.load_nninteractive_session()
        if session is None:
            return None

        dataset = InteractiveDataset(
            data_root="/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_raw/Dataset100_HaNSeg",
            target_shape=self.target_shape,
            use_preprocessed=True 
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        def custom_collate(batch):
            return batch

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=custom_collate
        )

        print(f"Training samples: {len(train_dataset)}")

        model = session.network

        evaluator = HaNSegEvaluator(session, use_preprocessed=False) if self.eval_every_epoch else None

        print("\n" + "="*60)
        print("EVALUATING PRETRAINED MODEL (BEFORE FINE-TUNING)")
        print("="*60)
        
        model.eval() 
        pretrain_dice = 0.0
        
        if evaluator is not None:
            try:
                pretrain_dice = evaluator.quick_eval(
                    hanseg_root=self.hanseg_root,
                    eval_cases=self.eval_cases,
                    num_interactions=5
                )
                print(f"PRETRAINED MODEL - Evaluation Dice: {pretrain_dice:.4f}")
            except Exception as e:
                print(f"Pretrained model evaluation failed: {e}")
        
        print("="*60)
        print("STARTING FINE-TUNING...")
        print("="*60 + "\n")

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, 
            min_lr=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        torch.cuda.empty_cache()

        print("Starting fine-tuning...")

        scaler = torch.cuda.amp.GradScaler()
        
        training_history = []
        
        best_eval_dice = pretrain_dice
        epochs_without_improvement = 0
        
        pretrain_results = {
            'epoch': 0,  
            'train_loss': 0.0,
            'eval_dice': pretrain_dice,
            'model_type': 'pretrained'
        }
        training_history.append(pretrain_results)

        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in pbar:
                batch_loss = self.batch_interactive_training_step(model, criterion, scaler, optimizer, batch)
                
                if batch_loss is not None:
                    epoch_loss += batch_loss
                    num_batches += 1
                    pbar.set_postfix(loss=batch_loss)

                if num_batches % 5 == 0:
                    torch.cuda.empty_cache()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            eval_dice = 0.0
            if self.eval_every_epoch and evaluator is not None:
                print(f"\nEvaluating epoch {epoch+1} model...")
                model.eval()
                
                torch.cuda.empty_cache()
                
                try:
                    eval_dice = evaluator.quick_eval(
                        hanseg_root=self.hanseg_root,
                        eval_cases=self.eval_cases,
                        num_interactions=5
                    )
                    print(f"Epoch {epoch+1} - Evaluation Dice: {eval_dice:.4f}")
                    
                    improvement = eval_dice - pretrain_dice
                    print(f"Improvement over pretrained: {improvement:+.4f}")
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
                model.train()
            

            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'eval_dice': eval_dice,
                'model_type': 'fine_tuned'
            }
            training_history.append(epoch_results)
            
            print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Eval Dice: {eval_dice:.4f}")

            if self.eval_every_epoch and evaluator is not None:
                scheduler.step(eval_dice)
                
                if eval_dice > best_eval_dice + self.min_delta:
                    best_eval_dice = eval_dice
                    epochs_without_improvement = 0
                    print(f"New best evaluation Dice: {best_eval_dice:.4f}")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement for {epochs_without_improvement} epoch(s)")
                    
                    if epochs_without_improvement >= self.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        print(f"Best evaluation Dice: {best_eval_dice:.4f}")
                        break

            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory: {memory_gb:.1f}GB")
                torch.cuda.empty_cache()

        print("\nInteractive fine-tuning completed!")
        
        print("\n" + "="*60)
        print("FINAL PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Pretrained model Dice:  {pretrain_dice:.4f}")
        if len(training_history) > 1:
            final_dice = training_history[-1]['eval_dice']
            print(f"Fine-tuned model Dice:  {final_dice:.4f}")
            print(f"Total improvement:      {final_dice - pretrain_dice:+.4f}")
        print("="*60)

        save_path = "/playpen/jesse/image_seg/nnInteractive_tuning/models/tuned_interactive_model.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved to: {save_path}")
        
        history_df = pd.DataFrame(training_history)
        history_path = "/playpen/jesse/image_seg/nnInteractive_tuning/training_history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to: {history_path}")

        return model

    def generate_corrective_interactions(self, gt_mask, current_pred, num_points=2):
        """Generate corrective interaction points based on GT-output difference"""
        interactions = []
        
        false_negatives = (gt_mask > 0) & (current_pred == 0)
        fn_coords = np.where(false_negatives)
        
        if len(fn_coords[0]) > 0 and num_points > 0:
            num_pos = min(num_points, len(fn_coords[0]))
            indices = np.random.choice(len(fn_coords[0]), num_pos, replace=False)
            for i in indices:
                z, y, x = fn_coords[0][i], fn_coords[1][i], fn_coords[2][i]
                interactions.append((z, y, x, True))  
        
        false_positives = (gt_mask == 0) & (current_pred > 0)
        fp_coords = np.where(false_positives)
        
        remaining_points = num_points - len(interactions)
        if len(fp_coords[0]) > 0 and remaining_points > 0:
            num_neg = min(remaining_points, len(fp_coords[0]))
            indices = np.random.choice(len(fp_coords[0]), num_neg, replace=False)
            for i in indices:
                z, y, x = fp_coords[0][i], fp_coords[1][i], fp_coords[2][i]
                interactions.append((z, y, x, False))
        
        return interactions

    def batch_interactive_training_step(self, model, criterion, scaler, optimizer, batch):
        batch_size = len(batch)
        
        ct_batch = []
        organ_mask_batch = []
        initial_interactions = []
        
        for sample in batch:
            ct_array = sample['ct_array']
            organ_mask = sample['organ_mask']
            interactions = sample['interactions']
            
            if len(interactions) == 0:
                continue
                
            ct_batch.append(ct_array)
            organ_mask_batch.append(organ_mask)
            initial_interactions.append(interactions[0])
        
        if len(ct_batch) == 0:
            return None
            
        actual_batch_size = len(ct_batch)
        
        try:
            input_batch = torch.zeros(actual_batch_size, 8, *self.target_shape, 
                                    device=self.device, dtype=torch.float16)
            
            for i, ct_array in enumerate(ct_batch):
                ct_numpy = ct_array.numpy() if isinstance(ct_array, torch.Tensor) else ct_array
                input_batch[i, 0] = torch.from_numpy(ct_numpy[0]).to(self.device, dtype=torch.float16)
            
            target_batch = torch.zeros(actual_batch_size, *self.target_shape, 
                                     dtype=torch.long, device=self.device)
            
            for i, organ_mask in enumerate(organ_mask_batch):
                organ_numpy = organ_mask.numpy() if isinstance(organ_mask, torch.Tensor) else organ_mask
                target_batch[i] = torch.from_numpy(organ_numpy).to(self.device)
            
            for iteration in range(3):
                input_batch[:, 1:, :, :, :] = 0
                
                if iteration == 0:
                    current_interactions = initial_interactions
                else:
                    current_interactions = []
                    
                    with torch.no_grad():
                        current_outputs = model(input_batch.float())
                        current_preds = torch.argmax(current_outputs, dim=1).cpu().numpy()
                    
                    for i in range(actual_batch_size):
                        gt_mask = organ_mask_batch[i]
                        gt_numpy = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
                        pred_mask = current_preds[i]
                        
                        corrective_points = self.generate_corrective_interactions(
                            gt_numpy, pred_mask, num_points=2
                        )
                        
                        if not corrective_points:
                            positive_coords = np.where(gt_numpy > 0)
                            if len(positive_coords[0]) > 0:
                                idx = np.random.choice(len(positive_coords[0]))
                                z, y, x = positive_coords[0][idx], positive_coords[1][idx], positive_coords[2][idx]
                                corrective_points = [(z, y, x, True)]
                        
                        current_interactions.append(corrective_points[0] if corrective_points else initial_interactions[i])
                
                for i, point in enumerate(current_interactions):
                    if point is not None:
                        z, y, x, is_positive = point
                        if is_positive:
                            input_batch[i, 1, z, y, x] = 1.0  
                        else:
                            input_batch[i, 2, z, y, x] = 1.0
                
                if iteration < 2:
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    with torch.no_grad():
                        outputs = model(input_batch.float())
                else:
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(input_batch.float())
                        loss = criterion(outputs, target_batch)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    final_loss = loss.item()
            
            del input_batch, target_batch, outputs
            
            return final_loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA OOM with batch size {actual_batch_size}")
                torch.cuda.empty_cache()
                return None
            else:
                raise e
        except Exception as e:
            print(f"Error in batch training: {e}")
            return None

def main():
    print("=== Interactive nnInteractive Fine-tuning with 3-Iteration Training ===")
    print("This script will:")
    print("1. Load pretrained nnInteractive model")
    print("2. Evaluate pretrained model performance with 5 interaction points")
    print("3. Fine-tune with 3-iteration interactive training:")
    print("   - 1st-2nd iterations: Frozen model (no gradient updates)")  
    print("   - 3rd iteration: Calculate loss and update parameters")
    print("4. Evaluate after each epoch and compare with pretrained performance")
    print("5. Save final model and training history")
    print("=" * 70)

    trainer = InteractiveTrainer()
    model = trainer.train()

    print("\nFine-tuning completed!")
    print("=" * 50)
    print("Results summary:")
    print("- Check training_history.csv for detailed epoch-wise performance")
    print("- CSV includes both pretrained (epoch 0) and fine-tuned performance")
    print("- Final model saved for further evaluation")
    print("=" * 50)
    print("Next steps:")
    print("- Run full evaluation script to test final model performance!")
    print("- Compare pretrained vs fine-tuned results in the history CSV")

if __name__ == "__main__":
    main()