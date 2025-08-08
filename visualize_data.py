import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from training_with_eval import InteractiveDataset

def visualize_slice(ct_slice, mask_slice=None, title=None, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Convert CT values back to HU units (approximately)
    ct_slice = ct_slice * 2000 - 1000  # Rough conversion to HU
    
    # Window/level adjustment for better CT visualization
    window_center = 40   # Soft tissue window
    window_width = 400
    vmin = window_center - window_width/2
    vmax = window_center + window_width/2
    
    if mask_slice is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title('CT Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        # Create a red mask for overlay
        mask_overlay = np.zeros((*mask_slice.shape, 4))  # RGBA
        mask_overlay[mask_slice > 0] = [1, 0, 0, 0.3]  # Red with 0.3 alpha
        plt.imshow(mask_overlay)
        plt.title('CT + Mask Overlay')
        plt.axis('off')
    else:
        plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title('CT Image')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_preprocessing_steps(data_root, preprocessed_path, case_name, organ_name, output_dir="visualization_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dataset = InteractiveDataset(
        data_root=data_root,
        target_shape=(128, 512, 512),
        use_preprocessed=True
    )
    
    dataset.preprocessed_path = Path(preprocessed_path) / "nnUNetPlans_3d_fullres"
    
    sample_idx = None
    for idx, (c, o) in enumerate(dataset.samples):
        if c == case_name and o == organ_name:
            sample_idx = idx
            break
    
    if sample_idx is None:
        print(f"Case {case_name} with organ {organ_name} not found in dataset")
        return
    
    sample = dataset[sample_idx]
    if sample is None:
        print("Failed to load sample")
        return
    
    ct_array = sample['ct_array'][0]  # Remove channel dimension
    organ_mask = sample['organ_mask']
    interactions = sample['interactions']
    
    print(f"\nData statistics:")
    print(f"CT array shape: {ct_array.shape}")
    print(f"CT value range: [{ct_array.min():.2f}, {ct_array.max():.2f}]")
    print(f"Mask shape: {organ_mask.shape}")
    print(f"Number of positive voxels in mask: {np.sum(organ_mask > 0)}")
    
    if interactions:
        plt.figure(figsize=(15, 5))
        window_center = 40   # Soft tissue window
        window_width = 400
        vmin = window_center - window_width/2
        vmax = window_center + window_width/2
        
        for i, (z, y, x, is_positive) in enumerate(interactions):
            print(f"\nVisualization coordinates:")
            print(f"Original point: z={z}, y={y}, x={x}")
            
            # Try different coordinate mappings
            mappings = [
                # Original mapping
                {'z': z, 'y': y, 'x': x},
                # Flipped y and z
                {'z': z, 'y': ct_array.shape[1]-y, 'x': x},
                # Flipped x
                {'z': z, 'y': y, 'x': ct_array.shape[2]-x},
                # All flipped
                {'z': ct_array.shape[0]-z, 'y': ct_array.shape[1]-y, 'x': ct_array.shape[2]-x}
            ]
            
            for idx, mapping in enumerate(mappings):
                # Create a new figure for each mapping
                plt.figure(figsize=(15, 5))
                
                # Axial view (z-plane)
                plt.subplot(1, 3, 1)
                ct_slice = ct_array[mapping['z']] * 2000 - 1000
                plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
                mask_overlay = np.zeros((*ct_slice.shape, 4))
                mask_overlay[organ_mask[mapping['z']] > 0] = [1, 0, 0, 0.3]
                plt.imshow(mask_overlay)
                plt.plot(mapping['x'], mapping['y'], 'r+' if is_positive else 'b+', markersize=15, linewidth=2)
                plt.title(f'Axial View (z={mapping["z"]})')
                plt.axis('off')
                
                # Coronal view (y-plane)
                plt.subplot(1, 3, 2)
                ct_slice = ct_array[:, mapping['y'], :].T * 2000 - 1000
                plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
                mask_overlay = np.zeros((*ct_slice.shape, 4))
                mask_overlay[organ_mask[:, mapping['y'], :].T > 0] = [1, 0, 0, 0.3]
                plt.imshow(mask_overlay)
                plt.plot(mapping['x'], mapping['z'], 'r+' if is_positive else 'b+', markersize=15, linewidth=2)
                plt.title(f'Coronal View (y={mapping["y"]})')
                plt.axis('off')
                
                # Sagittal view (x-plane)
                plt.subplot(1, 3, 3)
                ct_slice = ct_array[:, :, mapping['x']].T * 2000 - 1000
                plt.imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax)
                mask_overlay = np.zeros((*ct_slice.shape, 4))
                mask_overlay[organ_mask[:, :, mapping['x']].T > 0] = [1, 0, 0, 0.3]
                plt.imshow(mask_overlay)
                plt.plot(mapping['y'], mapping['z'], 'r+' if is_positive else 'b+', markersize=15, linewidth=2)
                plt.title(f'Sagittal View (x={mapping["x"]})')
                plt.axis('off')
                
                plt.suptitle(f"Mapping {idx+1} - Point ({mapping['z']}, {mapping['y']}, {mapping['x']})")
                plt.tight_layout()
                plt.savefig(output_dir / f"{case_name}_{organ_name}_mapping{idx+1}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"\nVisualization results saved to {output_dir}")

if __name__ == "__main__":
    data_root = "/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_raw/Dataset100_HaNSeg"
    preprocessed_path = "/playpen/jesse/image_seg/hanseg_code_ct/nnUNet_preprocessed/Dataset100_HaNSeg"
    case_name = "case_02"
    organ_name = "Brainstem"
    
    visualize_preprocessing_steps(data_root, preprocessed_path, case_name, organ_name) 