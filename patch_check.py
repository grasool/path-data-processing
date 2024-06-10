def check_patches(patches,mask_patches):
    valid_indices = [i for i, mask in enumerate(mask_patches) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = patches[valid_indices]
    filtered_masks = mask_patches[valid_indices]
    print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
    print("Mask shape:", filtered_masks.shape)
    return filtered_images,filtered_masks