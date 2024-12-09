import numpy as np
import cupy as cp
from scipy.fftpack import fft2, ifft2
from skimage.restoration import denoise_wavelet
import tifffile as tiff

########################################
# Utility Functions
########################################
def load_image(image_path):
    """
    Load the noisy image.
    """
    image = tiff.imread(image_path).astype(np.float32)
    return cp.array(image)

def save_image(image, output_path):
    """
    Save the denoised image to disk.
    """
    tiff.imwrite(output_path, cp.asnumpy(image).astype(np.float32))

########################################
# FFT-based Block Matching
########################################
def fft_block_matching(image, patch_size, search_radius, max_patches=16):
    """
    Fast block matching using FFT-based cross-correlation.
    Ensures that all extracted patches have uniform shape.
    """
    H, W = image.shape
    padded_image = cp.pad(image, search_radius, mode='reflect')
    padded_numpy = cp.asnumpy(padded_image)  # Convert to NumPy for FFT ops

    blocks = []
    padded_H, padded_W = padded_numpy.shape

    for i in range(search_radius, H + search_radius):
        for j in range(search_radius, W + search_radius):
            # Extract the reference patch
            ref_patch = padded_numpy[i:i + patch_size, j:j + patch_size]
            if ref_patch.shape != (patch_size, patch_size):
                # If we can't extract a full patch, skip
                continue

            # Pad the reference patch to the size of the padded image
            ref_patch_padded = np.zeros_like(padded_numpy)
            ref_patch_padded[i:i + patch_size, j:j + patch_size] = ref_patch

            # Compute cross-correlation using FFT
            correlation = ifft2(fft2(ref_patch_padded) * np.conj(fft2(padded_numpy))).real

            # Compute distances and find the top matches
            distances = np.sum(padded_numpy ** 2) + np.sum(ref_patch_padded ** 2) - 2 * correlation
            flat_indices = np.argsort(distances.ravel())

            # Extract matched patches, ensuring correct shape
            valid_patches = []
            for idx in flat_indices:
                if len(valid_patches) >= max_patches:
                    break
                y, x = divmod(idx, padded_W)
                candidate_patch = padded_numpy[y:y + patch_size, x:x + patch_size]
                if candidate_patch.shape == (patch_size, patch_size):
                    valid_patches.append(candidate_patch)

            # Only append if we have enough patches
            if len(valid_patches) == max_patches:
                # Convert matched patches to CuPy and store
                blocks.append(cp.array(valid_patches))
            else:
                # If not enough valid matches, you can choose to skip or
                # handle differently. For now, let's skip adding.
                continue

    return blocks

########################################
# Global Wavelet Thresholding
########################################
def global_wavelet_thresholding(blocks, wavelet='db1', level=3, threshold=0.2):
    """
    Perform global wavelet thresholding on stacked blocks.
    """
    # Ensure that `blocks` is not empty and all blocks have consistent shape
    if len(blocks) == 0:
        raise ValueError("No blocks found for denoising. Check block matching logic.")

    stacked_blocks = cp.stack(blocks, axis=0)  # Create a 4D array: (N_blocks, max_patches, patch_size, patch_size)
    volume_numpy = cp.asnumpy(stacked_blocks)  # Convert to NumPy for wavelet transform
    denoised_volume = denoise_wavelet(
        volume_numpy, wavelet=wavelet, mode='soft', wavelet_levels=level,
        multichannel=False, rescale_sigma=True
    )
    return cp.array(denoised_volume)

########################################
# Second Stage Wiener Filtering
########################################
def wiener_filter_stage2(basic_image, noisy_image, patch_size, search_radius):
    """
    Apply Wiener filtering in the transform domain.
    """
    H, W = noisy_image.shape
    basic_padded = cp.pad(basic_image, search_radius, mode='reflect')
    noisy_padded = cp.pad(noisy_image, search_radius, mode='reflect')

    denoised_image = cp.zeros_like(noisy_image)
    weight_map = cp.zeros_like(noisy_image)

    for i in range(H):
        for j in range(W):
            ref_basic_patch = basic_padded[i:i + patch_size, j:j + patch_size]
            ref_noisy_patch = noisy_padded[i:i + patch_size, j:j + patch_size]

            # Ensure both patches have the correct shape
            if ref_basic_patch.shape != (patch_size, patch_size) or ref_noisy_patch.shape != (patch_size, patch_size):
                continue

            # Compute Wiener weights
            numerator = cp.fft.fft2(ref_basic_patch) * cp.fft.fft2(ref_noisy_patch)
            denominator = cp.abs(cp.fft.fft2(ref_basic_patch)) ** 2 + 1e-8
            wiener_patch = numerator / denominator
            denoised_patch = cp.fft.ifft2(wiener_patch).real

            denoised_image[i:i + patch_size, j:j + patch_size] += denoised_patch
            weight_map[i:i + patch_size, j:j + patch_size] += 1

    # Avoid division by zero
    mask = weight_map == 0
    weight_map[mask] = 1
    denoised_image[mask] = noisy_image[mask]  # fallback to original noisy pixels where no weights

    return denoised_image / weight_map

########################################
# Full G-BM3D Algorithm
########################################
def gbm3d(noisy_image, patch_size=8, search_radius=16, wavelet='db1', level=3, threshold=0.2):
    """
    Full G-BM3D algorithm.
    """
    # Stage 1: Global Wavelet Thresholding
    print("Starting Stage 1: Global Wavelet Thresholding...")
    matched_blocks = fft_block_matching(noisy_image, patch_size, search_radius)
    basic_estimate = global_wavelet_thresholding(matched_blocks, wavelet, level, threshold)

    # After global thresholding, the shape of basic_estimate is expected to be:
    # (Number_of_reference_positions, max_patches, patch_size, patch_size)
    # You might need additional aggregation here if desired.
    # For simplicity, assume you want to just take the first block from stage 1 as your "basic image".
    # In a real BM3D algorithm, you would aggregate these patches to form the basic estimate image.
    # Here, we will assume a simplistic approach:
    # Average all the first patches in each block along the reference dimension as a basic estimate.
    basic_image = cp.mean(basic_estimate[:, 0, :, :], axis=0)

    # Stage 2: Wiener Filtering
    print("Starting Stage 2: Wiener Filtering...")
    final_estimate = wiener_filter_stage2(basic_image, noisy_image, patch_size, search_radius)

    return final_estimate

########################################
# Tiling for Large Images
########################################
def process_image_tiling(image, tile_size, patch_size, search_radius, wavelet, level, threshold):
    """
    Process the image in tiles to handle large data sizes.
    """
    H, W = image.shape
    denoised_image = cp.zeros_like(image)

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            # Handle small edge tiles
            tile_h, tile_w = tile.shape
            if tile_h < patch_size or tile_w < patch_size:
                # If the tile is too small to process, just copy it.
                denoised_image[i:i + tile_size, j:j + tile_size] = tile
                continue

            denoised_tile = gbm3d(tile, patch_size, search_radius, wavelet, level, threshold)
            denoised_image[i:i + tile_size, j:j + tile_size] = denoised_tile

    return denoised_image

########################################
# Main Script
########################################
def main(input_path, output_path, tile_size=512, patch_size=8, search_radius=16, wavelet='db1', level=3, threshold=0.2):
    """
    Main function to run G-BM3D on an input image.
    """
    print("Loading noisy image...")
    noisy_image = load_image(input_path)

    print("Processing image...")
    denoised_image = process_image_tiling(
        noisy_image, tile_size, patch_size, search_radius, wavelet, level, threshold
    )

    print("Saving denoised image...")
    save_image(denoised_image, output_path)
    print("Denoising complete!")

if __name__ == "__main__":
    input_image_path = '/content/drive/MyDrive/sample.tif' 
    output_image_path = '/content/drive/MyDrive/sample_op.tif' 
    main(input_image_path, output_image_path)
