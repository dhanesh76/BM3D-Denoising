import cv2
import numpy as np
import bm3d
from concurrent.futures import ProcessPoolExecutor
from google.colab.patches import cv2_imshow  # For displaying images in Google Colab

def read_image(input_path):
    """Read image using OpenCV (supports JPG, PNG, etc.)."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Grayscale only for BM3D
    if img is None:
        raise ValueError(f"Image not found at {input_path}")
    return img

def tile_image(image, block_size):
    """Split image into tiles."""
    h, w = image.shape
    tiles = []
    positions = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            tile = image[i:i+block_size, j:j+block_size]
            if tile.shape[0] != block_size or tile.shape[1] != block_size:
                pad_h = block_size - tile.shape[0]
                pad_w = block_size - tile.shape[1]
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), 'reflect')
            tiles.append(tile)
            positions.append((i, j))
    return tiles, positions

def process_tile(tile, sigma_psd=0.2):
    """Denoise tile using BM3D."""
    tile = tile.astype(np.float32) / 255.0
    denoised_tile = bm3d.bm3d(tile, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    denoised_tile = (denoised_tile * 255).astype(np.uint8)
    return denoised_tile

def reassemble_image(tiles, positions, image_shape, block_size):
    """Reassemble tiles back into a single image."""
    h, w = image_shape
    final_image = np.zeros((h, w), dtype=np.uint8)
    for tile, (i, j) in zip(tiles, positions):
        tile_h, tile_w = tile.shape
        final_image[i:i+tile_h, j:j+tile_w] = tile[:tile_h, :tile_w]
    return final_image

def process_large_image(input_path, block_size=256, sigma_psd=0.2, n_jobs=4):
    """Process the image using multi-core CPU BM3D processing."""
    print(f"Reading image from {input_path}...")
    image = read_image(input_path)
    print(f"Image size: {image.shape}, Block size: {block_size}")

    tiles, positions = tile_image(image, block_size=block_size)
    print(f"Total tiles to process: {len(tiles)}")

    print(f"Processing {len(tiles)} tiles in parallel using {n_jobs} CPU cores...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        denoised_tiles = list(executor.map(process_tile, tiles))
        # Show progress for each tile
        for idx, _ in enumerate(denoised_tiles, start=1):
            print(f"Tile {idx}/{len(tiles)} has been processed.")

    print(f"Reassembling image from tiles...")
    denoised_image = reassemble_image(denoised_tiles, positions, image.shape, block_size)

    # Save the denoised image to a file
    output_path = "/content/drive/MyDrive/sampleOP.tif"  # Path to save the denoised image
    cv2.imwrite(output_path, denoised_image)
    print(f"Denoised image saved to {output_path}")

    # Display the image in Colab (using cv2_imshow)
    cv2_imshow(denoised_image)

    return output_path  # Return the path to the saved image

if __name__ == "__main__":
    input_path = '/content/drive/MyDrive/sample.tif'  # Input image path
    block_size = 1024
    sigma_psd = 0.2
    n_jobs = 8  # Number of CPU cores to use
    output_path = process_large_image(input_path, block_size, sigma_psd, n_jobs)
