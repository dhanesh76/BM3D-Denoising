import cv2
import numpy as np
import bm3d
from concurrent.futures import ProcessPoolExecutor
import time
import os


def read_image(input_path):
    print(f"Opening image from {input_path}...")
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {input_path}")
    else:
        import rasterio
        with rasterio.open(input_path) as src:
            img = src.read(1)
    print(f"Image dimensions: {img.shape}")
    return img


def process_tile(tile, tile_index, total_tiles, sigma_psd=0.2):
    print(f"Processing tile {tile_index + 1} out of {total_tiles}...")
    try:
        tile = tile.astype(np.float32) / 255.0
        denoised_tile = bm3d.bm3d(tile, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        return (denoised_tile * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in processing tile: {e}")
        return tile


def process_tile_with_index(args):
    """Helper function to unpack arguments and call process_tile"""
    tile, idx, total_tiles, sigma_psd = args
    return process_tile(tile, idx, total_tiles, sigma_psd)


def tile_image(image, block_size):
    print("Starting tile-based processing...")
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
            print(f"Reading tile at position: ({i}, {j})...")

    print(f"Total tiles to process: {len(tiles)}")
    return tiles, positions


def reassemble_image(tiles, positions, image_shape, block_size):
    print("Reassembling tiles into a single image...")
    h, w = image_shape
    final_image = np.zeros((h, w), dtype=np.uint8)
    for tile, (i, j) in zip(tiles, positions):
        tile_h, tile_w = tile.shape
        final_image[i:i+tile_h, j:j+tile_w] = tile[:tile_h, :tile_w]
    return final_image


def process_large_image(input_path, output_image_path, output_time_path, block_size=256, sigma_psd=0.2, n_jobs=2):
    print(f"Opening image from {input_path}...")
    start_time = time.time()

    # Step 1: Read image
    image = read_image(input_path)

    # Step 2: Split the image into tiles
    tiles, positions = tile_image(image, block_size=block_size)

    # Step 3: Process each tile in parallel (use ProcessPoolExecutor for multi-process execution)
    print(f"Processing tiles in parallel using {n_jobs} jobs...")
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            denoised_tiles = list(executor.map(
                process_tile_with_index, 
                [(tile, idx, len(tiles), sigma_psd) for idx, tile in enumerate(tiles)]
            ))
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        return None

    # Step 4: Reassemble the image from the denoised tiles
    denoised_image = reassemble_image(denoised_tiles, positions, image.shape, block_size=block_size)
    
    # Save the output image
    cv2.imwrite(output_image_path, denoised_image)
    print(f"Denoised image saved at {output_image_path}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    # Save the total processing time to a file
    with open(output_time_path, 'w') as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
    print(f"Processing time saved at {output_time_path}")
    
    return denoised_image


if __name__ == "__main__":
    input_path = r"/content/drive/MyDrive/NoisySampleImage.jpg"  # Input image path
    output_image_path = r"/content/drive/MyDrive/DenoisedImage.jpg"  # Where to save the output image
    output_time_path = r"/content/drive/MyDrive/ProcessingTime.txt"  # Where to save the processing time

    block_size = 256  # Tile size
    sigma_psd = 0.2  # Noise standard deviation (adapt to the image's noise level)
    n_jobs = 4  # Number of parallel jobs

    denoised_img = process_large_image(input_path, output_image_path, output_time_path, block_size=block_size, sigma_psd=sigma_psd, n_jobs=n_jobs)
