from skimage import color, img_as_float32
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import bm3d
import rasterio
import matplotlib.pyplot as plt

def process_tile(tile, sigma_psd):
    """Denoise a single tile using BM3D."""
    print("Processing a tile...")
    return bm3d.bm3d(tile, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

def process_large_image(file_path, block_size=512, sigma_psd=0.2, n_jobs=2):
    """Process a large image using BM3D and display the result without saving."""
    print(f"Opening image from {file_path}...")

    # Open the large image using rasterio for memory-efficient handling
    with rasterio.open(file_path) as src:
        print("Reading metadata...")
        profile = src.profile
        height, width = src.height, src.width
        count = src.count  # Number of bands

        # Use a numpy array to store the final result
        denoised_image = np.zeros((height, width), dtype='float32')

        print("Starting tile-based processing...")
        tiles = []
        coords = []

        # Divide the image into tiles
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                print(f"Reading tile at position: ({i}, {j})...")
                
                # Read the tile depending on the number of bands
                if count == 1:
                    # Grayscale image, read the first band
                    tile = src.read(1, window=rasterio.windows.Window(j, i, block_size, block_size))
                    tile_gray = tile.astype(np.float32)  # Convert to float32 for processing
                elif count >= 3:
                    # RGB image, read the first three bands
                    tile = src.read([1, 2, 3], window=rasterio.windows.Window(j, i, block_size, block_size))
                    tile_rgb = np.moveaxis(tile, 0, -1)  # Move the bands to the last axis for RGB format
                    tile_gray = color.rgb2gray(tile_rgb)  # Convert RGB to grayscale
                tiles.append(tile_gray)
                coords.append((i, j))

        print(f"Total tiles to process: {len(tiles)}")

        # Process tiles in parallel with ProcessPoolExecutor
        print(f"Processing tiles in parallel using {n_jobs} jobs...")
        denoised_tiles = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_tile, tile, sigma_psd) for tile in tiles]
            for future in as_completed(futures):
                denoised_tiles.append(future.result())

        # Reassemble the image
        print("Reassembling the denoised tiles...")
        for (i, j), denoised_tile in zip(coords, denoised_tiles):
            denoised_image[i:i+block_size, j:j+block_size] = denoised_tile

    print("Denoising completed!")
    return denoised_image

# Example usage
input_path = r"D:\SIH_2024\denoising\south pole 1.tif"

# Process the image
denoised_img = process_large_image(input_path, block_size=256, sigma_psd=0.2, n_jobs=2)

# Display the denoised image
plt.figure(figsize=(10, 5))
plt.title("Denoised Image")
plt.imshow(denoised_img, cmap='gray')
plt.axis('off')
plt.show()
