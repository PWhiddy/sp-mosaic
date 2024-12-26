import os
from collections import defaultdict
import time
from multiprocessing import Pool, cpu_count, Process
from datetime import datetime
from tqdm import tqdm 

from reduce_res import compress_and_crop_image

def is_image_file_by_extension(file_name):
    """
    Determine if a file is an image based on its extension.
    """
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
        ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2", ".nef", ".arw", ".orf", ".rw2"
    }
    return any(file_name.lower().endswith(ext) for ext in image_extensions)

def process_image(file_path, output_dir):
    """
    Compress and crop an image, given its path and output directory.
    Output filenames will include a timestamp to ensure uniqueness.
    """
    try:
        # Get the timestamp from the file's last modified time
        timestamp = os.path.getmtime(file_path)
        timestamp_str = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        
        # Use the timestamp and base filename to create a unique output filename
        output_filename = f"{timestamp_str}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Compress and crop the image
        compress_and_crop_image(file_path, output_path, target_size=(1280, 1280), quality=20)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def monitor_output_directory(output_dir, interval=3):
    """
    Monitor the output directory and print the rate of files being processed per second.
    """
    start_time = time.time()

    progress_bar = tqdm(desc="Processing files ")

    while True:
        time.sleep(interval)
        current_count = len(os.listdir(output_dir))
        elapsed_time = time.time() - start_time
        rate = current_count / elapsed_time  # images per second

        progress_bar.set_description(
            f"Current count {current_count}, Total rate: {rate:.3f} images / second",
                  True  
        )

    progress_bar.close()

def search_images(directory, output_dir, num_processes=10):
    """
    Recursively search for image files in the specified directory and count each image type.
    Use multiprocessing to handle image processing tasks.
    """

    image_counts = defaultdict(int)
    total_files = 0
    total_images = 0
    non_image_files = 0
    error_files = 0
    all_file_type_counts = defaultdict(int)

    start_time = time.time()

    progress_bar = tqdm(desc="Finding files ", unit="file")
    file_jobs = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(directory):
        for file in files:
            try:
                total_files += 1
                file_path = os.path.join(root, file)
                if is_image_file_by_extension(file):
                    image_counts[os.path.splitext(file)[1].lower()] += 1
                    total_images += 1
                    file_jobs.append(file_path)  # Add image file for processing
                else:
                    non_image_files += 1
                all_file_type_counts[os.path.splitext(file)[1].lower()] += 1
            except Exception as e:
                error_files += 1

            progress_bar.set_postfix({
                "Total Files": total_files,
                "Image Files": total_images,
                "Non-Image Files": non_image_files,
                "Error Files": error_files,
            })
            progress_bar.update(1)

    progress_bar.close()

    # Start the directory monitoring process
    monitor_process = Process(target=monitor_output_directory, args=(output_dir,))
    monitor_process.daemon = True  # Allow it to exit when the main program exits
    monitor_process.start()

    # Use multiprocessing to handle the compression and cropping tasks in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_image, [(file_path, output_dir) for file_path in file_jobs])

    elapsed_time = time.time() - start_time
    print(dict(all_file_type_counts))
    print(f"Finished searching. Found {total_images} images out of {total_files} in {elapsed_time:.2f} seconds.")
    monitor_process.terminate()
    return dict(image_counts)

if __name__ == "__main__":
    directory_to_search = "/Volumes/home/lightroom"  # Specify the directory to search
    output_dir = "reduced_images"  # Specify the output directory

    if not os.path.isdir(directory_to_search):
        print("The specified path is not a valid directory.")
    else:
        search_images(directory_to_search, output_dir, num_processes=20)
