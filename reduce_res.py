from PIL import Image
import rawpy
import os

def compress_and_crop_image(image_path, output_path, target_size=(1280, 1280), quality=20):
    """
    Compress and trim an image to a square (by trimming excess from one side), then resize to the target size.

    Parameters:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the compressed and cropped image.
        target_size (tuple): Target dimensions (width, height). Default is (512, 512).
        quality (int): Quality of the output image (1-95, lower means higher compression). Default is 10.
    """
    try:
        # Check if the file is a RAW file
        if image_path.lower().endswith(('.nef', '.cr2', '.arw', '.dng', '.raw')):
            # Process RAW file with rawpy
            with rawpy.imread(image_path) as raw:
                img_data = raw.postprocess()
                img = Image.fromarray(img_data)
        else:
            # Open the image for other formats
            img = Image.open(image_path)

        # Ensure the image has an alpha channel for transparency handling
        img = img.convert("RGB") if img.mode != "RGB" else img

        # Get image dimensions
        width, height = img.size

        # Determine the size of the square to trim (smallest dimension)
        square_size = min(width, height)

        # Calculate the coordinates for trimming the image to square
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        # Trim the image to a square
        img_trimmed = img.crop((left, top, right, bottom))

        # Resize to target dimensions (512x512)
        img_resized = img_trimmed.resize(target_size, Image.Resampling.LANCZOS)

        # Save with high compression
        img_resized.save(output_path, "JPEG", quality=quality, optimize=True)

        #print(f"Image saved to {output_path} with size {target_size} and quality {quality}.")

    except Exception as e:
        print(f"Error processing the image: {e}")

# Example usage
if __name__ == "__main__":
    input_path = "/Volumes/homes/FrogNAS/lightroom/2022/2022-05-15/DSC_7574.NEF" #input("gib image path: ")
    output = "test_small.jpg"
    compress_and_crop_image(input_path.strip(), output)
