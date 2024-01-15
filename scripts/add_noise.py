import cv2
import numpy as np


def add_gaussian_noise(image, noise_level):
    # Generate Gaussian noise with zero mean and specified standard deviation
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

    return noisy_image


def main():
    # Specify the file path of the image
    image_path = "/mnt/ito/diffusion-anomaly/BraTS20_Training_336_t1ce.png"

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Failed to load the image.")
        return

    # Specify the noise level
    noise_level = 2

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(image, noise_level)

    # Save the noisy image
    noisy_image_path = "noise_img.png"
    cv2.imwrite(noisy_image_path, noisy_image)

    print("Noisy image saved successfully.")


if __name__ == "__main__":
    main()
