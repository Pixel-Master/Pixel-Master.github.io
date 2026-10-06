"""
This file provides a basic PRNU-Toolkit including noise extraction, computation and comparison with a reference noise

Copyright 2026

This software is licensed under the 'GPLv3' License as described in the 'LICENSE' file,
which should be included with this package. The terms are also available at
http://www.gnu.org/licenses/gpl-3.0.html

Usage example:

import prnu
import os

training_data_folder = '/Users/...'
test_file = '/Users/.../image.png'

avg = prnu.AverageNoise()
for file_path in os.path.listdir(training_data_folder):
    extractor = prnu.NoiseExtractor(os.path.join(training_data_folder, file_path))
    extractor.extract_noise(disable_zero_mean=True)
    avg.add_image_data(extractor)
avg.analyse()
extractor = prnu.NoiseExtractor(test_file)
extractor.extract_noise()
print(f'Correlation: {prnu.Compare(extractor, avg)}')
"""

# Imports
import cv2
import numpy as np
import os
import scipy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="PRNU-kit [%(pathname)s] at %(asctime)s, %(levelname)s: %(message)s",
                    force=True)


# This is the main class for PRNU-noise extraction
class NoiseExtractor:
    """
    This initialises the main class for PRNU-noise extraction.


    image_path      A path to an image file; can be png, jpeg, tiff
    image           If a path isn't provided the image can be inputted as a numpy.ndarray array of dimension 2
    disable_crop    Activate this option if you don't want the image to be cropped automatically to
                    512x512 in the middle. You can still crop manually by calling .crop() at any time

    """

    def __init__(self, image_path: str | None = None, image: np.ndarray | None = None, disable_crop=False):
        # Class-wide definitions
        # Default crop size
        self.crop_size = 512
        # This variable is used if the user makes the noise visible to prevent this option from being used twice and
        # to save the noise as a .png if it has been made visual
        self.visual = False
        # Variable to store the denoised image version
        self.denoised = None
        # Variable to store the image residual W
        self.noise_residual = None

        # Used to later store the noise in the same place, if the user calls write_to_file()
        if image_path is not None:
            self.file_name = os.path.basename(image_path)
            self.folder_name = os.path.dirname(image_path)
        else:
            # If no path was provided
            self.file_name = None
            self.folder_name = None

        # Loading image
        if image_path is None and image is None:
            # Wrong arguments used
            raise TypeError("There was neither an image path nor an np.ndarray provided in the args")
        # Image was supplied as a file path
        elif type(image_path) is str:
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Image was already supplied in memory as argument
        elif image is not None:
            self.image: np.ndarray = image

        # Test if image was loaded correctly
        if self.image is None:
            raise ReferenceError(f"{image_path} is not an image or couldn't be loaded correctly")

        # Automatically crop to save compute if not disabled by user
        if not disable_crop:
            self.crop()

    def extract_noise(self, disable_zero_meaning=False) -> None:
        """
        Extract the PRNU noise


        disable_zero_meaning    Usage of this option is recommended if shift_to_visual() is later executed or
                                if the noise of different images is later combined with AverageNoise
                                Use to increase correlation between different cameras of the same model
        """
        # Convert to float from 0-1 for more precise calculation and as to not cause overflows
        self.image = self.image.astype(np.float64) / 255

        # Denoise with Wiener and kernel size of (3, 3) because it has been found that this
        # is the optimal value for capturing as much noise as possible
        # while capturing as little image content as possible.
        # 3 is also the smallest possible value as according to the SciPy docs
        # mysize should be odd and 1 would not make it denoise
        self.denoised = scipy.signal.wiener(self.image, mysize=(3, 3))

        # Replace NaN values with 0
        self.denoised = np.nan_to_num(self.denoised)

        # Extract residual noise, described as W = Img_out - denoise(Img_out) in the paper
        self.noise_residual = self.image - self.denoised
        # Avoiding division by zero
        epsilon = 1e-10
        # Extract PRNU noise as PRNU is multiplicative noise. Described as F = W/denoise(img_out)
        self.image = self.noise_residual / (self.denoised + epsilon)

        # Might also be useful when detecting for the same model instead of the same individual sensor
        if not disable_zero_meaning:
            self.image = self.zero_meaning(self.image)

    def crop(self) -> None:
        """
        Crop the image 512x512 in the middle to save compute time and prevent peripheral illumination.
        Normally it's unnecessary to call this function as if not disabled by the user, it's called automatically.

        Change NoiseExtractor.crop_size to change the area being cropped
        """
        h, w = self.image.shape
        # Calculate starting points for center crop. max() if image is smaller than 512x512
        start_y = max(0, h // 2 - self.crop_size // 2)
        start_x = max(0, w // 2 - self.crop_size // 2)
        # Crop the image in the middle
        self.image = self.image[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]

    @staticmethod
    def zero_meaning(image: np.ndarray) -> np.ndarray:
        """
        Zero-mean the image.
        Takes the average of every row and column and removes it.
        This is done to reduce the correlation between cameras of the same model and to improve visibility


        image   a two-dimensional numpy.ndarray containing the image
        """
        # Remove average row value
        # keepdims is used to ensure the number of dimensions of image don't change
        row_means = np.mean(image, axis=1, keepdims=True)
        image = image - row_means
        # Do the same with the columns
        col_means = np.mean(image, axis=0, keepdims=True)
        image = image - col_means

        return image

    def shift_to_visual(self) -> None:
        """
        This makes the noise visible by shifting it in a way that makes the difference between the pixels large enough
        to see with the naked eye. Before executing the pixel values are extremely tiny decimal values
         and saving as an image would just return a completely black image.

        Can only be executed once per image and cannot be reversed as the very subtle differences that characterise PRNU
        get largely lost.
        """
        # Make sure to not make the noise "visible" twice
        if self.visual:
            return
        self.visual = True
        # Compute the standard deviation and average
        sigma = np.std(self.image)
        mean_val = np.mean(self.image)

        # If the standard deviation is 0, all values are the same.
        if sigma == 0:
            # This is done to prevent a division by zero
            self.image = np.zeros_like(self.image)
        else:
            # It is assumed that because of the zero meaning the overall average is 0
            # We clip the noise in a sensible way to prevent overflows.
            noise_clipped = np.clip(self.image, mean_val - 3 * sigma, mean_val + 3 * sigma)
            # The clipped noise is now being scaled to between 0 and 1 with 3 * sigma scaling to 1
            self.image = (noise_clipped - (mean_val - 3 * sigma)) / (6 * sigma)

        # Scale to 8-bit integer and image range (0-255)
        self.image = (self.image * 255).astype(np.uint8)

    def write_to_file(self, output: None | str = None) -> None:
        """
        Saves the computed noise to the file system.
        If the image was shifted to visual it is saved as a .png,
        if not as a numpy file .npy


        output  The full file path including the file name itself
                Required if the image was given directly as a numpy ndarray
                if not provided, the image will be saved in a folder,
                the same place as the original image folder but with a '_noise' attachment added to it.


        If the image hasn't been made visual, the noise residual and denoised original image will all be saved
        as they are needed for computing the correlation between a reference and a test image

        """
        # If the file name and image_path is known, the file is saved in a folder with a _noise added to it
        if self.file_name is not None and output is None:
            output = os.path.join(self.folder_name + "_noise", self.file_name)
            os.makedirs(self.folder_name + "_noise", exist_ok=True)
        elif output is None:
            # Image was given directly as a numpy ndarray.
            raise ValueError("Requires output dir or file image_path of input file")

        # Differentiate between "visual" images and data format
        if self.visual:
            # Image has been made visual and can therefore be saved as a png
            cv2.imwrite(output, self.image)
            logging.debug("Wrote to file:" + output)
        else:
            #
            np.save(output + "noise" + ".npy", self.image)
            np.save(output + "residual" + ".npy", self.noise_residual)
            np.save(output + "denoised" + ".npy", self.denoised)
            logging.debug("Wrote to file:" + output + ".npy")

    @staticmethod
    # Analyse all files in this folder and any subfolder
    def multiple(
            folder: str, output_dir: str = None, disable_crop: bool = False, shift_to_visual: bool = False,
            disable_zero_meaning: bool = False) -> str:
        """
        Convenience function that automatically analyses all files in any given folder and its subfolders
        and saves the result in the same place as the original folder but with a _noise added


        folder                  The folder whose images and subfolders are analysed.
        output_dir              The folder all computed noise fingerprints are placed into
        disable_crop            Don't crop the images 512x512 in the center
        shift_to_visual         Makes the noise visual and saves just one png instead of three .npy
        disable_zero_meaning    Don't remove the averages of every row and column, recommended if shift_to_visual=True
                                or if AverageNoise class is supposed to be used

        Returns the output directory
        """
        # Save the computed noise in a folder in the same place but with a "_noise" added to the end
        # provided no output_dir given
        if output_dir is None:
            output_dir = folder + "_noise"
        # Create this directory
        os.makedirs(output_dir, exist_ok=True)
        # Traverse the folder and every single subfolder
        for root_folder, _subfolder, file_paths in os.walk(folder):
            # Iterate through all files.
            for file_path in file_paths:
                try:
                    # Load the image into the extractor
                    extractor = NoiseExtractor(image_path=os.path.join(root_folder, file_path),
                                               disable_crop=disable_crop)
                except ReferenceError:
                    # File is not an image
                    continue
                # Extract the noise
                extractor.extract_noise(disable_zero_meaning=disable_zero_meaning)
                if shift_to_visual:
                    extractor.shift_to_visual()
                # Save the file
                extractor.write_to_file(os.path.join(output_dir, file_path))
        # Returns the output directory
        return output_dir


class AverageNoise:
    """
    Add the noise patterns of multiple images (of one camera) together and compute the average.
    Afterward the computed average is intended to be used as a reference with the Compare class.
    """

    def __init__(self):
        # Define variables that are only known later
        self.count = 0
        self.dimensions = None
        # Placeholder dimensions, exact shape is only known later.
        self.sum_noise = None
        self.image = None
        self.visual = False
        # denoised^2
        self.sum_denoised2 = None
        # The sum of both multiplied together
        self.sum_residual_denoised = None

    def add_image_data(self, noise_extractor: NoiseExtractor | None = None,
                       denoised: np.ndarray | None = None, residual: np.ndarray | None = None) -> None:
        """
        Add image as two np.ndarray or one NoiseExtractor

        Either a NoiseExtractor needs to be supplied or
        the computed denoised version together with the noise residual both as a two-dimensional numpy.ndarray

        Note: All added images should have the same dimension, else the image is disregarded.


        noise_extractor A NoiseExtractor class with an image loaded and extract_noise() called.
        denoised        Two-dimensional numpy.ndarray as an output of the NoiseExtractor class
        residual        Two-dimensional numpy.ndarray as an output of the NoiseExtractor class
        """
        # If the NoiseExtractor was supplied directly
        if noise_extractor is not None:
            denoised = noise_extractor.denoised
            residual = noise_extractor.noise_residual

        # Figuring out dimensions when executing the first time by just taking the shape of the denoised array.
        if self.dimensions is None:
            self.dimensions = denoised.shape
            # Setting both matrices to the correct dimensions and filling both with placeholder zeros
            self.sum_residual_denoised = np.zeros(self.dimensions, dtype=np.float64)
            self.sum_denoised2 = np.zeros(self.dimensions, dtype=np.float64)
        # If the dimension is different the average cannot be computed because the matrices don't overlap properly
        elif denoised.shape != self.dimensions:
            logging.error(f"Wrong dimensions {denoised.shape}, {self.dimensions} required")
            return

        # Summing up according to the formula 3.6
        self.sum_denoised2 += denoised ** 2
        self.sum_residual_denoised += residual * denoised

    # Add image from path
    def add_file(self, denoised_file_path: str, residual_file_path: str) -> None:
        """
        Add image as a file path.

        The paths to the residual and the denoised version both have to be supplied

        Note: All added images should have the same dimension, else the image is disregarded.


        denoised_file_path  Path to a denoised image as computed and saved by the NoiseExtractor file
        residual_file_path  Path to a residual as computed and saved by the NoiseExtractor file
        """
        # Load the matrices with numpy
        residual = np.load(residual_file_path)
        denoised = np.load(denoised_file_path)

        # If loading the image failed
        if residual is None or denoised is None:
            return
        # Add the loaded image data
        self.add_image_data(residual=residual, denoised=denoised)

    def add_folder(self, folder: str) -> None:
        """
        Add multiple images as a folder

        The folder must contain both the residual and denoised version as outputted by the NoiseExtractor class
        especially with the file name unchanged

        Note: All added images should have the same dimension, else the image is disregarded.


        folder  Path to a folder containing the residuals and denoised versions as computed by the NoiseExtractor class.
        """
        logging.info(f"Start combining from {folder}")
        for denoised_file_path in os.listdir(folder):
            # There is one "residual" and one "denoised" file. Both are needed corresponding to the same original image
            # To not run the function twice skip if the ending is "residual.npy"
            # as there has to be another file ending with "denoised.npy" corresponding to the same original image
            if not denoised_file_path.endswith("residual.npy"):
                continue
            self.add_file(os.path.join(folder, denoised_file_path),
                          os.path.join(folder, (denoised_file_path.removesuffix("denoised.npy") + "residual.npy")))

    def analyse(self, disable_zero_meaning=False) -> None:
        """
        Computes the average noise for an image according to the formula 3.6.
        Only call this once after all images have been added.


        disable_zero_meaning    Set true to increase correlations between different cameras of the same model.
        """
        # Formula from chapter 3.2 F = sum(W^n*denoised(Img_out)) / sum(denoised(Img_out)^2)
        epsilon = 1e-10  # Prevents division by zero
        self.image: np.ndarray = self.sum_residual_denoised / (self.sum_denoised2 + epsilon)

        # Do zero meaning now
        if not disable_zero_meaning:
            self.image = NoiseExtractor.zero_meaning(self.image)

    def shift_to_visual(self) -> None:
        """
        Makes the combined noise pattern visible to the naked eye. Same as NoiseExtractor.shift_to_visual()
        Cannot be reversed, only call this once per instance
        """
        # Make sure to not make the noise "visible" twice
        if self.visual:
            return
        self.visual = True
        # Compute the standard deviation and average
        sigma = np.std(self.image)
        mean_val = np.mean(self.image)

        # If the standard deviation is 0, all values are the same.
        if sigma == 0:
            # This is done to prevent a division by zero
            self.image = np.zeros_like(self.image)
        else:
            # It is assumed that because of the zero meaning the overall average is 0
            # We clip the noise in a sensible way to prevent overflows.
            noise_clipped = np.clip(self.image, mean_val - 3 * sigma, mean_val + 3 * sigma)
            # The clipped noise is now being scaled to between 0 and 1 with 3 * sigma scaling to 1
            self.image = (noise_clipped - (mean_val - 3 * sigma)) / (6 * sigma)

        # Scale to 8-bit integer and image range (0-255)
        self.image = (self.image * 255).astype(np.uint8)

    def write_to_file(self, output_dir: str):
        """
        Save the combined noise on the disk. If the noise has been made visual it will be saved as "combined.png",
        else it will be saved as "combined.npy" so it can be loaded again at a later time.


        output_dir  The directory where the file will be saved. Do not include the file name itself.


        Returns the path to the saved file
        """
        if self.visual:
            cv2.imwrite(os.path.join(output_dir, "combined.png"), self.image)
            logging.debug(f"Saved to {os.path.join(output_dir, 'combined.png')}")
        else:
            np.save(os.path.join(output_dir, "combined.npy"), self.image)
            logging.debug(f"Saved to {os.path.join(output_dir, 'combined.npy')}")

        return os.path.join(output_dir, "combined.npy")


class Compare:
    """
    Compares an average of multiple images that has been computed with AverageNoise to
    a single test image which has been computed with NoiseExtractor


    image       Test image that is compared to the reference, can either be supplied as a NoiseExtractor class.
                or the stripped path ending with .png or .jpeg to the
                residual and denoised version that both end with .npy
    reference   Average noise reference as computed with the AverageNoise class. Can either be an AverageNoise class
                or a direct path to the computed reference ending with .npy
    """

    def __init__(self, image: str | NoiseExtractor, reference: AverageNoise | str) -> None:
        # Load reference image
        # Loading reference from disk
        if type(reference) == str and reference.endswith(".npy"):
            self.reference = np.load(reference)
        # Supplied in memory as matrix
        elif type(reference) == AverageNoise:
            self.reference = reference.image
        else:
            raise TypeError(f"{reference} isn't the required type")

        # Load compare image
        # from disk
        if type(image) == str:
            self.image_denoised = np.load(image + "denoised.npy")
            self.image_residual = np.load(image + "residual.npy")
        # From memory
        elif type(image) == NoiseExtractor:
            self.image_denoised = image.denoised
            self.image_residual = image.noise_residual
        else:
            raise TypeError(f"{image} isn't the required type")
        # Placeholder value for the correlation score
        self.correlation_score = None

    def correlate(self) -> float:
        """
        Correlates the test image with the reference. Returns the correlation score,
        generally it has been experimentally found that above a value of 0.008 the two images can be
        assumed to be taken with the same camera.
        """
        # Make sure dimensions are the same.
        if self.reference.shape != self.image_residual.shape:
            logging.error(
                f"Dimension mismatch between reference ({self.reference.shape}) "
                f"and query image ({self.image_residual.shape}).")
            return 0

        # Modulate reference according to formula 3.8
        modulated_reference = self.reference * self.image_denoised

        # Flatten both matrices into one dimensional arrays
        ref_flat = modulated_reference.flatten()
        query_flat = self.image_residual.flatten()

        # Calculates Pearson correlation coefficient (PCE would be better but is a lot more complex)
        # numpy.corrcoef returns a 2x2 matrix. The correlation between the reference and test image is at [0,1]
        correlation_matrix = np.corrcoef(ref_flat, query_flat)
        self.correlation_score = correlation_matrix[0, 1]

        return self.correlation_score
