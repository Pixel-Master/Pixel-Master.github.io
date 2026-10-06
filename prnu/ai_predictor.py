"""
This file provides the code for training and using the ResNet-18-based AI model using PRNU fingerprints
to predict if a given image is AI-generated

Copyright 2026

This software is licensed under the 'GPLv3' License as described in the 'LICENSE' file,
which should be included with this package. The terms are also available at
http://www.gnu.org/licenses/gpl-3.0.html

Usage example for predicting AI-probability of a given file:

import ai_predictor

file_path = '/.../../.png'

model = ai_predictor.Model("prnu_ai_predictor.pth")
model.load()
print(model.predict(file_path))

"""

# Imports
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from sklearn.utils import shuffle
# PRNU-Analysis
import prnu


def create_prnu_resnet18(num_classes: int = 2) -> nn.Module:
    """Creates a ResNet-18 model adapted for 1-channel PRNU inputs and 2 target classes."""
    model = resnet18(weights=None)
    # This definition is needed because the PRNU images don't have three color channels but only one luminance
    # This redefines the first layer according to the official ResNet-18 definition but with only one input channel
    # because PRNU fingerprints are grayscale and only have luminance instead of three colors
    model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replacing the last layer (fc=fully connected)
    # We only need a two-class classification system (AI or real) instead of the default 1000 that come with ResNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class Model:
    def __init__(self, model_path: str):
        """
        Main class for the ResNet-18-based AI detector model using PRNU fingerprints
        Use this both for training and predicting


        model_path  Path to the model .pth file
        """
        # Making parameter class-wide
        self.model_path = model_path
        # PRNU fingerprints are cropped to 512x512 by default in this module
        self.target_shape: tuple = (512, 512)
        # Defining both classes
        self.classes = np.array(["ai", "real"])
        self.label_map = {"ai": 0, "real": 1}
        # Choose hardware acceleration according to availability (Apple Silicon MPS / NVIDIA CUDA / CPU)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Used device: {self.device}")
        # Initializing ResNet-18, moving to hardware accelerated device
        self.model = create_prnu_resnet18(num_classes=len(self.classes)).to(self.device)
        # Setting up the loss function
        self.criterion = nn.CrossEntropyLoss()
        # Defining learning rate according to experimental results to 1*10^-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def extract_prnu(self, file_path: str) -> np.ndarray | None:
        """
        Wrapper for prnu.NoiseExtractor and standardisation of the resulting noise

        file_path
        """
        try:
            extractor = prnu.NoiseExtractor(image_path=file_path)
        except ReferenceError:
            # Loading the image failed
            return None
        extractor.extract_noise()
        noise = extractor.image
        if noise is None or noise.shape != self.target_shape:
            # Extraction failed or wrong dimension
            return None

        # Standardise with mean and standard deviation
        std = np.std(noise)
        if std > 1e-6:
            noise = (noise - np.mean(noise)) / std
        else:
            noise = noise - np.mean(noise)

        # Expanding dimensions as the noise is two-dimensional but the training requires more.
        return np.expand_dims(noise, axis=0).astype(np.float32)

    def train_incremental(self, image_paths: list, labels: list, batch_size: int = 32, epochs: int = 25) -> None:
        """
        Trains the model over multiple epochs and batches


        image_paths A list with all paths pointing to training data
        labels      A list with the classes (here only "ai" or "real") corresponding to the items in image_paths
                    Every item in this list has to correspond to the item with the same index in image_paths
        batch_size  When using larger sets of training data it has to be split into multiple batches.
                    Tune this according to your resources
        epochs      The amount of training cycles over the entire set.
                    Set this according to the amount of training data
        """

        # Iterating through all epochs
        for epoch in range(1, epochs + 1):
            logging.info(f"=== Starting epoch {epoch}/{epochs} ===")

            # Shuffle data after every epoch. Makes sure the index in labels_shuffled changed accordingly
            # 99 is just a random seed, + epoch makes sure there is a unique seed for every epoch
            paths_shuffled, labels_shuffled = shuffle(image_paths, labels, random_state=99 + epoch)

            # Set the model into training mode
            self.model.train()
            # Create two empty lists to store the noise and the corresponding category
            x_batch, y_batch = [], []
            total_items = len(paths_shuffled)

            # Iterating through every single file path
            for idx, (path, label) in enumerate(zip(paths_shuffled, labels_shuffled), start=1):
                # Extract noise and standardise
                noise = self.extract_prnu(path)

                if noise is None:
                    # If a problem occurred while extracting the noise
                    continue
                # Storing the noise in the X-dimension
                x_batch.append(noise)
                # Storing the corresponding category in the Y-dimension.
                y_batch.append(self.label_map[label])

                # If the item is the last, training begins
                is_last_item = (idx == total_items)

                # Execute if batch is full or last item of the training data
                if len(x_batch) == batch_size or (is_last_item and len(x_batch) > 0):
                    # Creating tensors and moving them to GPU
                    inputs = torch.tensor(np.array(x_batch), dtype=torch.float32).to(self.device)
                    # Doing the same with the classes
                    targets = torch.tensor(y_batch, dtype=torch.long).to(self.device)

                    # Reset the (gradients of the) optimizer to zero for the next training step
                    self.optimizer.zero_grad()
                    # The model calculates its guess as to if the image is AI-generated
                    # for all the images in the batch
                    outputs = self.model(inputs)
                    # Loss function calculates difference between prediction and reality
                    loss = self.criterion(outputs, targets)
                    # During the backpropagation, the model calculates
                    # what the gradients should have been in order to reduce the loss function's error.
                    loss.backward()
                    # The optimizer adjusts the model weights based on the calculated gradients
                    self.optimizer.step()

                    # Debug
                    logging.info(
                        f"[epoch {epoch}/{epochs}] Processed ({idx}/{total_items}). Batch loss: {loss.item():.4f}")

                    # Reset lists for next training step
                    x_batch.clear()
                    y_batch.clear()
            # Saves the model to disk after every epoch as to not lose training progress
            self.save()

    def save(self) -> None:
        """Saves the model to disk"""
        torch.save(self.model.state_dict(), self.model_path)
        # Debug
        logging.info(f"Model saved under {self.model_path}")

    def load(self) -> bool:
        """
        Load the model from disk
        Use this when using a pre-trained model to predict AI-probability
        """
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            # Set model into evaluation mode
            self.model.eval()
            # Debug
            logging.info(f"Loaded model from {self.model_path}")
            return True
        # Model doesn't exist
        return False

    def predict(self, file_path: str, optimal_probability: float = 0.5):
        """
        Predict AI-probability for a single image


        file_path           Path to an image, can be .png .jpeg or .tiff
        optimal_probability If the model has a certain spin, the probability threshold equalizing
                            false-negatives and false-positives can be used to correct it.

        Return probability that a given image is AI generated
        """
        noise = self.extract_prnu(file_path)
        # Extraction failed, isn't an image
        if noise is None:
            return None
        # Set model into evaluation mode
        self.model.eval()
        # Don't calculate gradients to save compute as they aren't needed
        with torch.no_grad():
            # Calculating probability and converting the format into a Python-readable one

            # np.array adds an enclosing array
            # First, the noise must be converted back into a tensor of type float32
            # .to(self.device) moves the tensor to the GPU
            # self.model(...) gets the prediction of the model
            # softmax(1) then ensures that the probability falls between 0 and 1
            # squeeze() removes unnecessary tensor dimensions
            # .cpu() moves the tensor back to the CPU, this is necessary to ensure the convertibility with numpy
            # .numpy() converts the tensor into a numpy array
            # [0] gets the first element which is the likelihood that the image is AI
            probability = torch.softmax(
                self.model(torch.tensor(np.array([noise]), dtype=torch.float32).to(self.device)), dim=1
            ).squeeze().cpu().numpy()[0]

        scaled_probability = self._scale_probability(probability, optimal_probability)
        return scaled_probability

    @staticmethod
    def _scale_probability(probability: float, threshold) -> float:
        """
        Scale the probability around the optimal threshold,
        minimizing false-positives and false-negatives at the same time"""
        if threshold <= 0 or threshold >= 1:
            return probability

        if probability < threshold:
            return 0.5 * (probability / threshold)
        else:
            return 0.5 + 0.5 * ((probability - threshold) / (1.0 - threshold))


def collect_file_paths(folder_path: str, label: str) -> tuple[list[str], list[str]]:
    """
    Collects all file paths from folder_path and its subfolders


    folder_path The folder that will be scanned
    label       The label that will be mapped to every file

    Returns a list with all file paths and a second list of the same size only containing the item label
    """

    # Empty lists that are to be filled
    paths, labels = [], []
    # Valid file formats
    valid_format = ('.jpg', '.jpeg', '.png', '.tif', '.tiff',)

    if not os.path.exists(folder_path):
        logging.error(f"Path doesn't exist: {folder_path}")
        return paths, labels

    # Traversing through all subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Filter out system files and only allow valid file types
            if file.lower().endswith(valid_format) and not file.startswith('.'):
                paths.append(os.path.join(root, file))
                labels.append(label)

    # Return all collected file paths and the list with the corresponding label
    return paths, labels


# This is executed if this file is directly executed
if __name__ == "__main__":
    # This is the cap for each training set (AI and real)
    CAP_PER_CLASS = 15000
    # Path to the training data
    PATH_TO_REAL_SET = "img/real_train"
    PATH_TO_AI_SET = "img/ai_train"
    # Number of iterations through the training data
    EPOCHS = 25
    # Batch size 32 has been found to be the optimal batch size, needing about 10 GB in memory
    BATCH_SIZE = 32
    pipeline = Model(model_path="prnu_ai_predictor.pth")
    # Testing if model can be loaded, else starting training
    if not pipeline.load():
        logging.info("Starting training...")
        # Collecting all file paths
        real_paths, real_labels = collect_file_paths(PATH_TO_REAL_SET, "real")
        ai_paths, ai_labels = collect_file_paths(PATH_TO_AI_SET, "ai")

        # Shuffle first, then cap the paths as to have balanced dataset
        real_paths, real_labels = shuffle(real_paths, real_labels, random_state=99)
        real_paths, real_labels = real_paths[:CAP_PER_CLASS], real_labels[:CAP_PER_CLASS]

        ai_paths, ai_labels = shuffle(ai_paths, ai_labels, random_state=99)
        ai_paths, ai_labels = ai_paths[:CAP_PER_CLASS], ai_labels[:CAP_PER_CLASS]

        # Put both datasets together and shuffle again
        all_paths, all_labels = shuffle(real_paths + ai_paths, real_labels + ai_labels, random_state=99)
        # Starting training
        pipeline.train_incremental(all_paths, all_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
        # Saving the fully trained model again, just to be sure
        pipeline.save()

    logging.info("Finished training or loaded model, switching to evaluation loop")

    # Evaluation loop
    while True:
        ask_path = input("\nEnter a path to an image (or 'exit' to quit): ").strip()
        if ask_path.lower() in ["exit", "quit"]:
            break
        # Predict
        results = pipeline.predict(ask_path)
        if results:
            print(f"\nAI-Likelihood: {results}")
        else:
            logging.error("\nImage couldn't be loaded. Maybe too small? (512x512).")
