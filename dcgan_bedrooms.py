# -*- coding: utf-8 -*-
"""DCGAN bedrooms

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BJFFWfc2v4FFOe_H7Mty3xvvKsWrblPE

# Deep Convolutional Generative Adversarial Network (DCGAN) Implementation
#Varad Lad
##22070126057<br>Aiml A-3
## Overview

This notebook implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** as described in the paper ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"](https://arxiv.org/abs/1511.06434) by Radford et al. (2015). The goal is to train a model to generate realistic 64x64 pixel images using either the LSUN Bedrooms dataset or the CelebA Faces dataset. The implementation follows best practices for GAN training and includes detailed documentation for clarity.

## What is a GAN?

A **Generative Adversarial Network (GAN)** is a class of machine learning frameworks introduced by Ian Goodfellow et al. in 2014. It consists of two neural networks trained simultaneously:

- **Generator (G)**: Takes random noise (from a latent space) as input and generates fake data samples (e.g., images).
- **Discriminator (D)**: Takes both real and fake data samples as input and tries to distinguish between them (real vs. fake).

The two models are trained in a competitive setting:
- The Generator tries to "fool" the Discriminator by generating increasingly realistic data.
- The Discriminator tries to get better at distinguishing real data from fake data.

This adversarial process is formalized as a minimax game, where the Generator minimizes the Discriminator's ability to correctly classify fake samples, while the Discriminator maximizes its accuracy. The loss function for a GAN can be expressed as:

\[
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]
\]

Where:
- \(x \sim p_{\text{data}}(x)\): Real data samples.
- \(z \sim p_z(z)\): Random noise from the latent space.
- \(D(x)\): Discriminator's output for real data (probability of being real).
- \(G(z)\): Generator's output (fake data).
- \(D(G(z))\): Discriminator's output for fake data.

## What is a DCGAN?

A **Deep Convolutional Generative Adversarial Network (DCGAN)** is an extension of the GAN framework that uses convolutional neural networks (CNNs) for both the Generator and Discriminator. Introduced by Radford et al., DCGANs incorporate specific architectural guidelines to stabilize GAN training and improve the quality of generated images:

- **Convolutional Layers**: Replace fully connected layers with convolutional layers for better spatial feature extraction.
- **Fractionally-Strided Convolutions**: Used in the Generator to upsample the input (e.g., from a small spatial dimension to a larger image).
- **Batch Normalization**: Applied in both networks to stabilize training by normalizing activations.
- **LeakyReLU Activation**: Used in both networks (except the Generator's output layer, which uses `tanh`) to prevent vanishing gradients.
- **No Pooling Layers**: Pooling layers are avoided; strided convolutions handle downsampling/upsampling.

The DCGAN architecture in this notebook follows the structure shown in the figure below (if you have the figure in your Colab, you can reference it here):

- **Generator**: Takes a 100-dimensional noise vector \(Z\) (from a uniform distribution), projects it to a 4x4x1024 feature map, and uses four fractionally-strided convolutional layers to upsample it to a 64x64x3 RGB image.
- **Discriminator**: Takes a 64x64x3 image as input and uses convolutional layers to downsample it, outputting a probability (real or fake).

## Dataset

We will train the DCGAN on one of the following datasets:

- **LSUN Bedrooms**: A dataset of bedroom images from the Large-scale Scene Understanding (LSUN) dataset. It contains thousands of images suitable for scene generation tasks.

### Preprocessing Steps
1. **Resize Images**: All images are resized to 64x64 pixels to match the Generator's output dimensions.
2. **Normalize Pixel Values**: Scale pixel values to the range \([-1, 1]\) to match the `tanh` activation in the Generator's output layer.
3. **Batching**: Use a batch size (e.g., 32 or 64) for efficient training on a GPU.

For this notebook, we'll use a placeholder dataset (synthetic data) to demonstrate the implementation. To use a real dataset:
- Download the LSUN Bedrooms or CelebA dataset.
- Preprocess the images as described above.
- Load the data into TensorFlow or PyTorch for training.

## Model Architecture

### Generator
The Generator takes a 100-dimensional noise vector \(Z\) as input and generates a 64x64x3 image. The architecture follows the figure provided:

- **Input**: \(Z \in \mathbb{R}^{100}\), sampled from a uniform distribution.
- **Project and Reshape**: The noise vector is projected to a 4x4x1024 feature map (fully connected layer followed by reshaping).
- **Fractionally-Strided Convolutions**:
  - CONV 1: 4x4x1024 → 8x8x512 (stride 2)
  - CONV 2: 8x8x512 → 16x16x256 (stride 2)
  - CONV 3: 16x16x256 → 32x32x128 (stride 2)
  - CONV 4: 32x32x128 → 64x64x64 (stride 2)
- **Output Layer**: 64x64x64 → 64x64x3 (RGB image) with `tanh` activation.
- Batch normalization and LeakyReLU are applied after each convolutional layer (except the output layer).

### Discriminator
The Discriminator takes a 64x64x3 image (real or fake) as input and outputs a probability (real or fake). It uses standard convolutional layers to downsample the input:

- **Input**: 64x64x3 image.
- **Convolutional Layers**: Downsample the image (e.g., 64x64 → 32x32 → 16x16 → 8x8 → 4x4) while increasing the number of feature maps.
- **Output**: Flatten the final feature map and use a dense layer with sigmoid activation to output a probability.

## Training Process

Training a GAN involves optimizing both the Generator and Discriminator simultaneously:

1. **Discriminator Training**:
   - Sample a batch of real images from the dataset.
   - Generate a batch of fake images using the Generator.
   - Train the Discriminator to classify real vs. fake images (binary cross-entropy loss).
2. **Generator Training**:
   - Generate a batch of fake images.
   - Train the Generator to "fool" the Discriminator (i.e., maximize the Discriminator's loss when classifying fake images as real).
3. **Hyperparameters**:
   - Learning rate: 0.0002 (with Adam optimizer, \(\beta_1 = 0.5\)).
   - Batch size: 32 or 64.
   - Epochs: 100 (or until convergence).
4. **Stabilization Techniques**:
   - Use batch normalization to stabilize training.
   - Add noise to the Discriminator's inputs (label smoothing or Gaussian noise) to prevent overfitting.

## Expected Outputs

- **Generated Images**: After training, the Generator should produce 64x64 RGB images resembling bedrooms (if using LSUN) or faces (if using CelebA).
- **Loss Curves**: Plot the Generator and Discriminator losses over time to monitor training stability.
- **Visualizations**: Display a grid of generated images to evaluate quality.

## Implementation Notes

- This notebook uses TensorFlow for the implementation, but PyTorch can also be used.
- The code includes docstrings and inline comments for clarity.
- To deploy this implementation:
  1. Save the notebook and code to a GitHub repository.
  2. Share the repository link as part of the submission.
- For real dataset training, ensure you have sufficient GPU resources (Colab provides free access to GPUs).

## Next Steps

The following cells will:
1. Set up the environment and load the dataset.
2. Define the Generator and Discriminator models.
3. Implement the training loop.
4. Visualize the generated images and loss curves.

Let's proceed with the implementation!
"""

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import os
import time

# Mount Google Drive for potential dataset storage
drive.mount('/content/drive')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Hyperparameters
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 500
IMAGE_SIZE = 64
CHANNELS = 3

# Synthetic Data Placeholder (Replace with LSUN/CelebA loading)
def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data for demonstration purposes.
    Replace with LSUN Bedrooms or CelebA Faces dataset loading.

    Args:
        num_samples (int): Number of synthetic images to generate.

    Returns:
        np.ndarray: Synthetic images with shape (num_samples, 64, 64, 3).
    """
    data = np.random.normal(0, 1, (num_samples, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    data = np.clip(data, -1, 1)  # Normalize to [-1, 1] for tanh output
    return data

# Load or generate dataset
dataset = generate_synthetic_data()
dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(buffer_size=1000).batch(BATCH_SIZE)

# Generator Model
def build_generator(latent_dim=LATENT_DIM):
    """
    Build the DCGAN generator model based on the provided architecture.

    Args:
        latent_dim (int): Dimension of the input noise vector.

    Returns:
        Model: Keras model instance of the generator.
    """
    model = tf.keras.Sequential([
        # Input layer: Project 100-dim noise to 4x4x1024
        layers.Dense(4 * 4 * 1024, input_dim=latent_dim, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((4, 4, 1024)),

        # CONV 1: 4x4x1024 -> 8x8x512
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # CONV 2: 8x8x512 -> 16x16x256
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # CONV 3: 16x16x256 -> 32x32x128
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # CONV 4: 32x32x128 -> 64x64x64
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Output layer: 64x64x64 -> 64x64x3
        layers.Conv2D(CHANNELS, kernel_size=3, padding='same', use_bias=False, activation='tanh')
    ], name='generator')
    return model

# Discriminator Model
def build_discriminator():
    """
    Build the DCGAN discriminator model.

    Returns:
        Model: Keras model instance of the discriminator.
    """
    model = tf.keras.Sequential([
        # Input: 64x64x3
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        layers.LeakyReLU(alpha=0.2),

        # 32x32x64 -> 16x16x128
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # 16x16x128 -> 8x8x256
        layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # 8x8x256 -> 4x4x512
        layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Flatten and output
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ], name='discriminator')
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Loss Function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Training Step
@tf.function
def train_step(images):
    """
    Perform one training step for the GAN.

    Args:
        images (tf.Tensor): Batch of real images.

    Returns:
        dict: Losses for generator and discriminator.
    """
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Labels
        real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
        fake_labels = tf.zeros_like(fake_output)

        # Losses
        d_loss_real = cross_entropy(real_labels, real_output)
        d_loss_fake = cross_entropy(fake_labels, fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Compute gradients
    gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return {'d_loss': d_loss, 'g_loss': g_loss}

# Training Loop
def train(dataset, epochs):
    """
    Train the DCGAN model.

    Args:
        dataset (tf.data.Dataset): Dataset of images.
        epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        start = time.time()
        print(f'\nEpoch {epoch + 1}/{epochs}')

        for image_batch in dataset:
            losses = train_step(image_batch)
            print('.', end='', flush=True)

        # Generate and save sample images
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, LATENT_DIM)

        print(f'\nTime for epoch {time.time() - start:.2f} sec')
        print(f'Generator Loss: {losses["g_loss"]:.4f}, Discriminator Loss: {losses["d_loss"]:.4f}')

# Generate and Save Images
def generate_and_save_images(model, epoch, latent_dim):
    """
    Generate and save a grid of images.

    Args:
        model (Model): Trained generator model.
        epoch (int): Current epoch number.
        latent_dim (int): Dimension of the noise vector.
    """
    noise = tf.random.normal([16, latent_dim])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch}.png')
    plt.close()

# Train the model
train(dataset, EPOCHS)

import matplotlib.pyplot as plt

# Assuming you have trained the generator and it's named 'generator'
# and you have saved the images as 'image_at_epoch_*.png'

# Specify the epoch number for which you want to see the image
epoch_number = 50  # Replace with the desired epoch number

# Load the saved image
image_path = f'image_at_epoch_{epoch_number}.png'
image = plt.imread(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Hide the axes
plt.show()

!pip install tensorflow kaggle

import os
import glob
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# Set up Kaggle API credentials
# You need to create a kaggle.json file with your API credentials
# and place it in the CONFIG_DIR
os.environ['KAGGLE_CONFIG_DIR'] = "/content/"

# Download the LSUN bedroom dataset
# Check if the dataset directory already exists
if not os.path.exists("/content/lsun_bedroom"):
    print("Downloading LSUN bedroom dataset...")
    !kaggle datasets download -d jhoward/lsun_bedroom --unzip -p /content/
else:
    print("Dataset already exists.")

# Verify the dataset directory structure
print("Checking dataset structure...")
bedroom_images = glob.glob("/content/lsun_bedroom/*.jpg")
if len(bedroom_images) == 0:
    print("No images found in the main directory.")
    # Try to find the images in subdirectories
    bedroom_images = glob.glob("/content/lsun_bedroom/**/*.jpg", recursive=True)
    if len(bedroom_images) == 0:
        # Check if there might be a different directory structure
        all_files = glob.glob("/content/**/*", recursive=True)
        print(f"Found {len(all_files)} files in total in /content/")
        print("First 10 files:", all_files[:10])

        # Try to find the actual dataset directory
        jpg_files = [f for f in all_files if f.endswith('.jpg')]
        if jpg_files:
            print(f"Found {len(jpg_files)} JPG files.")
            print("First 5 JPG files:", jpg_files[:5])
            dataset_dir = os.path.dirname(jpg_files[0])
            print(f"Using dataset directory: {dataset_dir}")
        else:
            raise Exception("Could not find any JPG files in the downloaded dataset.")
    else:
        print(f"Found {len(bedroom_images)} images in subdirectories.")
        dataset_dir = os.path.dirname(bedroom_images[0])
else:
    print(f"Found {len(bedroom_images)} images in the main directory.")
    dataset_dir = "/content/lsun_bedroom"

print(f"Using dataset directory: {dataset_dir}")

# Setup parameters
BATCH_SIZE = 64
IMG_SIZE = 64
NOISE_DIM = 100
EPOCHS = 1000  # Reduced for faster training
CHECKPOINT_DIR = './training_checkpoints'
OUTPUT_DIR = './generated_images'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_lsun_dataset(data_dir, batch_size):
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    print(f"Found {len(image_paths)} images in {data_dir}")

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Make sure the dataset is properly downloaded.")

    # Convert paths to tensor of strings explicitly
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def preprocess(image_path):
        # Ensure the image_path is a string
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')

        # Debug information
        print(f"Processing image: {image_path}, type: {type(image_path)}")

        # Read the file
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

        # Data augmentation
        image = tf.image.random_flip_left_right(image)

        # Normalize to [-1, 1]
        return (image / 127.5) - 1

    # Process a single image to check if it works
    test_image = preprocess(image_paths[0])
    print(f"Test image shape: {test_image.shape}, min: {tf.reduce_min(test_image)}, max: {tf.reduce_max(test_image)}")

    # Create the dataset
    return dataset.map(lambda x: preprocess(x)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Generator
def build_generator():
    model = tf.keras.Sequential()

    # First layer using Input
    model.add(layers.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(8 * 8 * 512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(3, 5, strides=1, padding='same', activation='tanh'))

    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()

    # First layer using Input
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.Conv2D(64, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # No activation for WGAN-like loss

    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Model summaries
generator.summary()
discriminator.summary()

# Set up optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# Create checkpoints
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator,
    discriminator=discriminator
)

# Training step
@tf.function
def train_step(real_images):
    # Handle batch size issues
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator outputs
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    # Generate images
    predictions = model(test_input, training=False)

    # Scale images to [0, 1]
    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

    # Also display the figure if in interactive mode
    if epoch % 10 == 0 or epoch == EPOCHS-1:
        display_fig = plt.figure(figsize=(8, 8))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')
        plt.show()

# Try to load the dataset
try:
    print("Loading dataset...")
    dataset = load_lsun_dataset(dataset_dir, BATCH_SIZE)

    # Start training
    print("Starting training...")

    # Fixed noise vector for visualization
    seed = tf.random.normal([16, NOISE_DIM])

    # Initialize metrics tracking
    gen_losses, disc_losses = [], []

    # Start training
    for epoch in range(EPOCHS):
        start = time.time()

        # Initialize epoch metrics
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()

        # Create progress bar
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")

        # Train on each batch
        for image_batch in progress_bar:
            g_loss, d_loss = train_step(image_batch)

            # Update metrics
            epoch_gen_loss.update_state(g_loss)
            epoch_disc_loss.update_state(d_loss)

            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.numpy():.4f}',
                'D_loss': f'{d_loss.numpy():.4f}'
            })

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Generate and save images
        generate_and_save_images(generator, epoch+1, seed)

        # Calculate epoch time
        time_taken = time.time() - start

        # Log metrics
        print(f'Time for epoch {epoch+1} is {time_taken:.2f} sec')
        print(f'Generator loss: {epoch_gen_loss.result():.4f}, Discriminator loss: {epoch_disc_loss.result():.4f}')

        # Store losses for plotting
        gen_losses.append(epoch_gen_loss.result().numpy())
        disc_losses.append(epoch_disc_loss.result().numpy())

        # Plot losses every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS-1:
            plt.figure(figsize=(10, 5))
            plt.plot(gen_losses, label='Generator Loss')
            plt.plot(disc_losses, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('GAN Training Progress')
            plt.savefig(os.path.join(OUTPUT_DIR, f'losses_epoch_{epoch+1}.png'))
            plt.show()

    # Generate final images
    final_seed = tf.random.normal([36, NOISE_DIM])
    generated_images = generator(final_seed, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale to [0, 1]

    # Display final results
    plt.figure(figsize=(12, 12))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_generated_images.png'))
    plt.show()

    print("Training complete! Final images saved to", OUTPUT_DIR)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Trying an alternative approach...")

    # If we can't find the dataset, let's create a small synthetic dataset
    print("Creating a synthetic dataset for demonstration...")

    # Generate random images
    def create_synthetic_dataset(num_images=1000):
        images = []
        for _ in range(num_images):
            # Create a random image
            image = tf.random.normal([IMG_SIZE, IMG_SIZE, 3])
            # Scale to [-1, 1]
            image = tf.clip_by_value(image, -1, 1)
            images.append(image)

        # Create a dataset
        dataset = tf.data.Dataset.from_tensor_slices(images)
        return dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create synthetic dataset
    synthetic_dataset = create_synthetic_dataset()
    print("Synthetic dataset created. Starting training...")

    # Fixed noise vector for visualization
    seed = tf.random.normal([16, NOISE_DIM])

    # Initialize metrics tracking
    gen_losses, disc_losses = [], []

    # Start training with synthetic data
    for epoch in range(EPOCHS):
        start = time.time()

        # Initialize epoch metrics
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()

        # Create progress bar
        progress_bar = tqdm(synthetic_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")

        # Train on each batch
        for image_batch in progress_bar:
            g_loss, d_loss = train_step(image_batch)

            # Update metrics
            epoch_gen_loss.update_state(g_loss)
            epoch_disc_loss.update_state(d_loss)

            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.numpy():.4f}',
                'D_loss': f'{d_loss.numpy():.4f}'
            })

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Generate and save images
        generate_and_save_images(generator, epoch+1, seed)

        # Calculate epoch time
        time_taken = time.time() - start

        # Log metrics
        print(f'Time for epoch {epoch+1} is {time_taken:.2f} sec')
        print(f'Generator loss: {epoch_gen_loss.result():.4f}, Discriminator loss: {epoch_disc_loss.result():.4f}')

        # Store losses for plotting
        gen_losses.append(epoch_gen_loss.result().numpy())
        disc_losses.append(epoch_disc_loss.result().numpy())

        # Plot losses every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS-1:
            plt.figure(figsize=(10, 5))
            plt.plot(gen_losses, label='Generator Loss')
            plt.plot(disc_losses, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('GAN Training Progress')
            plt.savefig(os.path.join(OUTPUT_DIR, f'losses_epoch_{epoch+1}.png'))
            plt.show()

    # Generate final images
    final_seed = tf.random.normal([36, NOISE_DIM])
    generated_images = generator(final_seed, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale to [0, 1]

    # Display final results
    plt.figure(figsize=(12, 12))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_generated_images.png'))
    plt.show()

    print("Training complete with synthetic data! Final images saved to", OUTPUT_DIR)

