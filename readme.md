# Variational Autoencoder (VAE) on MNIST

A simple implementation of a **Variational Autoencoder (VAE)** using PyTorch to learn a probabilistic latent representation of handwritten MNIST digits and reconstruct the input images.

## Overview

A Variational Autoencoder is a generative deep-learning model that consists of two main components:

* **Encoder** – maps the input image into a probabilistic latent representation represented by a mean (`μ`) and log-variance (`logvar`).
* **Decoder** – samples a latent representation and reconstructs the original image.

The model follows the pipeline:

```text
MNIST Image (784)
       ↓
    Encoder
       ↓
   μ and logvar
       ↓
Reparameterization
       ↓
 Latent Space (20)
       ↓
    Decoder
       ↓
Reconstructed Image (784)
```

## Architecture

The VAE uses the following dimensions:

```text
Input Dimension  : 784
Hidden Dimension : 400
Latent Dimension : 20
```

### Encoder

```text
784 → 400 → μ
           → logvar
```

### Decoder

```text
20 → 400 → 784
```

## VAE Loss

The model is trained using two components:

### 1. Reconstruction Loss

Binary Cross Entropy (BCE) measures how accurately the reconstructed image matches the original image.

### 2. KL Divergence

KL divergence regularizes the latent distribution so that it remains close to a standard normal distribution.

The total loss is:

```text
VAE Loss = Reconstruction Loss + KL Divergence
```

## Reparameterization Trick

The encoder produces `μ` and `logvar`. The latent variable is sampled using:

```text
z = μ + ε × σ
```

where:

```text
σ = exp(0.5 × logvar)
```

and `ε` is sampled from a standard normal distribution.

This allows the VAE to perform backpropagation through the sampling process.

## Training

The model is trained on the MNIST dataset using:

* Optimizer: Adam
* Learning Rate: `1e-3`
* Batch Size: `128`
* Epochs: `10`

## Reconstruction

After training, the model reconstructs MNIST images by passing them through the encoder and decoder.

The notebook compares:

```text
Original Images
       ↓
     VAE
       ↓
Reconstructed Images
```

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Jupyter Notebook

## Dataset

This project uses the **MNIST handwritten digit dataset**.

The dataset is downloaded automatically when the notebook is executed and is intentionally **not included in this GitHub repository** to keep the repository lightweight.

The dataset is stored locally under:

```text
data/
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib numpy
```

### 3. Open the notebook

Open:

```text
VAE_demo.ipynb
```

using Jupyter Notebook, JupyterLab, or VS Code.

### 4. Run the notebook

The MNIST dataset will be downloaded automatically into the local `data/` directory.

## Project Structure

```text
.
├── VAE_demo.ipynb
├── README.md
├── .gitignore
└── data/
    └── MNIST/          # Not tracked by Git
```

## Key Learning

This project demonstrates how a VAE differs from a traditional Autoencoder by introducing a **probabilistic latent space**. Instead of encoding an input into one fixed latent vector, the encoder learns a distribution using `μ` and `logvar`, allowing the model to sample new latent representations and generate/reconstruct data.

## Future Improvements

* Visualize the 2D latent space
* Generate new MNIST digits by sampling from the latent space
* Compare VAE reconstruction quality with a standard Autoencoder
* Experiment with different latent dimensions
* Evaluate reconstruction loss and KL divergence separately
