# LDM Project for High-Quality Human Face Generation (In Progress)

This project is working on a Latent Diffusion Model (LDM) developed using the **VAE + Diffusion** architecture to generate high-quality human faces. Currently, the **VAE part is complete**, and active work is underway on the diffusion part. In later stages, the diffusion model will be integrated with the VAE to form a fully functional LDM.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [Installation and Requirements](#installation-and-requirements)
- [VAE Model Details](#vae-model-details)
- [Training Process and Usage](#training-process-and-usage)
- [Testing and Evaluation](#testing-and-evaluation)
- [Future Plans](#future-plans)
- [Additional Notes](#additional-notes)
- [Dataset License](#dataset-license)

---

## About the Project

The main goal of this project is to use the **VAE + Diffusion** architecture to generate high-quality human faces.
- **VAE Part:** Utilizes the encoder and decoder architectures of the model to compress the input image into a latent space and reconstruct it.
- **Diffusion Part:** Currently under development; in the future, it will be integrated with the VAE to create a complete Latent Diffusion Model (LDM).

---

## Installation and Requirements

To run the project, follow these steps:

1. **Environment Setup:**
   - It is recommended to use a GPU-supported environment (such as Google Colab) to run the project.
   - Required libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`, `kaggle`, etc.
   
2. **Retrieving Project Files:**
   - Copy the project folder from Google Drive or a local repository to the Colab environment:
     ```python
     !cp -r "/content/drive/MyDrive/..." /content/
     ```
     
3. **Dataset:**
   - Download the FFHQ dataset from Kaggle:
     ```bash
     !kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
     ```
   - The downloaded ZIP file is extracted to the specified directory using the `zipfile` library.

4. **Making the Necessary Settings:**
   - Training parameters (e.g., `batch_size`, `epochs`, `kl_weight`, etc.) can be configured via the `config.py` file.
   - Image transformations (resize, normalization, etc.) are defined and configured to suit the dataset.

---

## VAE Model Details

The VAE (Variational Autoencoder) model is used to encode image data into a latent space and reconstruct it. The main components of the model are as follows:

- **Encoder:**
  - **Downsampling:** `Downsample` blocks are used to reduce the dimensions of the input image.
  - **Residual Blocks:** Used to preserve information flow in deep networks and improve model performance.
  - **Attention Blocks:** Added to capture important features in the image.
  - **Latent Distribution:** The encoder output is split into two parts (mean and log_variance). Latent sampling is performed using the Reparameterization Trick.
  
- **Decoder:**
  - **Upsampling:** `Upsample` blocks are used to convert the data from the latent space back to the original image dimensions.
  - **Skip Connections:** Intermediate outputs (skip connections) from the encoder are used during reconstruction to prevent loss of detail.
  - **Output Activation:** In the final layer, `Tanh` activation is used so that the generated image is normalized to the [-1, 1] range.

- **Loss Functions:**
  - **Reconstruction Loss:** Computes the difference between the original and generated image using MSE (Mean Squared Error).
  - **KL Divergence Loss:** Used to regularize the latent distribution, preventing overfitting and ensuring a structured distribution.
  - **KL Annealing:** The weight of the KL loss is gradually increased during training.

- **Additional Techniques:**
  - **Gradient Clipping:** Applied during training to prevent gradient explosions.
  - **Early Stopping:** Training is stopped if the validation loss does not improve for a certain period, and the best model weights are saved to the `checkpoints` directory.
  - **Learning Rate Scheduler:** Dynamically adjusts the learning rate during training to optimize performance.

---

## Training Process and Usage

### Training

1. **Training via Notebook:**
   - Start the training process by running the `train_vae.ipynb` notebook.
   - During training, the model is trained using the FFHQ dataset with a validation phase.
   - Relevant metrics (total loss, KL loss, reconstruction loss) are calculated and visualized at the end of each epoch.

2. **Saving the Model:**
   - During training, the components of the model that achieved the lowest validation loss are saved to the following files:
     - `best_encoder.pth`
     - `best_decoder.pth`
     - `best_vae.pth`
   - These files are stored in the `checkpoints` directory.

3. **Workflow:**
   - **Data Loading:** The `FFHQDataset` class loads images from the specified directory and applies the necessary transformations.
   - **DataLoader:** DataLoaders are created for training and testing datasets.
   - **Training Loop:** In each epoch, a forward pass, loss calculation, backpropagation, and optimization steps are performed.
   - **Validation:** After training, the model's performance is evaluated on the validation set.

### Usage

- **After the Model Has Been Trained:**
  - The model can be reloaded using the weight files in the `checkpoints` directory.
  - New face images can be generated by sampling from the latent space.
  
- **Testing Procedures:**
  - The test procedures in the `test` folder allow for examination of the model's outputs and related metrics (loss, visualizations).
  - The images in the `results` directory provide a summary of the results obtained during training.

---

## Testing and Evaluation

Various evaluation methods are implemented in the project to measure the performance of the VAE model:

- **calculate_metrics:** Computes metrics used to measure the model's performance.
- **plot_losses:** Plots training and validation losses on a graph.
- **track_tensor_values:** Tracks changes in tensor values within the model.
- **visualize_random_input_output:** Compares randomly selected input images with the model's outputs.

These methods allow for a detailed analysis of both the training performance and the quality of the generated outputs.

---

## Future Plans

- **Development of the Diffusion Model:**
  - Currently, the VAE part is complete, and active work is underway on the diffusion part.
  - In the future, once the diffusion model training is completed, it will be integrated with the VAE to achieve a fully functional LDM.
  
- **Improvements:**
  - Continuous improvements will be made to the model architecture, training parameters, and data processing procedures.
  - New metrics and evaluation methods will be added to further enhance the model's performance.

---

## Additional Notes

- **Code Modifications:**
  - Any changes made to the code should be documented with comments.
  - Adjustments made in the `config.py` file directly affect the model's behavior; therefore, they should be configured carefully.

- **Troubleshooting:**
  - If you encounter any errors during training or model usage, first check the configuration settings.
  - If necessary, errors can be diagnosed using log messages and visualization outputs.

---

### Dataset License

The FFHQ dataset used in this project is provided under the **CC BY-SA 3.0** license. Please adhere to the license conditions when using the dataset.

---

## Thanks

We would like to thank everyone who contributed to this project. Your feedback and contributions during the development process are invaluable in improving the model.

---

*Note: The project is still under development, and new features and improvements are being added regularly. Follow the GitHub repository for updates.*
