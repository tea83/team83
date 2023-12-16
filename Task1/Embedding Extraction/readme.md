# CLIP Model Embedding Processor

This Python script utilizes the CLIP (Contrastive Language-Image Pre-training) model from the Hugging Face Transformers library to generate embeddings for images and corresponding text descriptions. The code processes images and text in batches, allowing for efficient computation.

## Dependencies

- `PIL` (Python Imaging Library): For working with images.
- `torch`: PyTorch library for deep learning.
- `transformers`: Hugging Face Transformers library for natural language processing models.
- `tqdm`: A library for creating progress bars in the console.

## Functions

### `load_clip_model()`

This function loads the CLIP processor, tokenizer, and model. It also determines the device to run the model on (cuda if available, otherwise cpu).

#### Returns

- `processor`: CLIP Processor instance.
- `model`: CLIP Model instance.
- `device`: The device (cuda or cpu).
- `tokenizer`: CLIPTokenizer instance.

### `process_batches(processor, tokenizer, model, device, image_paths, text_descriptions, batch_size)`

This function processes batches of images and corresponding text descriptions to generate image and text embeddings using the CLIP model.

#### Parameters

- `processor`: CLIP Processor instance.
- `tokenizer`: CLIPTokenizer instance.
- `model`: CLIP Model instance.
- `device`: The device to run the model on.
- `image_paths`: List of file paths to the images.
- `text_descriptions`: List of text descriptions corresponding to the images.
- `batch_size`: The batch size for processing.

#### Returns

- `image_embeddings`: Tensor containing image embeddings.
- `text_embeddings`: Tensor containing text embeddings.


# EfficientNet Image Embedding Generation

## Load and Process Image Paths

- Load image paths from the CSV file located at `/kaggle/input/valid-dataset-adobe/valid_paths_data.csv` using Pandas.

## Set Up Temporary Directory

- Create a temporary directory named `temp`.

## Define and Apply Image Path Function

- Define a function to modify image paths based on an index.
- Apply the function to create a new 'image_path' column in the DataFrame.

## Load EfficientNet Model

- Load a pre-trained EfficientNet model using the `efficientnet_pytorch` library.
- The model can be initialized with a specific architecture (default: 'efficientnet-b0') and loaded onto the specified device ('cuda' or 'cpu').

## Preprocess Image Function

- Define a function to prepare an image for input to an EfficientNet model.
- The image is resized, converted to a tensor, and normalized.

## Generate Image Embeddings Function

- Define a function to produce image embeddings using a pre-trained EfficientNet model.
- The embeddings are obtained for a batch of images.

## Generate Embeddings for DataFrame Function

- Define a function to generate image embeddings for a DataFrame containing image file paths.
- The DataFrame is processed in batches.

## Example Usage

- Provide an example of using the defined functions to generate image embeddings for the given DataFrame.

## Save Embeddings as NumPy Array

- Convert the image embeddings to a NumPy array.
- Save the NumPy array to a file named 'ENet_Embeds_final.npy'.

## MPNET Embeddings Generation

This document outlines the process of generating MPNET text embeddings for a dataframe containing text data and relevant information. These embeddings can be utilized for various downstream tasks like document similarity, clustering, or classification.

### Dependencies

* `torch`
* `pandas`
* `sentence-transformers`
* `tqdm`
* `numpy`

### Functions

* **`load_mpnet_model(model_name, device)`:** Loads a pre-trained MPNET model and places it on the specified device (`cuda` or `cpu`).
* **`generate_text_embeddings(model, text_batch, device)`:** Generates MPNET embeddings for a batch of text strings.
* **`generate_embeddings_for_text_dataframe(model, dataframe, text_column, batch_size, device)`:** Iterates through the dataframe in batches, generating and concatenating MPNET embeddings for the specified text column.
* **`mpNet(df)`:** Combines the above functions to create MPNET embeddings for the entire dataframe. It includes the following steps:
    * Joins relevant columns (`username`, `inferred company`, `content_processed`) into a single text column.
    * Loads the MPNET model and sets the device.
    * Generates and concatenates MPNET embeddings in batches.
    * Prints the shape of the resulting embedding tensor.
    * Saves the numpy array of embeddings to a specified file.

### Usage

1. Import the necessary libraries.
2. Load your dataframe containing text data and relevant columns.
3. Call the `mpNet(df)` function, passing your dataframe as the argument.
4. The function will print the shape of the generated embedding tensor and save it to the specified file.

