# CLIP Embeddings Generator

This Python script provides functionality to generate image and text embeddings using the CLIP (Contrastive Language-Image Pre-training) model from OpenAI. The code utilizes the Hugging Face Transformers library for easy integration with CLIP.

## Functions

### `load_clip_model()`

This function loads the CLIP processor, model, and tokenizer. It determines the computing device (GPU if available, otherwise CPU) and returns these components along with the tokenizer.

### `process_batches(processor, tokenizer, model, device, image_paths, text_descriptions, batch_size)`

This function processes batches of images and text descriptions to obtain their respective embeddings using the CLIP model. It takes in the CLIP processor, tokenizer, model, computing device, image paths, text descriptions, and batch size as inputs. It returns concatenated image and text embeddings.

### `get_embeddings(df, batch_size)`

This function acts as an interface for obtaining CLIP embeddings from a given DataFrame. It takes in the DataFrame `df` containing image paths and relevant text information, along with the desired batch size. It internally calls `load_clip_model()` and `process_batches()` to generate image and text embeddings. The resulting embeddings are printed and returned.

## Usage

```python
# Example Usage
import pandas as pd

# Assuming df is a DataFrame with columns 'image_path', 'username', 'inferred company', and 'content_processed'
# and batch_size is an integer specifying the batch size for processing
df = pd.read_csv("behaviour_detection_train.csv")
batch_size = 16

# Get CLIP embeddings
image_embeddings, text_embeddings = get_embeddings(df, batch_size)
# EfficientNet Embeddings Generator

This Python script provides functionality to generate image embeddings using the EfficientNet model. The code utilizes the `EfficientNet` implementation from the `efficientnet_pytorch` library and `torchvision.transforms` for image preprocessing.

## Function

### `EfficientNetEmbed(df)`

Generate image embeddings using an EfficientNet model for a given DataFrame.

#### Arguments:

- `df` (DataFrame): DataFrame containing image file paths.

#### Returns:

- `embeddings_tensor` (Tensor): Tensor containing the image embeddings for the entire DataFrame.

## Usage

```python
# Example Usage
import pandas as pd

# Assuming processed_df is a DataFrame with a column 'image_path'
img_emb_Enet = EfficientNetEmbed(processed_df)

