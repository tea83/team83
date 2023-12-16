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
You may open the relavent notebook and run it through. 
