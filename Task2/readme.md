

### Task 2- Generating Content

#### 2.1 `ImageCaptionNotebook.ipynb` 
- In this notebook,an open-source pretrained model (BLIP2) is utilized to generate relevant captions from images.


#### 2.2 `fine_tuning.ipynb` 
- This notebook accomplishes two tasks:
  - Prompt_Engineering: Captions generated from the pretrained model are concatenated with inferred company, date-time for temporal relevance, and likes.
  - Finetuning: Bloom7b, a powerful model 176B model is used.Two strategies, PEFT (Prefix-Tuning with Early Fine-tuning) and LoRA (Low-Rank Adaptive Weights), are applied for better finetuning and efficiency.


#### 2.3 `Task_2_Inference.ipynb` 
- The finetuned model is used to generate prediction on test data.
## How to Use

1. Clone the repository to your local machine.
2. Navigate to the desired task or subtask folder.
3. Open the relevant Jupyter notebook using your preferred environment (e.g., Jupyter Notebook, Jupyter Lab).
4. Follow the instructions and run the cells in the notebook to execute the code.

Feel free to explore each folder and notebook for detailed implementation and documentation.


# Image Captioning Notebook Documentation

This notebook focuses on image captioning using the Salesforce Blip Image Captioning model. The process involves reading a dataset from a CSV file, where each row contains information about an image, including the path to the image file. The notebook then uses the Blip Image Captioning model to generate captions for each image and saves the results in a new CSV file.

## Colab Link
You can view and run this notebook on Google Colab by clicking the following badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrionXV/InterIITAdobe/blob/main/task2/ImageCaptionNotebook.ipynb)

## Dependencies
- **requests**: A library for making HTTP requests in Python.
- **PIL**: Python Imaging Library for image processing.
- **transformers**: Hugging Face's Transformers library for natural language processing.
- **torch**: The PyTorch deep learning library.
- **pandas**: A data manipulation library for data analysis.
- **numpy**: A library for numerical operations in Python.
- **tqdm**: A library for displaying progress bars.

## Dataset Preparation
The notebook reads a dataset from a CSV file (`somepath.csv`) using the pandas library. It initializes a new column (`img_caption`) for storing image captions, which is initially set to None.

## Model Initialization
The notebook initializes the Blip Image Captioning model (`Salesforce/blip-image-captioning-large`) and its processor from the Hugging Face Transformers library. The model is loaded onto the GPU if available.

## Image Captioning Function
The notebook defines a function (`image_captioner`) that takes an image path as input, opens the image, and generates a caption using the Blip Image Captioning model. The captions are then stored in the `img_caption` column of the DataFrame.

## Caption Generation
The notebook iterates through the DataFrame, generating captions for each image using the defined function. The progress is displayed using tqdm. The resulting DataFrame is then saved to a new CSV file (`captioned_dataset.csv`).

# Fine-tuning Notebook Documentation

This notebook focuses on fine-tuning a language model, specifically the `bigscience/bloom-7b1` model, for a tweet generation task. The fine-tuning process involves loading the pre-trained language model, configuring it, and training it on a custom dataset containing tweet-related information. Additionally, the notebook utilizes the `peft` library for efficient fine-tuning.

## Colab Link
You can view and run this notebook on Google Colab by clicking the following badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Il0kWaqm-v6rtZZvU2cLROSUG-8YwXWN)

## Dependencies
- **datasets**: A library for working with various natural language processing datasets.
- **bitsandbytes, accelerate, loralib**: Libraries for efficient training and fine-tuning of language models.
- **transformers**: Hugging Face's Transformers library for natural language processing.
- **peft**: Library for efficient training of transformers.

## Model Fine-tuning
The notebook starts by installing the required libraries and setting up the environment. It then loads the pre-trained `bigscience/bloom-7b1` language model, configures it, and fine-tunes it using the provided tweet dataset. The fine-tuning process involves freezing certain model parameters, enabling gradient checkpointing, and other configuration settings.

## Efficient Fine-tuning with PEFT
The notebook utilizes the `peft` library to enhance the efficiency of fine-tuning. It configures the PEFT model and prints the number of trainable parameters in the model.

## Dataset Preparation
The notebook downloads a dataset (`data.zip`) from Google Drive, extracts it, and reads it into a DataFrame (`final_captioned_dataset.xlsx`). It then preprocesses the DataFrame by removing unnecessary columns and splitting it into training and validation sets.

## Tokenization and Data Preparation
The notebook cleans and merges columns in the dataset, preparing it for training. It uses the Hugging Face `transformers` library for tokenization and creates a DatasetDict with training and validation datasets.

## Model Training
The notebook trains the language model using the `transformers.Trainer` class. It specifies training arguments, such as batch size, learning rate, and output directory. The training process is then executed, and the trained model is pushed to the Hugging Face Model Hub.

## Inference with the Fine-tuned Model
The notebook demonstrates how to load the fine-tuned model from the Hugging Face Model Hub and perform inference on a sample input.

# Inference Notebook Documentation

This notebook is focused on performing inference using a fine-tuned language model (`bloom-7b1-lora-tagger-for-tweet-generation`) on a test dataset. The process involves loading the trained model, preparing the test dataset, creating prompts for inference, and generating predictions. The final output is then saved to an Excel file.

## Colab Link
You can view and run this notebook on Google Colab by clicking the following badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a0PDiQsyB0tCvYR0GED1E7QXTqyPAay0)

## Dependencies
- **bitsandbytes, accelerate, loralib**: Libraries for efficient training and fine-tuning of language models.
- **transformers**: Hugging Face's Transformers library for natural language processing.
- **peft**: Library for efficient training of transformers.
- **datasets**: A library for working with various natural language processing datasets.

## Test Dataset Preparation
The notebook starts by fetching the test dataset (`test.csv`) from Google Drive and loading it into a DataFrame. The first few rows of the dataset are displayed for inspection.

## Model Loading
The fine-tuned model (`bloom-7b1-lora-tagger-for-tweet-generation-1100iter`) is loaded using the PEFT library for efficient training. The model is then configured with the PEFT model.

## Test Dataset Tokenization and Cleaning
The test dataset is tokenized using the Hugging Face `transformers` library, and necessary cleaning steps are applied to handle missing values.

## Prompt Creation
Prompts for generating predictions are created by merging columns from the dataset.

## Prediction Generation
The notebook generates predictions for each prompt using the fine-tuned language model. The output tokens are then decoded, and the resulting predictions are stored in a list.

## Post-processing and Saving Predictions
The predictions are post-processed to extract the relevant information. The final predictions are then saved to an Excel file (`output_0-500.xlsx`).

## Additional Information

For further details and updates, refer to the notebook code and associated documentation in the provided Colab link.
