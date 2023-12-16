# Repository Name

## Overview

This repository contains code and notebooks for two distinct tasks. The repository is organized into three main folders:

### Task 1

#### 1.1 `train` Folder
- This folder contains a Jupyter notebook (`xgboost.ipynb`) where an XGBoost model is trained to predict likes based on the provided data.

#### 1.2 `inference` Folder
- Inside this folder, there are notebooks (`inference_notebook1.ipynb`, `inference_notebook2.ipynb`, etc.) for predicting likes for the final test data using the trained model.

#### 1.3 `faiss` Folder
- This folder includes a Jupyter notebook (`faiss_search.ipynb`) demonstrating the usage of the Faiss library for similarity search.

### Task 2

#### 2.1 `embedding_extraction` Folder
- This folder includes notebooks related to embedding extraction. The exact content and purpose of these notebooks are not detailed in the provided information.

#### 2.2 `ImageCaptionNotebook` Folder
- Inside this folder, there is a notebook (`ImageCaptionNotebook.ipynb`) where an open-source pretrained model (BLIP2) is utilized to generate relevant captions from images.

#### 2.3 `Task_2_Inference` Folder
- This folder contains a notebook (`Task_2_Inference.ipynb`) where the finetuned model is employed to generate predictions on test data.

#### 2.4 `fine_tuning` Folder
- Inside this folder, there are two notebooks:
  - `Prompt_Engineering.ipynb`: Captions generated from the pretrained model are concatenated with inferred company, date-time for temporal relevance, and likes. Two strategies, PEFT (Prefix-Tuning with Early Fine-tuning) and LoRA (Low-Rank Adaptive Weights), are applied for better finetuning and efficiency.
  - `Finetuning_Bloom_7b.ipynb`: The Stable Diffusion XL model is fine-tuned using Low-Rank Adaptive Weights (LoRA).

### Extra Folder

#### `image_generation` Folder
- This folder contains a notebook (name not specified) where the Stable Diffusion XL model is fine-tuned using Low-Rank Adaptive Weights (LoRA) for image generation.

### Additional Notebook

#### `adobe_eda.ipynb`
- This Jupyter notebook provides exploratory data analysis (EDA) of the provided data from Adobe.

## How to Use

1. Clone the repository to your local machine.
2. Navigate to the desired task or subtask folder.
3. Open the relevant Jupyter notebook using your preferred environment (e.g., Jupyter Notebook, Jupyter Lab).
4. Follow the instructions and run the cells in the notebook to execute the code.

Feel free to explore each folder and notebook for detailed implementation and documentation.

## Contributors

- List the contributors or mention that it is a solo project.

## License

- Specify the license under which the code is distributed.

## Acknowledgments

- Optionally, include acknowledgments or references to any external sources, libraries, or frameworks used in the project.

Enjoy exploring and using the repository! If you have any questions or issues, feel free to reach out to the contributors.

