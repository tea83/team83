# team83


### Task 2- Generating Content

#### 2.1 `ImageCaptionNotebook` Folder
- Inside this folder, there is a notebook (`ImageCaptionNotebook.ipynb`) where an open-source pretrained model (BLIP2) is utilized to generate relevant captions from images.


#### 2.2 `fine_tuning` Folder
- Inside this folder, there are two notebooks:
  - `Prompt_Engineering.ipynb`: Captions generated from the pretrained model are concatenated with inferred company, date-time for temporal relevance, and likes. Two strategies, PEFT (Prefix-Tuning with Early Fine-tuning) and LoRA (Low-Rank Adaptive Weights), are applied for better finetuning and efficiency.
  - `Finetuning_Bloom_7b.ipynb`: The Stable Diffusion XL model is fine-tuned using Low-Rank Adaptive Weights (LoRA).


#### 2.3 `Task_2_Inference` Folder
- This folder contains a notebook (`Task_2_Inference.ipynb`) where the finetuned model is employed to generate predictions on test data.

## How to Use

1. Clone the repository to your local machine.
2. Navigate to the desired task or subtask folder.
3. Open the relevant Jupyter notebook using your preferred environment (e.g., Jupyter Notebook, Jupyter Lab).
4. Follow the instructions and run the cells in the notebook to execute the code.

Feel free to explore each folder and notebook for detailed implementation and documentation.

