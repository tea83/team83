

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

