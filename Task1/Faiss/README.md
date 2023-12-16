# Multimodal Embedding and FAISS System Documentation

This notebook focuses on building a recommendation system using multimodal embeddings. The system utilizes embeddings from both image and text data to recommend items based on user preferences, represented by likes.

## Dependencies
The following libraries are used in this notebook:
- **os, pickle, numpy**: Basic Python libraries for file operations and numerical computations.
- **faiss**: A library for efficient similarity search and clustering of dense vectors.
  
## Functions
1. **load_emb(enc_type)**:
   - Loads embeddings from the specified encoding type ('img', 'txt', 'likes').
   - Returns the embeddings and corresponding IDs.

2. **preprocess(data1, data2, lk)**:
   - Combines embeddings from two sources and returns the multimodal embeddings along with likes.

3. **build_index(embeddings)**:
   - Builds and returns an index for the given embeddings using the FAISS library.

4. **calc_error(test, index, train_lk, test_lk, k=15)**:
   - Calculates the Root Mean Squared Error (RMSE) for a set of test embeddings.
   - Compares predicted likes based on nearest neighbors with actual likes.

## Workflow
1. **Load Embeddings**:
   - Load image, text, and likes embeddings using the `load_emb` function.

2. **Preprocess Data**:
   - Preprocess the data by combining image and text embeddings and extracting likes.

3. **Split Data**:
   - Split the multimodal data into training and testing sets.

4. **Build Index**:
   - Build an index using the training set embeddings.

5. **Calculate RMSE**:
   - Calculate the RMSE for the test set using the built index.

6. **Build Final Index**:
   - Build a final index using all multimodal embeddings.

7. **Save Index**:
   - Save the final index to a file ('mm-ind.index').

For further details and updates, refer to the notebook code and associated documentation in the provided Colab link.
