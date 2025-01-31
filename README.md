# Language-translation-with-a-Transformer

This repository contains a Transformer-based language translation model trained to translate Portuguese to English. The implementation follows a simplified approach using TensorFlow/Keras and includes a Jupyter Notebook demonstrating training on a small dataset for quick experimentation.

![image](https://www.tensorflow.org/images/tutorials/transformer/CrossAttention.png)

# 🚀 Features
✅ Transformer Model: Uses an encoder-decoder architecture for translation based on the paper.

✅ Small Dataset: A minimal dataset is included for demonstration.

✅ Training Example: The model is trained for 2 epochs in the notebook.

✅ Preprocessing & Tokenization: Uses TensorFlow Text for text processing.

✅ Notebook for Experimentation: Easy-to-run Jupyter Notebook for hands-on learning.

# Training the Transformer model
1. Use notbook[Nootebook](https://github.com/KaushiML3/Transformer_Language-translation/blob/main/notebook/transformer-Portuguese_2_English.ipynb)
    1. Run the provided Jupyter Notebook for training.
    2. Since training can be time-consuming, using a GPU is recommended for better performance.

2. Running the Python Script
    1. Clone the Git repository.
    2. Execute the training script: python src/training.py.
    3. Before running, configure the following:
        Dataset link
        Tokenizer model
        Other required parameters
    4. The dataset is stored in: artifact/dataset/.
    5. The trained model is saved in: artifact/model/.
    6. Run the inference script: python app.py.
