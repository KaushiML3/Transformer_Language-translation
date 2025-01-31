
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import os
import logging

import shutil

from custom_logger import setup_logger

logger=setup_logger("data_preprocessing")

class data_preprocess():

    def __init__(self,dataset_path,tokenizer_model_name,train_size,test_size):
        self.dataset_path=dataset_path
        self.tokenizer_model_name=tokenizer_model_name
        self.train_size=train_size
        self.test_size=test_size
        self.train_examples=None
        self.val_examples=None
        self.tokenizers=None
        self.data_download_tokenized()
 
    def data_download_tokenized(self):
        #download the dataset
        examples, metadata = tfds.load(self.dataset_path,with_info=True,as_supervised=True)

        self.train_examples, self.val_examples = examples['train'].take(self.train_size), examples['validation'].take(self.test_size)
    
        for pt_examples, en_examples in self.train_examples.batch(3).take(1):
            logger.info('>>>>. Examples in Portuguese:')
            for pt in pt_examples.numpy():
                logger.info(f"{pt.decode('utf-8')}")

            logger.info('>>>>>> Examples in English:')
            for en in en_examples.numpy():
                logger.info(f"{en.decode('utf-8')}")

        logger.info(f"Train dataset size :{len(self.train_examples)}")
        logger.info(f"Val dataset size :{len(self.val_examples)}")


        ############tokenized the dataset @@@@@@@@@@@@@@@@@@@@@@@

        # Define the extraction directory
        extract_dir = os.path.join("artifact", "data")
        # Ensure the directory exists
        os.makedirs(extract_dir, exist_ok=True)

        # Download and extract the model
        path_to_zip = tf.keras.utils.get_file(
            f"{self.tokenizer_model_name}.zip",
            f"https://storage.googleapis.com/download.tensorflow.org/models/{self.tokenizer_model_name}.zip",
            cache_dir=".artifact", cache_subdir="data", extract=True
        )

        # Move extracted files to the desired folder
        extracted_folder = path_to_zip.replace(".zip", "")  # Default extraction path
        try:
            if os.path.exists(extracted_folder):
                for file in os.listdir(extracted_folder):
                    shutil.move(os.path.join(extracted_folder, file), os.path.join(extract_dir, file))
        except  shutil.Error as e:
            logger.info(f"shutil error: {e}")
 

        # Get the extracted folder path
        model_path = os.path.join(extract_dir, self.tokenizer_model_name)

        # Verify extracted files
        logger.info(f"Extracted model files:, {os.listdir(model_path)}")

        # Load the model
        self.tokenizers = tf.saved_model.load(model_path) 
        train_data = self.train_examples
        test_data = self.val_examples
        tokenizer = self.tokenizers
        #return train_data,test_data,tokenizer
    

    def tokenized_data(self,MAX_TOKENS,BUFFER_SIZE,BATCH_SIZE):

        def prepare_batch(pt, en):
            pt = self.tokenizers.pt.tokenize(pt)      # Output is ragged.
            pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
            pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

            en = self.tokenizers.en.tokenize(en)
            en = en[:, :(MAX_TOKENS+1)]
            en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
            en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

            return (pt, en_inputs), en_labels
        
        def make_batches(ds):
            return (
                ds
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(prepare_batch, tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        # Create training and validation set batches.
        train_batches = make_batches(self.train_examples)
        val_batches = make_batches(self.val_examples)
        tokenizer = self.tokenizers
        return train_batches,val_batches,tokenizer

