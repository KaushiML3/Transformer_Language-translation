from pipeline.data_preprocessing_pipe import data_preprocess
from pipeline.model_train_pipe import model_training
from pipeline.inference_pipe import Translator,ExportTranslator
from util import create_folder_path
from custom_logger import setup_logger

import tensorflow as tf
import os


logger=setup_logger("training")

logger.info(f"Num GPUs Available: , {len(tf.config.list_physical_devices('GPU'))}")

if __name__=="__main__":

    ## data loading params
    dataset_path='ted_hrlr_translate/pt_to_en'
    tokenizer_model_name="ted_hrlr_translate_pt_en_converter"
    train_size=20000
    test_size=5000

    ## tokenizer params
    MAX_TOKENS=128
    BUFFER_SIZE=2000
    BATCH_SIZE=96


    ## training params
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    epochs=30


    #### version ###
    model_version=1
    
    try:
        logger.info("Start the data preprocessing")
        data_preproc=data_preprocess(dataset_path,tokenizer_model_name,train_size,test_size)
        logger.info("Start the data tokenizing")
        train,val,tokenizers=data_preproc.tokenized_data(MAX_TOKENS,BUFFER_SIZE,BATCH_SIZE)

        logger.info(f"Start the model training number of epoth {epochs}")
        model=model_training(num_layers,d_model,dff,num_heads,dropout_rate,epochs, train,val,tokenizers)
        logger.info("Model training done")

    except Exception as e:
        logger.error(f"Exception error: {str(e)}",exc_info=True)


    try:
        logger.info("Model Translatore start")
        #transformer is the model and tokenizer is previouse define.
        translator = Translator(tokenizers,model)
        inferenc_translator = ExportTranslator(translator)

        object_path= os.path.join("artifact","model",f"version_{model_version}")
        create_folder_path(object_path)
        # save the translator and model
        tf.saved_model.save(inferenc_translator, export_dir=object_path)
        logger.info(f"Saved the model in {object_path}")

    except Exception as e:
        logger.error(f"Exception error: {str(e)}",exc_info=True)

    #reloaded = tf.saved_model.load(save_object_path)




   




