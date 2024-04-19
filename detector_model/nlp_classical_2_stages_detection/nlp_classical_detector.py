import pickle, os
from pathlib import Path
from official.nlp import optimization  # to create AdamW optimizer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tqdm import tqdm

class classical_nlp_2_stages:
    epochs = 10
    steps_per_epoch = 6122
    num_train_steps = (steps_per_epoch * epochs) 
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    batch_size = 16
    device_mode = 'CPU'

    #classical_detector = pickle.load(open(os.getcwd() + '/detector_model/nlp_classical_2_stages_detection/classical_model_full.model', 'rb'))
    classical_detector = None
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')

    with tf.device(device_mode):
        nlp_detector_model = tf.keras.models.load_model(Path('detector_model/nlp_classical_2_stages_detection/trained_nlp_model.keras').resolve(), 
                custom_objects={'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer, 'WarmUp': optimization.WarmUp})

    def __init__(self) -> None:
        self.pipeline = pickle.load(open(os.getcwd() + '/detector_model/nlp_classical_2_stages_detection/classical_model_full.model', 'rb'))


    def classical_detector(self, dataset: list[str]) -> list[int]:
        """
            classical detector for a given list of string.

            Params:
            dataset (list[str]): list of queries/strings provided for detection of sqli

            Returns:
            first_stage_y_pred (list[int]): detection result of the provided list, each value in the list corresponds to those of the provided dataset
        """
        first_stage_y_pred = self.pipeline.predict(dataset)
        return first_stage_y_pred.tolist()


    def nlp_detector(self, dataset: list[str]) -> list[int]:
        """
            nlp detector for a given list of string.

            Params:
            dataset (list[str]): list of queries/strings provided for detection of sqli

            Returns:
            y_pred_np (list[int]): detection result of the provided list, each value in the list corresponds to those of the provided dataset
        """
        with tf.device(classical_nlp_2_stages.device_mode):

            samples = [dataset[i:i+classical_nlp_2_stages.batch_size] for i in range(0, len(dataset), classical_nlp_2_stages.batch_size)]

            sqli_res = []
            for step, batch in enumerate(tqdm(samples)):
                temp_res = classical_nlp_2_stages.nlp_detector_model.predict(batch, verbose = False)      
                temp_res = tf.cast(tf.sigmoid(temp_res) > 0.5, tf.int32).numpy()
                
                sqli_res += np.reshape(temp_res, (len(temp_res))).tolist()

            # pred_dataset = tf.data.Dataset.from_tensor_slices(
            #     (dataset)).batch(classical_nlp_2_stages.batch_size)

            # pred_dataset = pred_dataset.cache()

            # first_stage_pos_preds = pred_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

            # y_pred = classical_nlp_2_stages.nlp_detector_model.predict(first_stage_pos_preds, verbose = False)
            #y_pred_np = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.int32).numpy()

            return sqli_res
    