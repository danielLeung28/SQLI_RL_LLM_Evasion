import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle, os, math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
#import torch
from pathlib import Path
from official.nlp import optimization  # to create AdamW optimizer
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
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')

    with tf.device('/CPU:0'):
        second_stage_classifier_model = tf.keras.models.load_model(Path('nlp_classical_2_stages_detection/trained_nlp_model.keras').resolve(), 
                custom_objects={'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer, 'WarmUp': optimization.WarmUp})
        
    def __init__(self, batch_size = 8, device_mode = 'CPU:0') -> None:
        #print(os.getcwd())
        #self.tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        #print(torch.cuda.is_available())
        self.pipeline = pickle.load(open('nlp_classical_2_stages_detection/classical_model_full.model', 'rb'))
        #self.sqli_detection_model = AutoModelForSequenceClassification.from_pretrained('nlp_classical_2_stages_detection/trained_nlp_model', local_files_only=True).to('cuda')
        self.batch_size = batch_size
        self.device_mode = device_mode
        #exit()

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        return result

    # Train and evaluate the first stage
    def first_stage_passive_aggressive(self, get_pred=False):

        #pickle.dump(pipeline, open('classical_model_full.model', 'wb'))\

        """first_stage_y_pred = pipeline.predict(x_test)
        first_stage_y_pred = np.asarray(first_stage_y_pred)
        first_stage_positive_preds = np.take(x_test, np.where(first_stage_y_pred == 1)).tolist()[0]
        first_stage_positive_preds_true_labels = np.take(y_test, np.where(first_stage_y_pred == 1)).tolist()[0]"""

        first_stage_y_pred = self.pipeline.predict(self.x)
        first_stage_y_pred = np.asarray(first_stage_y_pred)
        #first_stage_positive_preds_true_labels = np.take(self.y, np.where(first_stage_y_pred == 1)).tolist()[0]
        #print(self.x)
        #print(first_stage_y_pred, first_stage_positive_preds)

        if(not get_pred):
            first_stage_positive_preds = np.take(self.x, np.where(first_stage_y_pred == 1)).tolist()[0]
            return first_stage_positive_preds
        else:
            return first_stage_y_pred


    def second_stage(self, first_stage_positive_preds):
        with tf.device(self.device_mode):
            batch_size = 16
            pred_dataset = tf.data.Dataset.from_tensor_slices(
                (first_stage_positive_preds)).batch(batch_size)

            pred_dataset = pred_dataset.cache()

            first_stage_pos_preds = pred_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

            y_pred = classical_nlp_2_stages.second_stage_classifier_model.predict(first_stage_pos_preds, verbose = False)#.take(2))
            y_pred_np = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.int32).numpy()

        # if(len(first_stage_positive_preds) > self.batch_size):
        #     y_pred = None
        #     for index, batch in enumerate(tqdm([first_stage_positive_preds[i:i+self.batch_size] for i in range(0, len(first_stage_positive_preds), self.batch_size)])):
        #     #for i in range(math.ceil(len(first_stage_positive_preds)/self.batch_size)):
        #         #first_stage_tokens = self.tokenizer(first_stage_positive_preds[i*self.batch_size:(i+1)*self.batch_size], return_tensors='pt', padding=True, truncation=True).to('cuda')

        #         first_stage_tokens = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to('cuda')
        #         y_pred_temp = self.sqli_detection_model(input_ids= first_stage_tokens['input_ids'], attention_mask=first_stage_tokens['attention_mask'])
        #         if(y_pred is None):
        #             y_pred = np.argmax(y_pred_temp['logits'].detach().cpu().numpy(), axis=1)
        #         else:
        #             y_pred = np.concatenate((y_pred, np.argmax(y_pred_temp['logits'].detach().cpu().numpy(), axis=1)), axis=0)
        # else:
        #     tokenized_inputs = self.tokenizer(first_stage_positive_preds, return_tensors='pt', padding=True, truncation=True).to('cuda')
        #     y_pred_temp = self.sqli_detection_model(input_ids= tokenized_inputs['input_ids'], attention_mask=tokenized_inputs['attention_mask'])
        #     y_pred = np.argmax(y_pred_temp['logits'].detach().cpu().numpy(), axis=1)

        #print(y_pred_np)
        #print(type(y_pred_np))
            return y_pred_np
        return y_pred

        #X, y_true = zip(*first_stage_pos_preds.unbatch())# take(2).
        #y_true_np = [y.numpy() for y in y_true]

        #print(y_true_np)
        #result = self.evaluate(y_true_np, y_pred_np)
        #print(result)

        #return result

    def eval_overall(self):
        first_stage_y_pred = self.first_stage_passive_aggressive()
        if(len(first_stage_y_pred) == 0):
            return [[0]]
        return self.second_stage(first_stage_y_pred)
    
    def change_data(self, x, y):
        self.x = x
        self.y = y


if (__name__ == '__main__'):
    #AUTOTUNE = tf.data.AUTOTUNE
    #batch_size = 16
    
    file = 'nlp_classical_2_stages_detection/datasets/SQLiV3.tsv'
    dataset = pd.read_csv(file, sep='\t', engine='python')
    train, test = train_test_split(
            dataset, test_size=0.2)
    x_train = train['payload'].values
    x_test = test['payload'].values
    y_train = train['label'].values
    y_test = test['label'].values
    print(x_test[:10])

    classical_nlp_test = classical_nlp_2_stages()
    classical_nlp_test.change_data(x_test, y_test)
    test_res = classical_nlp_test.second_stage(x_test.tolist()[:128])
    print(len(test_res))
    print(test_res)
    exit()
    print(classical_nlp_test.evaluate(y_test, test_res))

    #first_stage_y_pred, first_stage_positive_preds_true_labels = first_stage_passive_aggressive()
    #second_stage(first_stage_y_pred, first_stage_positive_preds_true_labels)
