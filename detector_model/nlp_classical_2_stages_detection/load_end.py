import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization  # to create AdamW optimizer
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def evaluate(y_test, y_pred):
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

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16

"""file = 'datasets/SQLiV3.tsv'
dataset = pd.read_csv(file, sep='\t', engine='python')
train, test = train_test_split(
        dataset, test_size=0.2)
x_train = train['payload'].values
x_test = test['payload'].values
y_train = train['label'].values
y_test = test['label'].values"""

file = 'Modified_SQL_Dataset.csv' #test
dataset = pd.read_csv(file, engine='python')
train, test = train_test_split(
        dataset, test_size=0.2)
x_train = train['Query'].values
x_test = test['Query'].values
y_train = train['Label'].values
y_test = test['Label'].values

# Train and evaluate the first stage
def first_stage_passive_aggressive():

    #pickle.dump(pipeline, open('classical_model_full.model', 'wb'))
    pipeline = pickle.load(open(os.getcwd() + '/nlp_classical_2_stages_detection/classical_model_full.model', 'rb'))

    first_stage_y_pred = pipeline.predict(x_test)
    first_stage_y_pred = np.asarray(first_stage_y_pred)
    first_stage_positive_preds = np.take(x_test, np.where(first_stage_y_pred == 1)).tolist()[0]
    first_stage_positive_preds_true_labels = np.take(y_test, np.where(first_stage_y_pred == 1)).tolist()[0]

    return first_stage_positive_preds, first_stage_positive_preds_true_labels


def second_stage(first_stage_positive_preds, first_stage_positive_preds_true_labels):
    batch_size = 8
    epochs = 10
    steps_per_epoch = x_test.shape[0]

    num_train_steps = (steps_per_epoch * epochs) 
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')

    second_stage_classifier_model = tf.keras.models.load_model(os.getcwd() + '/nlp_classical_2_stages_detection/trained_nlp_model.keras', 
            custom_objects={'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer})
    
    pred_dataset = tf.data.Dataset.from_tensor_slices(
        (first_stage_positive_preds, first_stage_positive_preds_true_labels)).batch(batch_size)

    pred_dataset = pred_dataset.cache()

    first_stage_pos_preds = pred_dataset.prefetch(buffer_size=AUTOTUNE)

    y_pred = second_stage_classifier_model.predict(first_stage_pos_preds)#.take(2))
    y_pred_np = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.int32).numpy()
    X, y_true = zip(*first_stage_pos_preds.unbatch())# take(2).
    y_true_np = [y.numpy() for y in y_true]

    result = evaluate(y_true_np, y_pred_np)
    print(result)


first_stage_y_pred, first_stage_positive_preds_true_labels = first_stage_passive_aggressive()
second_stage(first_stage_y_pred, first_stage_positive_preds_true_labels)
