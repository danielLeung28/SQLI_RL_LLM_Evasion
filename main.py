import os, subprocess, gc, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import ruamel.yaml
import pandas as pd
from generate_metrics import metrics_generator
from test_trained_model import trained_model_tester
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from detector_model.cnn.detector import CNNDetector
from detector_model.lstm.detector import LSTMDetector

tokenizer = AutoTokenizer.from_pretrained('cssupport/t5-small-awesome-text-to-sql')

base_project_folder = os.getcwd()

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
with open(base_project_folder + '/training_config/nlp_rl_iterative.yml') as file:
    file_info = yaml.load(file)

base_dataset_list = {}
dataset_name = ['first_dataset', 'second_dataset']
chosen_metrics_list = ['validity', 'is_sqli', 'ensemble_strict_evaded_detection']

largest_iter_done = 0
for val in os.listdir(base_project_folder + '/trained_model'):
    if('model_iter_') in val:
        model_iter_name = val.split(sep='_')
        if(largest_iter_done < int(model_iter_name[-1])):
            largest_iter_done = int(model_iter_name[-1])

dataset = pd.read_csv('dataset/first_dataset_base_metrics.csv')
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

train_dataset = dataset.head(int(len(dataset) * 0.9))
test_dataset = dataset.tail(len(dataset) - len(train_dataset))

metrics_list = test_dataset[chosen_metrics_list].to_dict('list')
metrics_list['evaded'] = metrics_list.pop(chosen_metrics_list[-1])

train_dataset.to_csv('dataset/training_dataset.csv', index=False)
n_iters = 16
model_path_folder = os.getcwd() + '/rl4lm_exps/rl4lm_experiment'

trained_model_tester.batch_size = 64
metrics_generator.tester.batch_size = 64
metrics_generator.tester.device_mode = '/GPU:0'
metrics_generator.url = "https://k0ahsdpgv5.execute-api.us-east-1.amazonaws.com/F5/test?user="

def eval_model(sqli_evasion_model, tokenizer, query_eval, metrics_for_queries, verbose = True):
    samples_in_prompts = trained_model_tester.generate_prompts_with_samples(query_eval, metrics_for_queries)
    generated_sqli = trained_model_tester.model_generation(tokenizer, sqli_evasion_model, samples_in_prompts)
    
    metrics_result = metrics_generator.calc_data_metrics(generated_sqli, verbose=verbose)
    metric_result_name_list = [metric for metric in metrics_result]
    for metric in metric_result_name_list:
        if('_evaded_detection_any' not in metric and metric not in ['validity', 'is_sqli']):
            del metrics_result[metric]
        elif('_evaded_detection_any' in metric):
            metrics_result[metric[:-len('_any')]] = metrics_result.pop(metric)

    metrics_result_df = pd.DataFrame(list(zip(query_eval, generated_sqli)), columns=['original_query', 'generated_query'])
    for metric, result in metrics_result.items():
        metrics_result_df[metric] = result

    return metrics_result_df


file_info['datapool']['args']['dataset'] = 'dataset/training_dataset.csv'
for i in range(n_iters):
    shutil.rmtree(model_path_folder, ignore_errors=True)

    if(largest_iter_done < i+1):
        with open(base_project_folder + '/training_config/nlp_rl_iter_' + str(i+1) + '.yml', 'w') as file:
            yaml.dump(file_info, file)

        print('-----------------------training iteration ' + str(i+1) + ' starts----------------------------')
        args_str = 'python RL4LMs/scripts/training/train_text_generation.py --config_path ' + 'training_config/nlp_rl_iter_' + str(i+1) + '.yml'
        subprocess.call(args_str.split(sep=' '))

        print('-----------------------training iteration ' + str(i+1) + ' done----------------------------')
        os.rename(os.path.join(model_path_folder, 'model'), base_project_folder + '/trained_model/model_iter_' + str(i+1))


        print('-----------------------evaluating model ' + str(i+1) + '----------------------------')

        sqli_evasion_model = AutoModelForSeq2SeqLM.from_pretrained(base_project_folder + '/trained_model/model_iter_' + str(i+1)).to('cuda')

        metrics_result_df = eval_model(sqli_evasion_model, tokenizer, test_dataset['original_query'].squeeze().to_list(), metrics_list)
        metrics_result_df.to_csv('dataset/' + dataset_name[0] + '_test_trained_metrics_' + str(i) + '.csv', index=False)

        print('---------------------------test set eval done---------------------------')
        
        train_dataset_iter = train_dataset.sample(frac=0.5)

        metrics_list_train = train_dataset[chosen_metrics_list].to_dict('list')
        metrics_list_train['evaded'] = metrics_list_train.pop(chosen_metrics_list[-1])
        metrics_result_df = eval_model(sqli_evasion_model, tokenizer, train_dataset['original_query'].squeeze().to_list(), metrics_list_train, verbose=False)

        train_dataset_temp = train_dataset.copy()
        train_dataset_temp.update(metrics_result_df)
        print(train_dataset_temp)
        print(train_dataset)

        # #randomly add 10% of the dataset such that the model will not be overfitted on its own generated output and the original dataset
        # metrics_result_df = pd.concat([metrics_result_df, train_dataset['generated_querys'].sample(frac=0.1)])
        train_dataset_temp.to_csv('dataset/training_dataset.csv', index=False)

        #remove all model loaded into gpu memory after evaluation to reduce vram usage
        del sqli_evasion_model
        LSTMDetector.wafInterface = None
        CNNDetector.wafInterface = None
        gc.collect()
        torch.cuda.empty_cache()

    file_info['alg']['policy']['args']['model_name'] = base_project_folder + '/trained_model/model_iter_' + str(i+1)

    #file_info['datapool']['args']['dataset'] = 'dataset/' + dataset_name[starting_dataset] + '_trained_metrics_' + str(i) + '.csv'