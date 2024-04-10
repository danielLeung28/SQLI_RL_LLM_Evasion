from tqdm import tqdm
import torch, time, os, gc
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, modeling_utils
from nlp_classical_2_stages_detection.load_end_module import classical_nlp_2_stages
from official.nlp import optimization  # to create AdamW optimizer
from sql_validator import checkSQLValidity

# register(
#     id = 'sqli_env',
#     entry_point= 'sqli_env:sqlInjectionAttackEnv'
# )

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#dataset = pd.read_csv('SQL_injection_only_dataset.csv')
dataset = pd.read_csv('Modified_SQL_Dataset_with_metrics.csv')
# dataset = pd.read_csv('nlp_classical_2_stages_detection/datasets/SQLiV3.tsv', sep='\t')

dataset = dataset[(dataset['validity'] == 1) & (dataset['is_sql_injection'] == 1)].reset_index(drop=True)
# print(dataset)
# exit()
#print(dataset)
#exit()
tester = classical_nlp_2_stages(device_mode='/CPU:0')

tokenizer = AutoTokenizer.from_pretrained('cssupport/t5-small-awesome-text-to-sql')

#sqli_evasion_model = AutoModelForSeq2SeqLM.from_pretrained('model').to('cuda')
sqli_evasion_model = AutoModelForSeq2SeqLM.from_pretrained('rl4lm_exps/rl4lm_experiment/model').to('cuda')
# sqli_evasion_model = AutoModelForSeq2SeqLM.from_pretrained('./model_iter_1').to('cuda')

max_length = 256
samples_per_epoch = 16
mini_batch_size = 32
training_round = 8

dataset = dataset.rename(columns={'generated_query': 'payload'})
print(dataset)

# samples = [temp_test[i:i+samples_per_epoch] for i in range(0, len(temp_test), samples_per_epoch)]

# nlp_only_detection_org = []
# for step, batch in enumerate(tqdm(samples)):
#     temp_res = tester.second_stage(batch)
#     nlp_only_detection_org += np.reshape(temp_res, (len(temp_res))).tolist()


# print('org set metrics:', nlp_only_detection_org.count(1), nlp_only_detection_org.count(1) / len(nlp_only_detection_org))


# prompt_template = '<schema></schema><question>modify this query to evade SQL injection detection: {0}</question><sql>'
# prompt_template = "<schema></schema><question>modify this query to evade SQL injection detection: -8370'   )    )    or   (  3501  =  6407  )  *6407 and    (    (   'agcm' like 'agcm</question><sql>"
#prompt_template = 'Query: {0} \n for: modify this query to evade SQL injection detection'
#prompt_template = 'Query: {0} \n for: This is an SQL injection query, modify it so that it can evade detection'

#test
def get_samples_with_prompt(row):
    row_dict = row.to_dict()

    if(row_dict['validity'] == 1 and row_dict['is_sql_injection'] == 1 and row_dict['evaded_detection'] == 1):
        prompt_template = 'Query: {0} \n for: This query can evade SQL injection detection, create more SQL injection query that can evade detection, while maintaining the same meaning as the given query, when you are done, terminate the generated query'
    elif(row_dict['validity'] == 1 and row_dict['is_sql_injection'] == 1):
        prompt_template = 'Query: {0} \n for: This is an SQL injection query, modify the query to evade SQL injection detection, it should be a SQL injection query that has the same meaning as the given query, when you are done, terminate the generated query'
    elif(row_dict['validity'] == 1):
        prompt_template = 'Query: {0} \n for: This is an SQL query, modify it to be an SQL injection query that can evade detection, when you are done, terminate the generated query'
    else:
        prompt_template = 'Query: {0} \n for: This is an invalid SQL query, modify it so that it becomes a valid SQL injection query, when you are done, terminate the generated query'
    
    return prompt_template.format(row['payload'])

dataset['payload_with_prompt'] = dataset.apply(get_samples_with_prompt, axis=1)
data_with_prompt = dataset['payload_with_prompt'].squeeze().to_list()
original_dataset_list = dataset['payload'].squeeze().to_list()
# print(len(data_with_prompt), data_with_prompt[0:2])
# exit()

#prompt_template = 'Query: {0} \n for: This is an SQL injection query, modify it so that it can evade detection, when you are done, terminate the generated query'

# model = AutoModelForCausalLM.from_pretrained(
#     'rl4lm_exps/rl4lm_experiment/model', config='rl4lm_exps/rl4lm_experiment/model/config.json',
#     device_map='auto',
# )

# tokenizer_list = ['PipableAI/pip-sql-1.3b', 'distilbert/distilgpt2', 'facebook/bart-base', 'distilbert-base-uncased']
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_list[1], padding_size = 'left',
#     device_map='auto')
# tokenizer.pad_token = tokenizer.eos_token

samples = data_with_prompt

samples = [samples[i:i+mini_batch_size] for i in range(0, len(samples), mini_batch_size)]
sql_validator = checkSQLValidity()

generated_sqli = []

for step, batch in enumerate(tqdm(samples)):
    #start_time = time.time()
    #tokenized_outputs = tokenizer([samples])
    #batch = [prompt_template.format(sample) for sample in batch]
    #print(batch, type(batch))
    prompt_output = tokenizer(batch, 
                        return_tensors = 'pt', truncation=True, padding='max_length', max_length=max_length).to('cuda')
    predicted = sqli_evasion_model.generate(prompt_output['input_ids'], attention_mask = prompt_output['attention_mask'], 
                                max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id)
    predicted_str_list = tokenizer.batch_decode(predicted, skip_special_tokens=True, trauncation=True)
    # print(prompt_output)
    # print(predicted)
    generated_sqli += predicted_str_list


del sqli_evasion_model
gc.collect()
torch.cuda.empty_cache()


validity = sql_validator.check_SQL_validity(generated_sqli)
tester.change_data(generated_sqli, None)
#detected = self.eval_model.eval_overall()[0][0]

print('gen output len', len(generated_sqli))
samples = [generated_sqli[i:i+len(generated_sqli)//64] for i in range(0, len(generated_sqli), len(generated_sqli)//64)]
print('split len', sum([len(batch) for batch in samples]))
nlp_only_detection = []
for step, batch in enumerate(tqdm(samples)):
    temp_res = tester.second_stage(batch)
    nlp_only_detection += np.reshape(temp_res, (len(temp_res))).tolist()

#nlp_only_detection = np.array(nlp_only_detection)
#print(nlp_only_detection)
#return a list of detected sample, length of 0 means not detected, and 1 means detected
classical_only_detection = tester.first_stage_passive_aggressive(get_pred=True)

#print(validity[1]['result'])
validity = validity[1]['result']
#print(nlp_only_detection)
# print(classical_only_detection)
#print('reward val', validity, detected, type(validity), type(detected))

valid_list = []
is_sql_injection_list = []
is_valid_injection_list = []
evaded_detection_list = []
evaded_if_valid_sqli = []
evaded_overall = []


samples = dataset['payload'].to_list()
valid_sqli_evaded_query = []
#validity, is sql injection, evaded detection, is valid sql injection,
# evaded if valid and sql injection, overall evasion
for i in range(len(validity)):
    valid_list.append(validity[i])
    is_sql_injection_list.append(classical_only_detection[i])
    is_valid_injection_list.append(1 if validity[i] == 1 and classical_only_detection[i] == 1 else 0)
    evaded_detection_list.append(1-int(nlp_only_detection[i]))
    evaded_if_valid_sqli.append((1 if nlp_only_detection[i] < 1 else 0) if validity[i] == 1 and classical_only_detection[i] == 1 else np.nan)
    evaded_overall.append(1 if (validity[i] > 0 and nlp_only_detection[i] == 0 and classical_only_detection[i] == 1) else 0)


generated_sqli_df = list(zip(original_dataset_list, generated_sqli, valid_list, is_sql_injection_list, evaded_detection_list))
# generated_sqli_df = np.array([generated_sqli, valid_list, is_sql_injection_list, evaded_detection_list])
# generated_sqli_df = np.reshape(generated_sqli_df, newshape=(len(generated_sqli), 4))
generated_sqli_df = pd.DataFrame(generated_sqli_df, columns=['original_query', 'generated_query', 'validity', 'is_sql_injection', 'evaded_detection'])
generated_sqli_df.to_csv('generated_sqli_full_set.csv', index=False)

evaded_if_valid_sqli = np.array(evaded_if_valid_sqli)
#print(evaded_if_valid_sqli)
evaded_if_valid_sqli = evaded_if_valid_sqli[~np.isnan(evaded_if_valid_sqli)]
key, counts = np.unique(evaded_if_valid_sqli, return_counts=True)
evaded_if_valid_sqli_counter = dict(zip(key, counts))
print(valid_list.count(1), valid_list.count(1) / len(valid_list))
print(is_sql_injection_list.count(1), is_sql_injection_list.count(1) / len(is_sql_injection_list))
print(is_valid_injection_list.count(1), is_valid_injection_list.count(1) / len(is_valid_injection_list))
print(evaded_detection_list.count(1), evaded_detection_list.count(1) / len(evaded_detection_list))
print(evaded_if_valid_sqli_counter[1], evaded_if_valid_sqli_counter[1] / len(evaded_if_valid_sqli))
print(evaded_overall.count(1), evaded_overall.count(1) / len(evaded_overall))

# for i in range(training_round): #test trained model
#     samples = sqli_dataset_org['train'].train_test_split(train_size=samples_per_epoch)['train']
#     #print(len(samples['Query']), samples['Query'])
#     samples = np.array(samples['Query'])
#     samples = samples.reshape((samples_per_epoch//mini_batch_size, mini_batch_size))

    
#     for step, batch in enumerate(tqdm(samples)):
#         start_time = time.time()
#         #tokenized_outputs = tokenizer([samples])
#         batch = [prompt_template.format(sample) for sample in batch]
#         print(batch, type(batch))
#         prompt_output = tokenizer(batch, 
#                            return_tensors = 'pt', truncation=True, padding='max_length', max_length=max_length).to('cuda')
#         predicted = model.generate(prompt_output['input_ids'], attention_mask = prompt_output['attention_mask'], 
#                                    max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id)
#         predicted_str = tokenizer.batch_decode(predicted, skip_special_tokens=True, trauncation=True)
#         print(prompt_output)
#         print(predicted)
#         print(predicted_str)
#         exit()
        
#         res = model.generate(current_obs['prompt_input_encode_tokens'], attention_mask=current_obs['prompt_input_attention_mask_tokens'], 
#                              max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id)
#         print('generated', time.time() - start_time)
#         print(res)