from tqdm import tqdm
import torch, time, os, gc, sys
import requests

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, modeling_utils
from generate_metrics import metrics_generator

class trained_model_tester:
    max_length = 256
    batch_size = 16

    @staticmethod
    def __get_sample_with_prompt(sample: str, validity: int, is_sqli: int, evaded: int) -> str:
        """format the sample to include the prompt and the sample, which should be a sqli query, though it may not necessarily fulfill all criteria

        Args:
            sample (str): the sqli query sample
            validity (int): if the given sample is a valid SQL query
            is_sqli (int): if the given sample is a sqli injection query
            evaded (int): if the given sample can evade detection, here, we do not check for the type of evasion performed, as such, the evaded
                result can be based on any combination of detectors used

        Returns:
            str: the prompt with the sample attached to it
        """

        if(validity == 1 and is_sqli == 1 and evaded == 1):
            prompt_template = 'Query: {0} \n for: This query can evade SQL injection detection, generate a SQL injection query that can evade detection based on the given query, when you are done, terminate the generated query'
        elif(validity == 1 and is_sqli == 1):
            prompt_template = 'Query: {0} \n for: This is an SQL injection query, modify the query to evade SQL injection detection, when you are done, terminate the generated query'
        elif(validity == 1):
            prompt_template = 'Query: {0} \n for: This is an SQL query, modify it to be an SQL injection query that can evade detection, when you are done, terminate the generated query'
        else:
            prompt_template = 'Query: {0} \n for: This is an invalid SQL query, modify it so that it becomes a valid SQL injection query, when you are done, terminate the generated query'

        return prompt_template.format(sample)
    
    @classmethod
    def generate_prompts_with_samples(cls, samples: list[str], metrics_list: dict[list]) -> list[str]:
        """given a list of samples, add prompts to it according to the metrics given, then return the list with the prompts

        Args:
            samples (list[str]): the sqli query sample
            metrics_list (dict[list]): each key is the metric name, here, it should included validity, is_sqli, and evaded, which is any type of
                evasion generated by some combination of detectors. The list contains the metric for each of the sample given

        Returns:
            list[str]: _description_
        """
        samples_with_prompt = []
        for i in range(len(metrics_list['validity'])):
            sample = samples[i]
            sample_validity = metrics_list['validity'][i]
            sample_is_sqli = metrics_list['is_sqli'][i]
            sample_evaded = metrics_list['evaded'][i]
            samples_with_prompt.append(cls.__get_sample_with_prompt(sample, sample_validity, sample_is_sqli, sample_evaded))

        return samples_with_prompt
    
    @classmethod
    def model_generation(cls, tokenizer: AutoTokenizer, model: AutoModel, samples_with_prompt: list[str], verbose: bool = True) -> list[str]:
        """_summary_

        Args:
            tokenizer (AutoTokenizer): _description_
            model (AutoModel): _description_
            samples_with_prompt (list[str]): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            list[str]: _description_
        """
        samples = [samples_with_prompt[i:i+cls.batch_size] for i in range(0, len(samples_with_prompt), cls.batch_size)]

        model_generated_samples = []
        for step, batch in enumerate(tqdm(samples, disable= not verbose)):
            prompt_output = tokenizer(batch, 
                                return_tensors = 'pt', truncation=True, padding='max_length', max_length=cls.max_length).to('cuda')
            predicted = model.generate(prompt_output['input_ids'], attention_mask = prompt_output['attention_mask'], 
                                        max_new_tokens=cls.max_length, pad_token_id=tokenizer.eos_token_id)
            predicted_str_list = tokenizer.batch_decode(predicted, skip_special_tokens=True, trauncation=True)
            model_generated_samples += predicted_str_list

        return model_generated_samples


if __name__ == '__main__':
    dataset = pd.read_csv('dataset/first_dataset_trained_metrics_8.csv')
    dataset = dataset.rename(columns={'generated_query': 'payload'})
    dataset = dataset.astype({'payload': 'U'})

    #TEST
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    train_dataset = dataset.head(int(len(dataset) * 0.9))
    test_dataset = dataset.tail(len(dataset) - len(train_dataset))
    dataset = test_dataset.reset_index(drop=True)
    #TEST end
    
    payloads = dataset['payload'].squeeze().to_list()
    # res = requests.get('https://k0ahsdpgv5.execute-api.us-east-1.amazonaws.com/F5/test?user=' + '1\' OR 1=1;#')
    # res = res.json()
    # print(res)
    # print('message' in res)
    # res = requests.get('https://k0ahsdpgv5.execute-api.us-east-1.amazonaws.com/F5/test?user=' + '1')
    # res = res.json()
    # print(res)
    # print('message' in res)
    #res = WAFDetector.predict('https://k0ahsdpgv5.execute-api.us-east-1.amazonaws.com/F5/test?user=', payloads[:16])
    batch_size = 1

    #payloads = payloads[:256]
    print(dataset)
    print(dataset.columns)
    metrics_list = dataset[['validity', 'is_sqli', 'cnn_evaded_detection']]

    print(metrics_list)
    metrics_list.to_dict()
    metrics_list['evaded'] = metrics_list.pop('cnn_evaded_detection')
    print([name for name in metrics_list])

    tokenizer = AutoTokenizer.from_pretrained('juierror/flan-t5-text2sql-with-schema-v2')

    sqli_evasion_model = AutoModelForSeq2SeqLM.from_pretrained('juierror/flan-t5-text2sql-with-schema-v2').to('cuda')
    from detector_model.waf_detector.detector import WAFDetector
    #print(sqli_evasion_model.get_memory_footprint())


    samples_with_prompts = trained_model_tester.generate_prompts_with_samples(dataset['payload'].squeeze().to_list(), metrics_list)
    generated_output = trained_model_tester.model_generation(tokenizer, sqli_evasion_model, samples_with_prompts)

    dataset['generated_query'] = pd.Series(generated_output)
    metrics_result = metrics_generator.calc_data_metrics(dataset['generated_query'].squeeze().to_list())

    exit()
    metric_result_name_list = [metric for metric in metrics_result]
    #metric_to_be_saved = '_evaded_detection_any'
    for metric in metric_result_name_list:
        if('_evaded_detection_any' not in metric and metric not in ['validity', 'is_sqli']):
            del metrics_result[metric]
        elif('_evaded_detection_any' in metric):
            metrics_result[metric[:-len('_any')]] = metrics_result.pop(metric)

    metrics_result_df = pd.DataFrame(list(zip(dataset['payload'].squeeze().to_list(), dataset['generated_query'])), columns=['original_query', 'generated_query'])
    for metric, result in metrics_result.items():
        metrics_result_df[metric] = result
    metrics_result_df.to_csv('dataset/testing_second_trained_metrics.csv')
    print(dataset)