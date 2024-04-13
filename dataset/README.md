# Explanation of the datasets in this folder:
## Original sets:
### (This section describe all original dataset, with some modifications for our purpose)
combined_extra_set.csv: from https://github.com/ChrisAHolland/ML-SQL-Injection-Detector/tree/master, duplicates are removed, yielding 4655 unique samples, all of them are sqli samples
Modified_SQL_Dataset.csv: source is https://www.kaggle.com/datasets/sajid576/sql-injection-dataset, 30905 samples in total, 19537 benign samples, and 11382 malicious sqli samples
extracted_sqli_no_dupe.csv: source is https://github.com/Morzeux/HttpParamsDataset, 31067 samples in total, contains 19304 benign samples, and 11763 malicious samples. However, it is mixed with other type of attacks, including XSS, command injection, and path traversal attacks. Further, benign samples has no labeling as to which type of attack it is from, therefore, only sqli malicious samples are extracted. The duplicates are also removed, resulted in 10464 malicious sqli samples
SQL_injection_only_dataset.csv: this is the same as Modified_SQL_Dataset.csv, but with all benign samples removed, having a total of 11382 malicious samples in this set only

## Combined sets:
### (This section described the dataset used, which is a combination of the datasets above. In addition, combined_extra_set.csv is split into two part randomly and equally, and then one part is added to each of the following 2 datasets)
first_dataset.csv: It is a combined set of malicious sqli samples in Modified_SQL_Dataset.csv and first part of combined_extra_set.csv, combined_extra_set.csv has a total of 13709 malicious samples
second_dataset.csv: A combination of all of extracted_sqli_no_dupe.csv and the second part of combined_extra_set.csv, it has a total of 12792 malicious samples

## Generated sets:
### (This section describe all datasets with generated data, including those with metrics, or the generated result by the model. As there are 2 datasets, each with its own generated data, it is replaced by N_all to indicate first_dataset and second_dataset respectively)
N_all_base_metrics.csv: This contains the data from 1 of the 2 combined datasets, then the base metrics as described in the paper are generated and stored. Here, base metrics means the metrics generated directed from the data in the dataset, and not from the generated output by the model. This is useful for the training process, by giving the model extra information to work for in the prompt.
N_all_pretrained_metrics.csv: This is the metrics for the generated queries by the pretrained model. The pretrained model is the model used for training in our paper prior to any training as described in our paper, the purpose is for comparing the trained model as described in our paper to this, as a control, such that we can see whether the method can improve the performance of the base model.
N_all_trained_metrics_X.csv: X here is the number of training iterations the trained model has went through as described in the paper using our approach. The metrics here are calculated using the generated queries by the model after X iterations.