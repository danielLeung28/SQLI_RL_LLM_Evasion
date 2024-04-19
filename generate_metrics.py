from tqdm import tqdm
import time, os, sys
import torch
import pandas as pd
import numpy as np
sys.path.append(os.getcwd() + '/detector_model/cnn')
sys.path.append(os.getcwd() + '/detector_model/lstm')

from detector_model.cnn.detector import CNNDetector
from detector_model.lstm.detector import LSTMDetector
from detector_model.nlp_classical_2_stages_detection.nlp_classical_detector import classical_nlp_2_stages

from sql_validator import checkSQLValidity

class metrics_generator:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    LSTMDetector.device = device
    CNNDetector.device = device
    sql_validator = checkSQLValidity()
    tester = classical_nlp_2_stages()
    batch_size = 32

    detector_model = {'cnn': CNNDetector.predict, 'lstm': LSTMDetector.predict, 'classical': tester.classical_detector}
    detector_list = [detector for detector in detector_model]
    detection_metric_types = ['evaded_detection_any', 'evaded_detection_given_valid_sqli', 'evaded_overall_valid_sqli']
    base_metric_list = ['validity', 'is_sqli', 'is_valid_sqli']

    #calculate ensemble metric, which including combining all detection model in certain ways
    @classmethod
    def calc_ensemble_detection(cls, calculated_metrics: dict[list], detector_count: int, detector_num: int, validity: int, is_sqli: int, strict_mode: bool = False) -> None:
        """calculate ensemble detection for the given sample and detectors used, as all the detection results are calculated, the sample itself
            does not need to be an argument in this function. Ensemble detection means that we use multiple detector to check the data, and come
            to a conclusion as to whether it is malicious. The normal strategy is half of all model needs to agree for it to be considered as 
            malicious, strict strategy is considering it as malicious even if only a single detector identified it as malicious, this is useful for
            evaluating the performance of the model across different detectors.

        Args:
            calculated_metrics (dict[list]): the dictionary list containing the calculated metrics information, with the ensemble detection
                result being added to this
            detector_count (int): number of detectors that detected the sample as malicious
            detector_num (int): total number of detectors used for detection of the sample
            validity (int): if the sample is valid
            is_sqli (int): if the sample is a sqli
            strict_mode (bool, optional): If strict strategy should be used for metric generation. Defaults to False, which is to use normal strategy.
        """

        ensemble_method_name = 'ensemble_strict' if strict_mode else 'ensemble_avg'
        metric_names = [ensemble_method_name + '_' + metric_type for metric_type in cls.detection_metric_types]
        if(metric_names[0] not in calculated_metrics):
            for metric_name in metric_names:
                calculated_metrics[metric_name] = []

        if(strict_mode):
            ensemble_detected = 0 if detector_count/detector_num == 0 else 1
        else:
            ensemble_detected = 0 if detector_count/detector_num < 0.5 else 1

        calculated_metrics[metric_names[0]].append(1-ensemble_detected)
        calculated_metrics[metric_names[1]].append((1 if ensemble_detected < 1 else 0) if validity == 1 and is_sqli == 1 else np.nan)
        calculated_metrics[metric_names[2]].append(1 if (validity > 0 and ensemble_detected == 0 and is_sqli == 1) else 0)
        

    #calculate all metrics based on the collected data
    @classmethod
    def calc_complex_metric(cls, validity_res: list[int], is_sqli_res: list[int], detector_res: dict[list]) -> dict[list]:
        """calculate all complex metrics based on the detection result, and the sample's validity and being a sqli query

        Args:
            validity_res (list[int]): a list of the validity of the samples used
            is_sqli_res (list[int]): a list of whether it is a sqli query for each of the samples used
            detector_res (dict[list]): dictionary with key being each of the detector used, and the list is the detection result for the specific detector
                for each of the supplied query.

        Returns:
            dict[list]: return a dictionary of the metrics generated, the key is the name of the metrics, and the list for the metric is the metric
                resulted generated for each of the sample result given to the function
        """

        calculated_metrics = {}
        for base_metric in cls.base_metric_list:
            calculated_metrics[base_metric] = []

        for detector in detector_res:
            for detection_metric_type in cls.detection_metric_types:
                calculated_metrics[detector + '_' + detection_metric_type] = []

        for i in range(len(validity_res)):
            sample_validity = validity_res[i]
            sample_is_sqli = is_sqli_res[i]
            #sample_detection_res = detector_res[i]
            sample_detection_res = {detector: res[i] for detector, res in detector_res.items()}
            #print(sample_validity, sample_is_sqli, sample_detection_res)

            calculated_metrics['validity'].append(sample_validity)
            calculated_metrics['is_sqli'].append(sample_is_sqli)
            calculated_metrics['is_valid_sqli'].append(1 if sample_validity == 1 and sample_is_sqli == 1 else 0)

            detector_count = 0
            for detector, res in sample_detection_res.items():
                metric_names = [detector + '_' + metric_type for metric_type in cls.detection_metric_types]

                calculated_metrics[metric_names[0]].append(1-res)
                calculated_metrics[metric_names[1]].append((1 if res < 1 else 0) if sample_validity == 1 and sample_is_sqli == 1 else np.nan)
                calculated_metrics[metric_names[2]].append(1 if (sample_validity > 0 and res == 0 and sample_is_sqli == 1) else 0)
                detector_count += res

            cls.calc_ensemble_detection(calculated_metrics, detector_count, len(detector_res), sample_validity, sample_is_sqli)
            cls.calc_ensemble_detection(calculated_metrics, detector_count, len(detector_res), sample_validity, sample_is_sqli, strict_mode=True)

        return calculated_metrics

    @classmethod
    def check_sqli(cls, queries: list[str], batch_size: int = 16, verbose: bool = True) -> list[int]:
        """check that for all of the samples given, whether each sample is a sqli, unlike other detection model used, this one is used exclusively
            to check that whether each sample is a sqli or not as a separate metric, rather than a detection metric

        Args:
            queries (list[str]): queries for generating sqli results.
            batch_size (int, optional): batch_size for sqli results generation. Defaults to 16, as the model uses a lot of memory, and higher batch size
                can cause problem.
            verbose (bool, optional): whether tqdm should show the progress bar. Defaults to True.

        Returns:
            list[int]: sqli result for the given queries
        """
        # samples = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]

        # sqli_res = []
        # for step, batch in enumerate(tqdm(samples, disable=not verbose)):
        #     temp_res = cls.tester.nlp_detector(batch)
            
        #     sqli_res += np.reshape(temp_res, (len(temp_res))).tolist()

        temp_res = cls.tester.nlp_detector(queries)
        
        sqli_res = np.reshape(temp_res, (len(temp_res))).tolist()

        return sqli_res

    @classmethod
    def compute_validity(cls, queries: list[str]) -> list[int]:
        """compute the validity of given queries

        Args:
            queries (list[str]): queries for generating validity results.

        Returns:
            list[int]: validity result for the given queries
        """
        return cls.sql_validator.check_SQL_validity(queries)[1]['result']

    @classmethod
    def generate_detection_result(cls, queries: list[str], detectors: list[str] = None, batch_size: int = None
                        , verbose: bool = True) -> dict[list]:
        """ generate detection result based on supplied queries, using the detectors implemented in the class

        Args:
            queries (list[str]): queries for generating detection results.
            detectors (list[str], optional): list of detectors to be used. Defaults to all detectors implemented.
            batch_size (int, optional): batch size for the detectors, useful for increasing speed, at the cost of higher mem usage. 
                Defaults to None, which uses the class variable batch_size instead.
            verbose (bool, optional): whether tqdm should show the progress bar. Defaults to True.

        Raises:
            ValueError: raised if invalid detector values are provided, where detector values provided are for detector that does not exist.

        Returns:
            dict[list]: dictionary with key being each of the detector used, and the list is the detection result for the specific detector
                for each of the supplied queries.
        """
        
        if(batch_size is None):
            batch_size = cls.batch_size

        if(detectors is None):
            detectors = detector_list

        invalid_detector = []
        for detector in detectors:
            if(detector not in cls.detector_list):
                invalid_detector.append(detector)

        if(len(invalid_detector) > 0):   
            raise ValueError('detector supplied does not exist, implemented detector:', cls.detector_list, 'invalid values supplied:', invalid_detector)
        
        samples = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        detection_result = {detector: [] for detector in detectors}

        for step, batch in enumerate(tqdm(samples, disable=not verbose)):
            for detector, detector_func in cls.detector_model.items():
                if detector not in detectors:
                    continue

                detection_result[detector] += detector_func(batch)

        return detection_result


    @classmethod
    def calc_data_metrics(cls, queries: list[str], detectors: list[int] = None, metrics: list[int] = None, batch_size: int = None,
                        queries_result: dict[list] = None, validity_result: list[int] = None, is_sqli_result: list[int] = None,
                        verbose: bool = True) -> dict[list]:
        """ generate metrics from the given queries, based on our paper

        Args:
            queries (list[str]): queries for generating detection results.
            detectors (list[int], optional): list of detectors to be used. Defaults to all detectors implemented.
            metrics (list[int], optional): list of metrics to be used for metric calculation. Defaults to all metrics implemented.
            batch_size (int, optional): batch size for the detectors, useful for increasing speed, at the cost of higher mem usage. 
                Defaults to None, which uses the class variable batch_size instead.
            queries_result (dict[list], optional): the evasion result for the detectors to be used, should be a dict of detector, with a list of the 
                detection result. If the detectors provided does not have a corresponding result given here, it will be generated. 
                Defaults to {}, where all result according to the detector list provided will be generated instead.
            validity_result (list[int], optional): provide a list of validity result of the queries. 
                Defaults to None, which means all queries will be checked if they are valid SQL queries.
            is_sqli_result (list[int], optional): provide a list of whether each of the query provided is sqli. 
                Defaults to None, which means all queries will be checked if they are sqli.
            verbose (bool, optional): whether tqdm should show the progress bar, and if metrics should be printed out. Defaults to True.

        Returns:
            dict[list]: a dictionary of generated metrics, the keys are the name of the metrics, each of them are a list of 
            the generated metrics based on the queries given
        """

        if(queries_result is None):
            queries_result = {}

        if(detectors is None):
            detectors = cls.detector_list

        #check if any evasion detection result needs to be generated, if so, generate the result before computing the metrics
        detection_required = detectors
        for detector_result in queries_result:
            if(detector_result in detectors):
                detection_required.remove(detector_result)
        
        if(len(detection_required) > 0):
            temp_queries_result = cls.generate_detection_result(queries, detectors=detection_required, batch_size=batch_size, verbose=verbose)
            for detector, detector_results in temp_queries_result.items():
                queries_result[detector] = detector_results

        if(validity_result is None):
            validity_result = cls.compute_validity(queries)
        

        if(is_sqli_result is None):
            is_sqli_result = cls.check_sqli(queries, verbose=verbose)

        all_metrics_dict = cls.calc_complex_metric(validity_result, is_sqli_result, queries_result)

        if(verbose):
            cls.print_metrics(all_metrics_dict)

        return all_metrics_dict
    
    @staticmethod
    def print_metrics(queries_metrics: dict[list]):
        """print the metrics out in a readable form

        Args:
            queries_metrics (dict[list]): a dictionary of generated metrics, the keys are the name of the metrics, each of them are a list of 
            the generated metrics based on the queries given
        """
        for metric_name, metric_results in queries_metrics.items():
            metric_results = np.array(metric_results)
            metric_results = metric_results[~np.isnan(metric_results)]
            sample_count = len(metric_results)
            key, counts = np.unique(metric_results, return_counts=True)
            combined_counts = dict(zip(key, counts))
            if(1 not in combined_counts):
                combined_counts[1] = 0
            print('metric name: {0:<45}, total number: {1}, positive number: {2}, positive ratio: {3:.8g}'.format(metric_name, sample_count, combined_counts[1], combined_counts[1]/sample_count))


if __name__ == '__main__':
    dataset = pd.read_csv('dataset/second_dataset.csv')
    dataset = dataset.astype({'payload': 'U'})

    original_dataset_list = dataset['payload'].squeeze().to_list()



    generated_sqli = original_dataset_list

    metrics_result = metrics_generator.calc_data_metrics(generated_sqli)

    detector_model = {'cnn': CNNDetector.predict, 'lstm': LSTMDetector.predict, 'classical': metrics_generator.tester.classical_detector}
    detector_list = [key for key in detector_model] + ['ensemble_avg', 'ensemble_strict']


    generated_sqli_df = list(zip(original_dataset_list, generated_sqli, metrics_result['validity'], metrics_result['is_sqli']))
    generated_sqli_df = pd.DataFrame(generated_sqli_df, columns=['original_query', 'generated_query', 'validity', 'is_sql_injection'])
    for detector_name in detector_list:
        generated_sqli_df[detector_name + '_evaded_detection_any'] = pd.Series(metrics_result[detector_name + '_evaded_detection_any'])

    #metrics_generator.print_metrics(metrics_result)