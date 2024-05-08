import torch
import pathlib
from interface import CNNInterface


class CNNDetector:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur_file_location = str(pathlib.Path(__file__).parent.resolve())

    model_path = cur_file_location + '/model.pt'
    word2idx_path = cur_file_location + '/word2idx.json'

    wafInterface = None

    @classmethod
    def init_interface(cls):
        if(cls.wafInterface is None):
            cls.wafInterface = CNNInterface(cls.device, cls.model_path, cls.word2idx_path)

    @classmethod
    def __predict_each(cls, payload, get_raw_prob):
        cls.init_interface()
        res = cls.wafInterface.get_score(payload=payload)
        if(get_raw_prob):
            return res
        return 1 if res > 0.5 else 0
    
    @classmethod
    def predict(cls, payloads, get_raw_prob = False):
        res_list = []
        for payload in payloads:
            res_list.append(cls.__predict_each(payload, get_raw_prob))
        return res_list