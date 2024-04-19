import pathlib
import torch
from lstm_interface import LSTMInterface


class LSTMDetector:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur_file_location = str(pathlib.Path(__file__).parent.resolve())

    model_path = cur_file_location + '/model.pt'
    word2idx_path = cur_file_location + '/word2idx.json'

    wafInterface = None

    @classmethod
    def init_interface(cls):
        if(cls.wafInterface is None):
            cls.wafInterface = LSTMInterface(cls.device, cls.model_path, cls.word2idx_path)

    @classmethod
    def __predict_each(cls, payload):
        cls.init_interface()
        res = cls.wafInterface.get_score(payload=payload)
        if(res > 1 or res < 0):
            raise Exception(payload, res)
        return 1 if res > 0.5 else 0
    
    @classmethod
    def predict(cls, payloads):
        res_list = []
        for payload in payloads:
            res_list.append(cls.__predict_each(payload))
        return res_list