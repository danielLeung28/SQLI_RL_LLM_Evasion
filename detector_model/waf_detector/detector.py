import requests
import multiprocessing

class WAFDetector:

    @classmethod
    def __predict_each(cls, id, base_url, payload, return_dict):
        res = requests.get(base_url + payload).json()
        #print(id, payload, res)
        if('message' in res):
            return_dict[id] = 1
        else:
            return_dict[id] = 0
    
    @classmethod
    def predict(cls, base_url, payloads):
        res_list = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(len(payloads)):
            processes.append(multiprocessing.Process(target=cls.__predict_each, args=(i, base_url, payloads[i], return_dict)))
            processes[-1].start()

        for process in processes:
            process.join()
            
        #print(return_dict)
        for i in range(len(payloads)):
            res_list.append(return_dict[i])

        #print(res_list)

        return res_list