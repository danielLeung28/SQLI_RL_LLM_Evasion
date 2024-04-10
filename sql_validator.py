import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import sqlvalidator
import signal, time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class checkSQLValidity:

    __sqli_format = 'SELECT * FROM test where username = {} and pw = \'test\''
    __sql_format_req = ['\'', '\"', 'select', 'insert', 'delete', 'create', 'drop', 'alter']
    batch_size = 16
    timeout_time = 1

    #sql validator may get stuck sometimes, so a time limit is imposed, if it get stuck, it means that the query is invalid
    @staticmethod
    def timeout_handler(signum, frame):
        raise Exception

    signal.signal(signal.SIGALRM, timeout_handler)

    #some generated sql may just be a word, or other mix of random characters and phrase, so check that it contain elements
    #of a sql is required, the keywords along with quote characters are used in the checking
    def __check_validation_second_stage(self, query):
        query = query.lower()
        base_format_validity = False

        for req in self.__sql_format_req:
            if(req in query):
                base_format_validity = True
                break
            
        if(base_format_validity):
            query_res = sqlvalidator.parse(self.__sqli_format.format(query))
            
            if(query_res.is_valid()):
                return True

        return False
    
    def check_single_SQL_validity(self, query):
        test_query = sqlvalidator.parse(query) #test normal query
        validity = False
        try:
            if(test_query.is_valid()):
                validity = True
                #return True
            else:#if the sql isn't of normal type, i.e. starts with select etc, then it might contain only the part for sqli,
                    #such as ' OR 1=1; drop database test;, as such, this portion insert the query into a test sql prepared
                    #for sqli, and then use the validator to see if it is valid or not
                if(self.__check_validation_second_stage(query)): 
                    validity = True
                    #return True

        except Exception as e:
            pass

        return validity


    def check_SQL_validity(self, queries):
        valid_count = 0
        validated_query = {'Query' : [], 'result': []}

        for query in queries:
            signal.setitimer(signal.ITIMER_REAL, checkSQLValidity.timeout_time)
            #signal.alarm(checkSQLValidity.timeout_time)

            validated_query['Query'].append(query)
            try:
                validity = self.check_single_SQL_validity(query)
                validated_query['result'].append(1 if validity else 0)
                valid_count += 1 if validity else 0
            except Exception: #if sqlvalidator get stuck, then the query is invalid, and set the result as invalid
                validated_query['result'].append(0)

        # #make sure that the timeout handler is finished before returning to prevent exception being raised outside the block
        # try:
        #     time.sleep(checkSQLValidity.timeout_time)
        # except Exception:
        #     pass

        #disable the alarm to prevent exception being raised outside the block
        signal.setitimer(signal.ITIMER_REAL, 0)
        #signal.alarm(0)
        return valid_count, validated_query
    

if(__name__ == '__main__'):
    #dataset = pd.read_csv('Modified_SQL_Dataset.csv')
    dataset = pd.read_csv('testing_generated_output.csv', index_col=0)
    #sqli_dataset_org = load_dataset('csv', data_files='Modified_SQL_Dataset.csv')
    SQL_validator = checkSQLValidity()
    print(dataset)
    valid_count, validated_query = SQL_validator.check_SQL_validity(dataset['0'].squeeze().to_list())
    print(valid_count, len(validated_query['Query']))
    print(validated_query['Query'][4886:4889])

    query_validity_df = pd.DataFrame(validated_query)
    print(query_validity_df)
    query_validity_df.to_csv('query_validity_signal.csv', index=False)
    #valid_count, validated_query = SQL_validator.check_SQL_validity(sqli_dataset_org['train'][:]['Query'])
    #print(valid_count, len(sqli_dataset_org['train']))

    #validated_query_df = pd.DataFrame.from_dict(validated_query)
    #print(validated_query_df)