import pandas as pd
import sys


"""
Preprocess the simplerl dataset to parquet format
"""

import os
import datasets
import argparse
import  pandas as pd
data_source = 'polaris_tinyv'

system_prompt = """
Please reason step by step, and put your final answer within \\boxed{}.\n\n
"""
if __name__ == '__main__':


    train_dataset = datasets.load_dataset("tmp")["train"]

    def process_fn_train(example, idx):
        # messages =[{"role": "user",
        #                         "content": 'Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{}'.format(example['problem'])}]
        print(example)
        prompt = [
            {'content': system_prompt,
             'role': 'system'},
                  {'content': example.data['problem'] , 'role': 'user'}
                  ]

        data = {
            "data_source": data_source,
            "prompt": prompt,
            'criteria': example.data['criteria'],
            'difficulty': example.data['difficulty'],
            'query_id': example.data['query_id'],
            'answer': example.data['answer'],
            'problem': example.data['problem'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example.data['answer'],
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example.data['answer'],
                "question": example.data['problem'],
                'criteria': example.data['criteria'],
            }
        }
        return data


    train_dataset = train_dataset.map(function=process_fn_train, with_indices=True)
    train_dataset.to_parquet('train_polaris.parquet')
