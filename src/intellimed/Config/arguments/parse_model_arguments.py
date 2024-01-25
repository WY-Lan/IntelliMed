import os
import sys
from .get_arguments import ModelArguments, DataTrainingArguments
from transformers import TrainingArguments, HfArgumentParser

def parse_model_arguments():
    # 创建 ArgumentParser 对象
    trainargument = TrainingArguments
    trainargument.report_to = "none"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, trainargument))
    print(sys.argv)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args