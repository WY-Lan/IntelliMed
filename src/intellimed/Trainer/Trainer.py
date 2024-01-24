from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union,Literal
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch.nn as nn
import transformers
from transformers import TrainingArguments
import os


class Trainer():
    """
        目前这里先设置这几个参数,如果是针对其它任务，可以增加对应的参数
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],
            args: TrainingArguments=None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            data_collator: Optional[DataCollator] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            fine_tunnine_type: Literal["standard", "lora", "prefix_tuning", "p_tuning", "prompt_tuning", "ia3"]="standard",
            data_args = None,
            logger = None,
            last_checkpoint = None
    ):
        # 先定义好这个模型的参数，
        self.trainer = transformers.Trainer(
            model = model,
            args = args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        self.fine_tunnine_type = fine_tunnine_type
        self.training_args = args
        self.data_args = data_args
        self.logger = logger
        self.last_checkpoint = last_checkpoint

    
    def train(self):
        assert self.trainer.train_dataset != None, "train_dataset is none but need train"

        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint

        if self.fine_tunnine_type == "standard":
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            self.trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics["train_samples"] = len(self.trainer.train_dataset)

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
        elif self.fine_tunnine_type == "lora":
            
            print("peft")
    
    def evaluate(self):
        assert self.trainer.eval_dataset != None, "eval_dataset is none but need train"
        self.logger.info("*** Evaluate ***")
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = len(self.trainer.eval_dataset)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
    
    def predict(self, predict_dataset):
        self.logger.info("*** Predict ***")

        results = self.trainer.predict(predict_dataset, metric_key_prefix="test")
        predictions = results.predictions
        metrics = results.metrics
        metrics["test_samples"] = len(predict_dataset)

        self.trainer.log_metrics("test", metrics)
        self.trainer.save_metrics("test", metrics)
        self.trainer.log(metrics)

        import json
        output_dir = self.training_args.output_dir
        word_ids = predict_dataset["word_ids"]
        json.dump({"predictions": results.predictions.tolist(), "label_ids": results.label_ids.tolist(), "word_ids": word_ids},
                      open(f"{output_dir}/test_outputs.json", "w"))

        if os.environ.get('USE_CODALAB', 0):
            import json
            json.dump(metrics, open("test_stats.json", "w"))