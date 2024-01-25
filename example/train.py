# 首先是要解析参数
import os
from transformers import set_seed
from intellimed.Config.load_checkpoint import load_checkpoint
from intellimed.Config.logging import set_logger
from intellimed.Config.arguments import parse_model_arguments
from intellimed.Data.dataset.ner import process
from intellimed.Trainer import Trainer
from datasets import load_metric
from functools import partial
from intellimed.Data.metric.ner import get_metric


if __name__ == "__main__":
    # 后去参数
    model_args, data_args, training_args = parse_model_arguments()
    logger = set_logger(training_args)
    # Detecting last checkpoint
    logger, last_checkpoint = load_checkpoint(logger, training_args)
    set_seed(training_args.seed)
    model, train_dataset, eval_dataset, predict_dataset, tokenizer, data_collator, label_list = process(training_args, data_args, model_args)
    compute_metrics = partial(get_metric.compute_metrics, label_list=label_list, data_args = data_args)

    # initial our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        fine_tunnine_type = model_args.fine_tune_type,
        data_args = data_args,
        logger=logger,
        last_checkpoint=last_checkpoint
    )

    trainer.train()
    trainer.evaluate()
    trainer.predict(predict_dataset)