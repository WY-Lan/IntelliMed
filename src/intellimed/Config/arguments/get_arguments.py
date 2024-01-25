from dataclasses import dataclass, field
from typing import Optional, Union, List, Any
from marshmallow.validate import OneOf


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # Model configuration
    model_name_or_path: str = field(
        default= None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # Model configuration name or path
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # Tokenizer configuration name or path
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # Cache directory path
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # Model version to use
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # Use authentication token
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    fine_tune_type: Optional[str] = field(
        default="standard",
        metadata={
            "help": "Choose from 'standard', 'lora', 'prefix_tuning', 'p_tuning', 'prompt_tuning', 'ia3'.",
        },
    )
    # 开始添加参数
    # lora
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "Lora attention dimension."}
    )

    lora_alpha: Optional[int] = field(
        default=8,
        metadata={"help": "Lora alpha."}
    )

    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    
    num_virtual_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Prefix encoder hidden size."}
    )

    # prefix_tuning
    prefix_encoder_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Prefix encoder hidden size."}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"}
    )

    # prompt_tuning
    prompt_tuning_init : str = field(
        default="RANDOM",
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    tokenizer_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The keyword arguments to pass to `AutoTokenizer.from_pretrained`. Only used if prompt_tuning_init is "
                "`TEXT`"
            ),
        },
    )

    # p_tuning 
    encoder_reparameterization_type: str = field(
        default="MLP",
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    p_tuning_encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    p_tuning_encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    p_tuning_encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )
    # ia3
    target_modules: Optional[Any] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ia3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default = None,
         metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    return_macro_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return macro metrics (i.e. final mean is taken across tag categories) during evaluation or just the overall ones (i.e. micro)."},
    )

