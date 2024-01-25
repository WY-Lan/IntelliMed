from peft import get_peft_model, LoraConfig, TaskType
from peft import PrefixTuningConfig
from peft import PromptEncoderConfig
from peft import PromptTuningConfig
from peft import IA3Config



def add_peft_config(model, model_args):
    if model_args.fine_tune_type == "lora":
        # lora方式，这里暂时有3个参数
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="all"
        )
        model = get_peft_model(model, peft_config)

    elif model_args.fine_tune_type == "prefix_tuning":
        # 这个参数还要看看正确否
        peft_config = PrefixTuningConfig(
            task_type=TaskType.TOKEN_CLS,
            num_virtual_tokens=model_args.num_virtual_tokens,
            encoder_hidden_size=model_args.prefix_encoder_hidden_size,
            prefix_projection=model_args.prefix_projection
        )  
        model = get_peft_model(model, peft_config)
    
    elif model_args.fine_tune_type == "prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.TOKEN_CLS,
            num_virtual_tokens=model_args.num_virtual_tokens,
            prompt_tuning_init=model_args.prompt_tuning_init,
            prompt_tuning_init_text=model_args.prompt_tuning_init_text,
            tokenizer_name_or_path=model_args.tokenizer_name_or_path,
            tokenizer_kwargs=model_args.tokenizer_kwargs
        )
        model = get_peft_model(model, peft_config)

    elif model_args.fine_tune_type == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.TOKEN_CLS,
            num_virtual_tokens=model_args.num_virtual_tokens,
            encoder_reparameterization_type=model_args.encoder_reparameterization_type,
            encoder_hidden_size=model_args.p_tuning_encoder_hidden_size,
            encoder_num_layers=model_args.p_tuning_encoder_num_layers,
            encoder_dropout=model_args.p_tuning_encoder_dropout)
        model = get_peft_model(model, peft_config)

    elif model_args.fine_tune_type == "ia3":
        # 这里要检查peft_config的条件
        peft_config = IA3Config(
            peft_config = TaskType.TOKEN_CLS,
            target_modules = model_args.target_modules
        )
        model = get_peft_model(model, peft_config)
    return model