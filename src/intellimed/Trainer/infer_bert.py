from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def infer_bert(text ,model_name_or_path):
    # 指定预训练模型的路径或名称
    model_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # 输入文本，使用tokenizer编码为token IDs
    assert type(text) == str, "the infer text is not a str"
    inputs = tokenizer(text, return_tensors="pt")

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测的标签
    predicted_labels = torch.argmax(outputs.logits, dim=2)

    # 将预测的标签转换为标签名称
    predicted_labels_names = [model.config.id2label[label_id] for label_id in predicted_labels[0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
    return tokens, predicted_labels_names
