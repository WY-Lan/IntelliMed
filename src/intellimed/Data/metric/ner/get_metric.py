import numpy as np
from datasets import load_metric

metric = load_metric("seqeval")
def compute_metrics(p, label_list, data_args):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # This is just flattening the result dict
        # e.g. {'MISC': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 1}, 'PER': {'precision': 1.0, 'recall': 0.5, 'f1': 0.66, 'number': 2}, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # -> {'MISC_precision': 0.0, 'MISC_recall': 0.0, 'MISC_f1': 0.0, 'MISC_number': 1, 'PER_precision': 1.0, 'PER_recall': 0.5, 'PER_f1': 0.66, 'PER_number': 2, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    if data_args.return_macro_metrics:
        Ps, Rs, Fs = [], [], []
        for type_name in results:
            if type_name.startswith("overall"):
                continue
            print ('type_name', type_name)
            Ps.append(results[type_name]["precision"])
            Rs.append(results[type_name]["recall"])
            Fs.append(results[type_name]["f1"])
        return {
            "macro_precision": np.mean(Ps),
            "macro_recall": np.mean(Rs),
            "macro_f1": np.mean(Fs),
        }
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }