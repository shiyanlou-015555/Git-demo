# ACH 2022/11/12
import random
import torch
import numpy as np
import os
import json
from dataclasses import dataclass, field, fields
from typing import Dict, List, Any, Union, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def set_seed(seed):
    """
    set seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def save_args(args):
    """
    save parameters
    """
    all_args = {}
    for dtype in args:
        keys = {f.name for f in fields(dtype) if f.init}
        for k in keys:
            all_args[k] = getattr(dtype, k)
    output_file = os.path.join(all_args["output_dir"], "parameters.txt")

    with open(output_file, "w", encoding="utf-8") as fout:
        for k, v in all_args.items():
            fout.write(f"{k}={v}\n")

def read_file(file):
    """
    read data for out models
    """
    cnt = 0
    data = {"queries": [], "sents": []}
    # keywords = [item for item in item["keywords"] if is_contains_chinese(item)]
    with open(file) as fin:
        for line in fin:
            item = json.loads(line)
            if len(item["sentence1"]) + len(item["sentence2"]) > 300:
                cnt += 1
                continue
            # keywords = [item for item in item["keywords"] if is_contains_chinese(item)]
            # if len(keywords)<1:
            #     cnt += 1
            #     continue
            data["queries"].append(item["sentence1"])
            data["sents"].append(item["sentence2"])
            # keywords = [item for item in item["keywords"] if is_contains_chinese(item)]
            # assert len(keywords)>=1
            # data["keywords"].append(keywords)
    if cnt > 0:
        print(f"有{cnt}个样本不符合格式要求")

    return data

def is_contains_chinese(strs):
    """
    只对中文进行替换
    """
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def load_data(data_args):
    """
    load datas
    """
    train_set = read_file(data_args.train_file)
    val_set = read_file(data_args.validation_file)
    test_set = read_file(data_args.test_file)

    return train_set, val_set, test_set


def load_all_dataset(
    train_set: Dict,
    val_set: Dict,
    test_set: Dict,
    tokenizer: PreTrainedTokenizerBase,
    model_args: Any,
    data_args: Any
):
    train_val_cache = os.path.join(data_args.cache_dirs, "train_val_encoding.cache")
    if not model_args.overwrite_cache and os.path.exists(train_val_cache):
        print("this code is error")
    else:
            train_q = train_set["queries"]
            train_s = train_set["sents"]
            # train_keywords = train_set["keywords"]
            val_q = val_set["queries"]
            val_s= val_set["sents"]
            # val_keywords = val_set["keywords"]

    test_q = test_set["queries"]
    test_s = test_set["sents"]
    # test_keywords = test_set["keywords"]
    train_dataset = CTGDataset(train_q, train_s, tokenizer)
    val_dataset = CTGDataset(val_q, val_s, tokenizer)
    test_dataset = CTGDataset(test_q, test_s, tokenizer)

    return train_dataset, val_dataset, test_dataset


class CTGDataset(Dataset):
    def __init__(self, q, s, tokenizer):
        super().__init__()
        self.encodings = {}
        self.encodings["query"] = q
        self.encodings["sent"] = s
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        temp = {}
        sentence = item["query"]
        # # label = item["query"] +"[PAD]" + item["sent"]
        # label = keyword
        label = item["sent"]
        inputs = self.tokenizer(sentence)
        outputs = self.tokenizer(label)
        for k,v in inputs.items():
            temp[k] = v
        temp["labels"] = outputs["input_ids"]
        return temp

    def __len__(self):
        return len(self.encodings['query'])


@dataclass
class CTGDataCollator:
    """
    datacollector
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features):
        max_length = max(len(inputs["input_ids"]) for inputs in features)
        max_length_label = max(len(inputs["labels"]) for inputs in features)
        batch_outputs = {}
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys()}
        for i in range(len(features)):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(inputs, max_length=max_length,max_length_label = max_length_label)

            for k, v in outputs.items():
                if k not in batch_outputs:
                    batch_outputs[k] = []
                batch_outputs[k].append(v)
        if "token_type_ids" in batch_outputs.keys():
            batch_outputs.pop("token_type_ids")
        # return {torch.tensor(batch_outputs[key]) for key in batch_outputs.keys()}
        return BatchEncoding(batch_outputs, tensor_type="pt")

    def _pad(self, inputs, max_length, max_length_label):
        difference = max_length - len(inputs["input_ids"])
        inputs["input_ids"] = inputs["input_ids"] + [self.tokenizer.pad_token_id] * difference
        inputs["attention_mask"] = inputs["attention_mask"] + [0] * difference
        if "token_type_ids" in inputs.keys():
            inputs.pop("token_type_ids")
        difference_label = max_length_label-len(inputs["labels"])
        inputs["labels"] = inputs["labels"] + [self.tokenizer.pad_token_id] * difference_label
        return inputs



from transformers.optimization import AdamW


def create_optimizer(args, model):
    """
    optimizer
    """
    no_decay = ["bias", "LayerNorm.weight"]
    model_param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def compute_metrics(eval_pred, tokenizer):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        
        # 字符级别
        decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in decoded_preds]
        decoded_labels = [" ".join((label.replace(" ", ""))) for label in decoded_labels]
        # 词级别，分词
        # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
        # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
        rouge = Rouge()
        labels_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in labels]
    
    
        total = 0
    
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
            total += 1
            scores = rouge.get_scores(hyps=decoded_pred, refs=decoded_label)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[decoded_label.split(' ')],
                hypothesis=decoded_pred.split(' '),
                weights=(1,0)
            )
        bleu /= len(decoded_labels)
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        result = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l}
        print(result)
        # 测试平均与分别计算是否一致
        result2 = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        print(result2)
        print(bleu)
        # result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
    
        result = {key: value * 100 for key, value in result.items()}
        result["gen_len"] = np.mean(labels_lens)
        result["bleu"] = bleu * 100
        return result, decoded_preds , decoded_labels

def save_predict(output,test_set, out_dir, model_tokenizer):
    """save output to file

    Args:
        output (List): [preidction, labels]
        out_dir (str): output directory
    """
    output_file = os.path.join(out_dir, "prediction_result.csv")
    metric,predictions,labels = compute_metrics(output, model_tokenizer)
    # Replace -100 in the labels as we can't decode them.
    print("--------------label---------------")
    print(metric)

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(f"预测关键词|Masked掉的关键词|query|sents\n")
        for p, l,query,sent in zip(predictions, labels,test_set["queries"],test_set["sents"]):
            fout.write(f"{p}|{l}|{query}|{sent}\n")
