"""
Fine-tuning the library models for sequence to sequence.
"""
import os
import sys
import logging
from utils import set_seed, save_args, load_data, load_all_dataset, CTGDataCollator, create_optimizer, save_predict
set_seed(42)
from model.utils import build_model
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger(__name__)
# import datasets
# import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
# from datasets import load_dataset
# import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    BartForConditionalGeneration,
    BartConfig,
    BertTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from model.model_cpt import CPTModel, CPTForConditionalGeneration
MODEL_CLASSES = {
    "BART": (BartConfig,BartForConditionalGeneration,BertTokenizer),
    "T5":(AutoConfig,T5ForConditionalGeneration,AutoTokenizer),
    "CPT":(AutoConfig, CPTForConditionalGeneration, BertTokenizer)
}



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    cache_dirs: Optional[str] = field(
        default="./cache",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)
    # Save args
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    save_args((model_args, data_args, training_args))
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    train_set, val_set, test_set = load_data(data_args)

    tokenizer, config, model = build_model(MODEL_CLASSES['CPT'], model_args)
    train_dataset, val_dataset, test_dataset = load_all_dataset(
        train_set, val_set, test_set, tokenizer, data_args, data_args)
    data_collator = CTGDataCollator(tokenizer)
    optimizer = create_optimizer(training_args, model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
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
        return result
    data_collator = CTGDataCollator(tokenizer)
    optimizer = create_optimizer(training_args,model)
    # Initialize our trainer
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer,None)
    )


    max_length = data_args.val_max_target_length

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    #Training
    if training_args.do_train:
        logger.info("****** Evaluate before training ******")

        metrics = trainer.evaluate(max_length=max_length,num_beams=num_beams)
        trainer.log_metrics("eval_0", metrics)
        trainer.save_metrics("eval_0", metrics)

        logger.info("****** Training ******")

        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model(f"{training_args.output_dir}/best_model")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("****** Evaluate ******")
            
        metrics = trainer.evaluate(max_length=max_length,num_beams=num_beams)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("****** Predict ******")
        output = trainer.predict(test_dataset,max_length=max_length,num_beams=num_beams)
        trainer.log_metrics("predict", output.metrics)
        trainer.save_metrics("predict", output.metrics)
        # 保存文件代码
        save_predict(output,test_set, training_args.output_dir, tokenizer)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()