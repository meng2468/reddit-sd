#!/usr/bin/env python
# coding=utf-8
""" 
Finetuning the library models for stance detection
--------------------------------------------------
    Original code by HuggingFace 
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
"""

import os
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric, list_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

# custom imports
from tools import dataloader
# from models.models import SimpleStDClassifier


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.7.0.dev0")

available_models = [ "bert-base-uncased" ]
available_datasets = [ "SemEval2016Task6" ]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use: "+ ' | '.join(available_datasets)}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is not None:
            if self.dataset_name in available_datasets or self.dataset_name in list_datasets():
                pass
            else:
                raise ValueError("Need a valid dataset name from datasets hub or one of: ({})".format(' | '.join(available_datasets)))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # ===== Get the datasets =====
    print('{:=^50}'.format(" Initialise "))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Log on each process the small summary:
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    print(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ===== Get the datasets =====
    print('{:=^50}'.format(" Load dataset "))
    if data_args.dataset_name is not None:
        if data_args.dataset_name in list_datasets():
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        else:
            # Downloading and loading a custom dataset
            datasets = dataloader.load_custom_dataset(data_args, cache_dir=model_args.cache_dir) # TODO

    else:
        # Loading a dataset from local files
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need a test file for `do_predict`.")

        for key in data_files.keys():
            print(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # Targets
    if "target" in datasets["train"].features.keys():
        target_list = datasets["train"].unique("target")
        target_list.sort()  # Let's sort it for determinism
        num_targets = len(target_list)
    else:
        target_list = ["default"]

    # Labels
    label_list = datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # ===== Preprocessing the datasets =====
    print('{:=^50}'.format(" Preprocess "))
    config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples['text'], )
        )
        result = tokenizer(
            *args,
            padding="max_length" if data_args.pad_to_max_length else False, # False=pad later, dynamically at batch creation, to the max sequence length in each batch
            max_length=max_seq_length, 
            truncation=True)

        # Map labels to IDs
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    print(f"Found {len(target_list)} targets.")

    # ===== BIG ASS LOOP THROUGH ALL TARGETS =====
    for target in target_list:
        formatted_target = target.strip().lower().replace('\s', '-')
        print('{:*^50}'.format(' ' + target + ' '))
        print('('+ formatted_target +')')

        # Filter & Split datasets
        targeted_dataset = datasets.filter(lambda example: example['target'] == target)
        if training_args.do_train:
            assert 'train' in targeted_dataset, "--do_train requires a train dataset"
            train_dataset = targeted_dataset['train']
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            assert 'validation' in targeted_dataset, "--do_eval requires a validation dataset"
            eval_dataset = targeted_dataset['validation']
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.test_file is not None:
            assert 'test' in datasets, "--do_predict requires a test dataset"
            predict_dataset = targeted_dataset['test']
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        # Log a few random samples from the training set:
        if training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                print(f"Sample {index} of the training set: {train_dataset[index]}.")

        # ===== Load pretrained model =====
        print('{:=^50}'.format(" Load pretrained model "))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=Path(model_args.cache_dir)/Path(formatted_target),
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # model = SimpleStDClassifier(base_model, num_labels, base_model_output_size=config.hidden_size)

        # ===== Initialize Trainer =====
        print('{:=^50}'.format(" Initialise Trainer "))
        
        # get the metric function
        metric_names = "accuracy f1 recall precision".split()
        metrics = [load_metric(m) for m in metric_names]

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            results = {
                k: metric.compute(predictions=preds, references=p.label_ids) for k, metric in metrics.items()
            }
            for k, result in results.items():
                if len(result) > 1:
                    results[f"combined {k}"] = np.mean(list(result.values())).item()
            return results

        # data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # ===== Training =====
        if training_args.do_train:
            print('{:=^50}'.format(" Train "))

            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics['train_samples'] = min(max_train_samples, len(train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics('train', metrics)
            trainer.save_metrics('train', metrics)
            trainer.save_state()

        # ===== Evaluation =====
        if training_args.do_eval:
            print('{:=^50}'.format(" Evaluate "))
        
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics('eval', metrics)
            trainer.save_metrics('eval', metrics)

        # ===== Prediction =====
        if training_args.do_predict:
            print('{:=^50}'.format(" Predict "))
            
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset.remove_columns_('label')
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{formatted_target}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    print('{:*^50}'.format(" Predict results '" + target + "' "))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
    
    # end bigass FOR loop


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()