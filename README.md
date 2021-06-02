# Stance Detection for Reddit
## Description
## Datasets
To download relevant stance detection datasets runt he dataset.sh file in tools. 
Requirements:
- unzip (terminal)
## Requirements
For the parser and text processing you will need Python 3.6+ and the following:
- spacy==3.0.6+
- pandas==1.2.4+
- en_core_web_sm (install with python -m spacy download en_core_web_sm)

## Benchmarks
Compute benchmark results on popular Stance Detection datasets using out-of-the-box classical NLP model using `run_trainer.py`.
Currently, the code supports the following models and datasets:

**Models**
- bert-base-uncased

**Datasets**
- SemEval2016Task6

Here is how to run the script on one of them.
```
export MODEL=bert-base-uncased
export DATASET=SemEval2016Task6

python run_trainer.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir results/$DATASET/$MODEL \
    --cache_dir cache/$DATASET/$MODEL \
    --do_eval \
    --do_train
```

We get the following results on the test set of each dataset using a Nvidia P100 GPU.
Scores are formatted as *Accuracy / F1* and are computed as an average over all targets.
|            |  SemEval2016Task6 | ARC | FNC-1 | IAC | PERSPECTRUM |
|------------|:-----------------:|:---:|:-----:|:---:|:-----------:|
|    BERT    | 0.73434 / 0.63666 |     |       |     |             |
|   RoBERTa  |                   |     |       |     |             |
| DistilBERT |                   |     |       |     |             |
|   ALBERT   |                   |     |       |     |             |
|    XLNet   |                   |     |       |     |             |

### Usage
```
usage: run_trainer.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
                      [--config_name CONFIG_NAME]
                      [--tokenizer_name TOKENIZER_NAME]
                      [--cache_dir CACHE_DIR] [--no_use_fast_tokenizer]
                      [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
                      [--model_revision MODEL_REVISION]
                      [--use_auth_token [USE_AUTH_TOKEN]]
                      [--dataset_name DATASET_NAME]
                      [--dataset_config_name DATASET_CONFIG_NAME]
                      [--max_seq_length MAX_SEQ_LENGTH]
                      [--overwrite_cache [OVERWRITE_CACHE]]
                      [--pad_to_max_length [PAD_TO_MAX_LENGTH]]
                      [--max_train_samples MAX_TRAIN_SAMPLES]
                      [--max_eval_samples MAX_EVAL_SAMPLES]
                      [--max_predict_samples MAX_PREDICT_SAMPLES]
                      [--train_file TRAIN_FILE]
                      [--validation_file VALIDATION_FILE]
                      [--test_file TEST_FILE] --output_dir OUTPUT_DIR
                      [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
                      [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
                      [--do_predict [DO_PREDICT]]
                      [--evaluation_strategy {no,steps,epoch}]
                      [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
                      [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                      [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                      [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                      [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                      [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                      [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                      [--learning_rate LEARNING_RATE]
                      [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                      [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON]
                      [--max_grad_norm MAX_GRAD_NORM]
                      [--num_train_epochs NUM_TRAIN_EPOCHS]
                      [--max_steps MAX_STEPS]
                      [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                      [--warmup_ratio WARMUP_RATIO]
                      [--warmup_steps WARMUP_STEPS]
                      [--logging_dir LOGGING_DIR]
                      [--logging_strategy {no,steps,epoch}]
                      [--logging_first_step [LOGGING_FIRST_STEP]]
                      [--logging_steps LOGGING_STEPS]
                      [--save_strategy {no,steps,epoch}]
                      [--save_steps SAVE_STEPS]
                      [--save_total_limit SAVE_TOTAL_LIMIT]
                      [--no_cuda [NO_CUDA]] [--seed SEED] [--fp16 [FP16]]
                      [--fp16_opt_level FP16_OPT_LEVEL]
                      [--fp16_backend {auto,amp,apex}]
                      [--fp16_full_eval [FP16_FULL_EVAL]]
                      [--local_rank LOCAL_RANK]
                      [--tpu_num_cores TPU_NUM_CORES]
                      [--tpu_metrics_debug [TPU_METRICS_DEBUG]]
                      [--debug DEBUG]
                      [--dataloader_drop_last [DATALOADER_DROP_LAST]]
                      [--eval_steps EVAL_STEPS]
                      [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                      [--past_index PAST_INDEX] [--run_name RUN_NAME]
                      [--disable_tqdm DISABLE_TQDM]
                      [--no_remove_unused_columns]
                      [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
                      [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                      [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                      [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                      [--greater_is_better GREATER_IS_BETTER]
                      [--ignore_data_skip [IGNORE_DATA_SKIP]]
                      [--sharded_ddp SHARDED_DDP] [--deepspeed DEEPSPEED]
                      [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                      [--adafactor [ADAFACTOR]]
                      [--group_by_length [GROUP_BY_LENGTH]]
                      [--length_column_name LENGTH_COLUMN_NAME]
                      [--report_to REPORT_TO [REPORT_TO ...]]
                      [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                      [--no_dataloader_pin_memory]
                      [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
                      [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
                      [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
                      [--push_to_hub [PUSH_TO_HUB]]
                      [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                      [--mp_parameters MP_PARAMETERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name
  --cache_dir CACHE_DIR
                        Where do you want to store the pretrained models
                        downloaded from huggingface.co
  --no_use_fast_tokenizer
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not.
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by
                        the tokenizers library) or not.
  --model_revision MODEL_REVISION
                        The specific model version to use (can be a branch
                        name, tag name or commit id).
  --use_auth_token [USE_AUTH_TOKEN]
                        Will use the token generated when running
                        `transformers-cli login` (necessary to use this script
                        with private models).
  --dataset_name DATASET_NAME
                        The name of the dataset to use: SemEval2016Task6
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the
                        datasets library).
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --overwrite_cache [OVERWRITE_CACHE]
                        Overwrite the cached preprocessed datasets or not.
  --pad_to_max_length [PAD_TO_MAX_LENGTH]
                        Whether to pad all samples to `max_seq_length`. If
                        False, will pad the samples dynamically when batching
                        to the maximum length in the batch.
  --max_train_samples MAX_TRAIN_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of training examples to this value if set.
  --max_eval_samples MAX_EVAL_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of evaluation examples to this value if
                        set.
  --max_predict_samples MAX_PREDICT_SAMPLES
                        For debugging purposes or quicker training, truncate
                        the number of prediction examples to this value if
                        set.
  --train_file TRAIN_FILE
                        A csv or a json file containing the training data.
  --validation_file VALIDATION_FILE
                        A csv or a json file containing the validation data.
  --test_file TEST_FILE
                        A csv or a json file containing the test data.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory.
  --do_train [DO_TRAIN]
                        Whether to run training.
  --do_eval [DO_EVAL]   Whether to run eval on the dev set.
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set.
  --evaluation_strategy {no,steps,epoch}
                        The evaluation strategy to use.
  --prediction_loss_only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only
                        returns the loss.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred.Batch size per GPU/TPU core/CPU for
                        evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU.
  --learning_rate LEARNING_RATE
                        The initial learning rate for AdamW.
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some.
  --adam_beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer
  --adam_beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer
  --adam_epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.
  --warmup_ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_dir LOGGING_DIR
                        Tensorboard log dir.
  --logging_strategy {no,steps,epoch}
                        The logging strategy to use.
  --logging_first_step [LOGGING_FIRST_STEP]
                        Log the first global_step
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_strategy {no,steps,epoch}
                        The checkpoint save strategy to use.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints
  --no_cuda [NO_CUDA]   Do not use CUDA even when it is available
  --seed SEED           Random seed that will be set at the beginning of
                        training.
  --fp16 [FP16]         Whether to use 16-bit (mixed) precision instead of
                        32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --fp16_backend {auto,amp,apex}
                        The backend to be used for mixed precision.
  --fp16_full_eval [FP16_FULL_EVAL]
                        Whether to use full 16-bit precision evaluation
                        instead of 32-bit
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script)
  --tpu_metrics_debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is
                        preferred. TPU: Whether to print debug metrics
  --debug DEBUG         Whether or not to enable debug mode. Current options:
                        `underflow_overflow` (Detect underflow and overflow in
                        activations and weights), `tpu_metrics_debug` (print
                        debug metrics on TPU).
  --dataloader_drop_last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible
                        by the batch size.
  --eval_steps EVAL_STEPS
                        Run an evaluation every X steps.
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process.
  --past_index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step.
  --run_name RUN_NAME   An optional descriptor for the run. Notably used for
                        wandb logging.
  --disable_tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
  --no_remove_unused_columns
                        Remove columns not required by the model when using an
                        nlp.Dataset.
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset.
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels.
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training.
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not.
  --ignore_data_skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data.
  --sharded_ddp SHARDED_DDP
                        Whether or not to use sharded DDP training (in
                        distributed training only). The base option should be
                        `simple`, `zero_dp_2` or `zero_dp_3` and you can add
                        CPU-offload to `zero_dp_2` or `zero_dp_3` like this:
                        zero_dp_2 offload` or `zero_dp_3 offload`. You can add
                        auto-wrap to `zero_dp_2` or with the same syntax:
                        zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. ds_config.json) or an already loaded
                        json file as a dict
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing).
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
  --group_by_length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching.
  --length_column_name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length.
  --report_to REPORT_TO [REPORT_TO ...]
                        The list of integrations to report the results and
                        logs to.
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`.
  --no_dataloader_pin_memory
                        Whether or not to pin memory for DataLoader.
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader.
  --skip_memory_metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics.
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer.
  --push_to_hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the
                        model hub after training.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your
                        model.
  --mp_parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer
```