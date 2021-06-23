# Stance Detection for Reddit
Research project done for the NLP SS21 module  at Tsinghua University.

## Description


### Datasets
To download relevant stance detection datasets runt he dataset.sh file in tools. 
Requirements:
- unzip (terminal)

### Requirements
For the parser and text processing you will need Python 3.6+ and the following:
- spacy==3.0.6+
- pandas==1.2.4+
- en_core_web_sm (install with python -m spacy download en_core_web_sm)

### Benchmarks
Compute benchmark results on popular Stance Detection datasets using out-of-the-box classical NLP model using `run_trainer.py`.
Currently, the code supports the following models and datasets:

**Models**
- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased`
- `albert-base-v2`
- `xlnet-base-cased`

**Datasets**
- SemEval2016Task6
- ARC
- FNC-1
- SEthB
- SEthC

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
|  |  SemEval2016  |      ARC      |     SEthB     |     SEthC     |
|:-----------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|  Random Baseline  | .3363 / .2910 | .2413 / .1663 | .3355 / .3179 | .3127 / .2526 |
| Majority Baseline | .5833 / .2377 | .7607 / .3684 | .4581 / .2094 | .7523 / .2862 |
|        SVM        | .6765 / .5451 |  .8059 / **.4794**  | .5677 / .4532 | .7584 / .5385 |
|       ALBERT      | .6335 / .3419 | .7737 / .4035 | .4710 / .2135 | .7828 / .4576 |
|        BERT       | .7343 / .6366 | .7903 / .4331 | **.6065** / .4281 | .7907 / **.6139** |
|     DistilBERT    | .7004 / .5958 | .7935 / .4235 | .5806 / **.4436** | .7946 / .5932 |
|      RoBERTa      | **.7470** / **.6413** | .7920 / .4359 | .6000 / .4292 | **.8071** / .5809 |
|       XLNet       | .6887 / .5037 | **.8099** / .4780 | .5742 / .4371 | .8038 / .5942 |
|        SotA       |    - / .6979   |    - / .6583   | .7142 / .6251 |  .8071 / .6139 |

