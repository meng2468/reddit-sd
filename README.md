# Stance Detection for Reddit
This project started as a course project for the 2021 Natural Language Processing course at Tsinghua University
The final project report is available [here](Final_Report.pdf) and a project poster is available [here](Project_Poster.pdf)

## Introduction
Stance Detection is an important area of research in Natural Language Processing with growing interest from the community. 
By leveraging large amounts of publicly available data from social media, it can aid in tasks ranging from fake news detection to product decision making and review. 
Although similar to aspect-based Sentiment Analysis, both the amount of annotated datasets and the application of cutting-edge Machine Learning techniques are still lacking. 
This project contributes to Stance Detection by addressing these main two challenges over the case of Ethereum.

First we create our own datasets SEthB and SEthC which, in contrast to the majority of datasets in Std (and more generally Sentiment Analysis) which are created off data from Twitter, utilizes comments taken from the popular discusison forum Reddit. Our motivation of doing so derives from Reddit being the most popular area of discussion for Cryptocurrencies and being able to provide guidelines and an initial methodology for creating a Stance Detection dataset for Reddit. Furthermore, our approach to dataset creation promotes a recently explored annotation technique based on clustering (Zotova et al., 2021) which possesses significant potential and requires further exploration in future work.

Second, due to the recent performance of large language models on different domains in Natural Language Processing, the application of Transfer Learning has seen many initial successes in Sentiment Analysis. Therefore we decide to take a step forward in this direction and introduce a new comparative study of multiple PLMs of interest in the particular case StD. By transferring knowledge between models with different domains, we approach state-of-the-art results on popular StD datasets with only but minimal human intervention in data preprocessing, model fine-tuning and task description.

## Project Details
### Requirements
To get the project running you will need Python 3.6+ and do the following
- Download popular stance detection datasets using: `sh tools/datasets.sh`
- Install the relevant packages with: `pip install -r requirements.txt`
- Install the language model used in spaCy: `python -m spacy download en_core_web_sm`

### Data Retrieval and Processing
To perform both data retrieval and filtering as described in the report, sample code is provided in `example.py`.
Data retrieval from reddit is done using the Pushshift API in `tools/parser.py`.
Our filtering and preliminary pre-processing is implemented in `tools/processing.py`.
A simple annotation script is provided in `tools/annotate.py`, and can easily be edited to annotate newly-retrieved datasets.

### Transfer Learning with Pre-Trained Language Models
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

