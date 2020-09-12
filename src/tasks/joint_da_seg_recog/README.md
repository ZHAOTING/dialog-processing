# Joint dialogue act segmentation and recognition

This project reimplements the CSL 2019 journal paper [*Joint dialog act segmentation and recognition in human conversations using attention to dialog context*, Zhao and Kawahara, 2020](https://www.sciencedirect.com/science/article/abs/pii/S0885230818304030).

## Reproduction issue
This codebase is a reimplementation of the work and cannot reproduce the reported results in the CSL paper because of the following reasons:

* The original codebase is outdated and does not fit in the current project framework.

* A bug has been found in the original metric implementation by Piotr Zelasko @ JHU, and the actual performance should be worse.

* This new codebase differs from the original codebase in a few aspects: 1) correct metric computation, 2) a new data split that follows a commmonly used SwDA split, 3) improved model implementation that makes use of submodules in this repo, and 4) a few improved hyperparameters.

* Only the ED and Attn-ED models have been implemented here as they are the most important ones.

* Although I have not done extensive experiments using this codebase (because I do not have access to computation resource for the moment), some preliminary experiments showed that the reimplemented models can still achieve reasonable results (better segmentation error rates and worse but not that bad joint error rates in comparison to numbers reported in the CSL paper). 

### Prepraing data
* Build dataset.
~~~
python -m corpora.swda.build_joint_da_seg_recog_dataset
~~~

* Collect word embeddings for initialization from pretrained embeddings (e.g. Glove Twitter 200D word embedding).
~~~
python -m corpora.swda.get_pretrained_embedding -e glove -t joint_da_seg_recog -p {path to glove.twitter.27B.200d.txt} -o ../data/swda/joint_da_seg_recog/glove_twitter_200.json
~~~

### Training/Evaluating
* ED model
~~~
CUDA_VISIBLE_DEVICES=0 python -m tasks.joint_da_seg_recog.train --history_len 1 --model ed
~~~

* Attn-ED model
~~~
CUDA_VISIBLE_DEVICES=0 python -m tasks.joint_da_seg_recog.train --history_len 3 --model attn_ed
~~~