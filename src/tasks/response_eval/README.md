# Dialogue response evaluation

This project implements the ACL 2020 paper [*Designing Precise and Robust Dialogue Response Evaluators*, Zhao et al., 2020](https://arxiv.org/abs/2004.04908).

Human annotations used in the paper can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt?usp=sharing) or [Zenodo](https://zenodo.org/record/3828180).

## Reproduction
Here we use an example to illustrate how to reproduce the main results in the paper.

### Prepraing unsupervised data
* Build response generation dataset if you haven't yet.
    ~~~
    python -m corpora.dd.build_response_gen_dataset
    ~~~

### Preparing supervised data
* Download Amazon MTurk annotations on the DailyDialog corpus from [Google Drive](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt?usp=sharing) or [Zenodo](https://zenodo.org/record/3828180).

* Detect and mark outliers.
    ~~~
    python -m tasks.amt.mark_outliers --amt_result_path {path_to_the_DailyDialog_file}
    ~~~

* Analyze annotations.
    * To reproduce Figure 1 (a)
        ~~~
        python -m tasks.amt.analysis --result_path {path_to_the_DailyDialog_file} --rm_outlier True --plot True --plot_type score_dist --target_score overall --plot_output_path {path_to_1a_eps_file}
        ~~~
    * To reproduce Figure 1 (b)
        ~~~
        python -m tasks.amt.analysis --result_path {path_to_the_DailyDialog_file} --rm_outlier True --plot True --plot_type sys_box --plot_output_path {path_to_1b_eps_file}
        ~~~
    * To calculate Krippendorff's alpha before removing outliers
        ~~~
        python -m tasks.amt.analysis --result_path {path_to_the_DailyDialog_file} --rm_outlier False --agreement True
        ~~~
    * To calculate Krippendorff's alpha after removing outliers
        ~~~
        python -m tasks.amt.analysis --result_path {path_to_the_DailyDialog_file} --rm_outlier True --agreement True
        ~~~

* Build response evaluation dataset.
    ~~~
    python -m corpora.dd.build_response_eval_dataset --amt_result_path {path_to_the_DailyDialog_file}
    ~~~

### Unsupervised training
~~~
CUDA_VISIBLE_DEVICES=0 python -m tasks.response_eval.train_unsupervised --corpus dd --model roberta --model_size large --tokenizer roberta --init_lr 3e-6 --batch_size 3 --eval_batch_size 30 --seed 10 --validate_after_n_step 5000 --n_epochs 2 --enable_log True --save_model True
~~~

### Supervised training
~~~
CUDA_VISIBLE_DEVICES=0 python -m tasks.response_eval.train_supervised --corpus dd --model roberta --model_size large --tokenizer roberta --init_lr 3e-6 --batch_size 3 --eval_batch_size 30 --seed 42 --enable_log True --save_model True --model_path {path_to_the_model_trained_unsupervisedly}
~~~

### Evaluate
~~~
CUDA_VISIBLE_DEVICES=0 python -m tasks.response_eval.eval --corpus dd --model roberta --model_size large --tokenizer roberta --eval_batch_size 30 --model_path {path_to_the_model_trained_supervisedly} --output_model_name roberta_large_supervised
~~~

### Evaluate the same model on the PersonaChat corpus
* Download Amazon MTurk annotations on the PersonaChat corpus from [Google Drive](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt?usp=sharing) or [Zenodo](https://zenodo.org/record/3828180).

* Detect and mark outliers.
    ~~~
    python -m tasks.amt.mark_outliers --amt_result_path {path_to_the_PersonaChat_file}
    ~~~

* Build response evaluation dataset.
    ~~~
    python -m corpora.personachat.build_response_eval_dataset --amt_result_path {path_to_the_PersonaChat_file}
    ~~~

* Evaluate the model trained as above.
    ~~~
    CUDA_VISIBLE_DEVICES=0 python -m tasks.response_eval.eval --corpus personachat --model roberta --model_size large --tokenizer roberta --eval_batch_size 30 --model_path {path_to_the_model_trained_supervisedly} --output_model_name roberta_large_supervised
    ~~~

## Apply RoBERTa-eval to scoring arbitrary dialogues
* Obtain a trained RoBERTa-eval model following instructions above or from [Zenodo)](https://zenodo.org/record/3828286).

* Run example script.
    ~~~
    python -m tasks.response_eval.apply_roberta_eval --model_path {path_to_the_trained_model}
    ~~~

* Adjust the example code and test on your own data :)