# -*- coding: utf-8 -*-
import code
import json
import collections
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.statistics import OutlierDetector, InterAnnotatorAgreementMetrics
from utils.metrics import SentenceMetrics
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from utils.statistics import CorrelationMetrics

def str2bool(v):
    return v.lower() in ('true', '1', "True")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True, help="path to annotation results")
    parser.add_argument("--rm_outlier", type=str2bool, default=False, help="remove outlier")

    parser.add_argument("--result_statistics", type=str, help="show result statistics")
    
    parser.add_argument("--read_response", type=str2bool, default=False, help="read responses with their scores")

    parser.add_argument("--plot", type=str2bool, default=False, help="draw plots")
    parser.add_argument("--target_score", type=str, default="overall", help="target score to draw about")
    parser.add_argument("--plot_type", type=str, help="plot to draw")
    parser.add_argument("--plot_output_path", type=str, help="path to output plot")

    parser.add_argument("--agreement", type=str2bool, default=False, help="compute agreement")

    parser.add_argument("--compute_cor", type=str2bool, help="compute correlation")
    parser.add_argument("--cor_type", type=str, default="uttr_pearson", help="type of correlations")
    parser.add_argument("--cor_plot_path", type=str, help="path to correlation plot")
    parser.add_argument("--cor_output_path", type=str, help="path to output correlation file")
    args = parser.parse_args()

    ## Names
    SCORE_NAMES = [
        "grammar", 
        "fact", 
        "content", 
        "relevance", 
        "overall"
    ]
    MODEL_NAMES = [
        "ground-truth", 
        "GPT2_medium", 
        "GPT2_small", 
        "HRED_attn", 
        "VHRED_attn", 
        "S2S_attn", 
        "S2S", 
        "negative-sample"
    ]
    DECODE_METHOD_NAMES = [
        "greedy_temp1.0_k0_p0.0", 
        "top_temp1.0_k0_p0.5", 
        "sample_temp1.0_k0_p0.0"
    ]
    MODEL_NAMES_ABR_MAP = collections.OrderedDict({
        "ground-truth": "GT",
        "GPT2_medium": "GPT2_md",
        "GPT2_small": "GPT2_sm",
        "HRED_attn": "HRED",
        "VHRED_attn": "VHRED",
        "S2S_attn": "Attn-Seq2seq",
        "S2S": "Seq2seq",
        "negative-sample": "NS",
    })
    METRICS_NAMES = ["bleu", "avg_emb_sim", "ext_emb_sim", "greedy_emb_sim", "distinct1", "distinct2", "length"] + ["overall"]
    LABLE_NAME_MAPPINGS = {
        "bleu": "BLEU",
        "avg_emb_sim": "Average",
        "ext_emb_sim": "Extrema",
        "greedy_emb_sim": "Greedy",
        "distinct1": "Distinct 1",
        "distinct2": "Distinct 2",
        "length": "Sentence length",
        "grammar": "Grammar",
        "content": "Content",
        "relevance": "Relevance",
        "overall": "Overall"
    }

    ## Load results
    with open(args.result_path, encoding="utf-8") as f:
        result_json = json.load(f)

    ## Extract data from results
    response_score_df = pd.DataFrame()
    fact_count = collections.defaultdict(int)
    worker_id_d = {}
    for dialog_id, dialog in result_json.items():
        for model_name, response in dialog["responses"].items():
            if model_name not in ["ground-truth", "negative-sample"]:
                model_name, decode_method = model_name.split(" ")
            else:
                model_name, decode_method = model_name, "human"
            if "scores" in response:
                score_name2list = collections.defaultdict(list)
                for worker_annotation in response["scores"].values():
                    worker_id = worker_annotation["worker_id"]
                    worker_id_d[worker_id] = True
                    for score_name in SCORE_NAMES:
                        score = worker_annotation[score_name]
                        if args.rm_outlier and worker_annotation[f"{score_name}_is_outlier"]:
                            continue

                        if score_name == "fact":
                            fact_count[score] += 1
                            if score == "na":
                                continue
                            score = 1 if score == "true" else 0

                        score_name2list[score_name].append(score)
                
                response_df_item = {
                    "text": response["uttr"],
                    "model": model_name,
                    "model_abs": MODEL_NAMES_ABR_MAP[model_name],
                    "decode_method": decode_method,
                    "model_decode_method": f"{model_name}_{decode_method}"
                }
                for score_name in score_name2list.keys():
                    response_df_item.update({f"{score_name}_list": score_name2list[score_name]})
                    response_df_item.update({score_name: np.mean(score_name2list[score_name])})
                
                response_score_df = response_score_df.append(response_df_item, ignore_index=True)

    print(f"{len(worker_id_d)} workers")
    
    ## Output score- and model-wise statistics CSV file
    if args.result_statistics:
        print(response_score_df.groupby("model").mean())
        print(response_score_df.groupby("decode_method").mean())
        print(response_score_df.groupby("model_decode_method").mean())
        code.interact(local=locals())

    ## Plot score-wise statisitcs
    if args.plot:
        
        ## Score-wise distribution
        if args.plot_type == "score_dist":
            fig = plt.figure(figsize=(3, 3))
            scores = response_score_df[args.target_score].tolist()
            # code.interact(local=locals())

            # Plot a simple histogram with binsize determined automatically
            sns.set(style="whitegrid")
            ax = sns.distplot(
                scores, 
                bins=np.arange(1,7)-0.5, 
                kde=False, 
                color="b",
                hist_kws={"alpha": 1, "color": "steelblue", "rwidth": 0.8, "align": "mid"}
            )
            plt.xticks([1,2,3,4,5])
            plt.xlabel("Score")
            plt.ylabel("Number of responses")

            # plt.setp(axes)
            plt.tight_layout()
            if args.plot_output_path:
                plt.savefig(args.plot_output_path, format='eps')
            else:
                plt.show()

        ## System-wise boxplot
        if args.plot_type == "sys_box":
            fig = plt.figure(figsize=(8, 4))
            sns.set(style="whitegrid")
            sns.boxplot(
                x="model_abs",
                y=args.target_score,
                data=response_score_df,
            )
            plt.xlabel("")
            # plt.xticks(rotation=90)
            plt.ylabel("Score")
            plt.yticks([1,2,3,4,5])
            plt.tight_layout()
            if args.plot_output_path:
                plt.savefig(args.plot_output_path, format='eps')
            else:
                plt.show()

        ## Uttr-wise scatter plot
        if args.plot_type == "uttr_scatter":
            sns.set(style="white", palette="muted", color_codes=True)
            # fig = plt.figure(figsize=(7, 7))
            scores1 = response_score_df["overall"].tolist()
            scores2 = response_score_df[args.target_score].tolist()
            sns.regplot(
                x=scores1, 
                y=scores2, 
                x_jitter=.0, 
                y_jitter=.1
            )
            plt.xlabel("overall")
            plt.ylabel(args.target_score)
            plt.xticks([1, 2, 3, 4, 5])
            plt.yticks([1, 2, 3, 4, 5])
            plt.tight_layout()
            if args.plot_output_path:
                plt.savefig(args.plot_output_path, format='eps')
            else:
                plt.show()

        ## Sys-wise scatter plot
        if args.plot_type == "sys_scatter":
            sns.set(style="white", palette="muted", color_codes=True)
            # fig = plt.figure(figsize=(7, 7))
            sys_score_df = response_score_df.groupby("model_decode_method").mean()
            scores1 = sys_score_df["overall"].tolist()
            scores2 = sys_score_df[args.target_score].tolist()
            sns.regplot(
                x=scores1, 
                y=scores2
            )
            plt.xlabel("overall")
            plt.ylabel(args.target_score)
            plt.xticks([1, 2, 3, 4, 5])
            plt.yticks([1, 2, 3, 4, 5])
            plt.tight_layout()
            if args.plot_output_path:
                plt.savefig(args.plot_output_path, format='eps')
            else:
                plt.show()

    ## Compute inter-annotator agreement
    if args.agreement:
        metrics = InterAnnotatorAgreementMetrics()
        for score_name in SCORE_NAMES:
            # Krippendorff's alpha
            # construct input
            annotation_dict = collections.defaultdict(dict)
            for dialog_id, dialog in result_json.items():
                for model_name, response in dialog["responses"].items():
                    if "scores" not in response:
                        continue
                    unit_name = f"{dialog_id} {model_name}"
                    for coder_name, coder_annotation in response["scores"].items():
                        if args.rm_outlier and coder_annotation[f"{score_name}_is_outlier"]:
                            continue
                        score = coder_annotation[score_name]
                        annotation_dict[coder_name][unit_name] = score
            annotation_list = list(annotation_dict.values())

            # calculation
            if score_name == "fact":
                metric_type = "nominal"
                convert_items = str
            else:
                metric_type = "interval"
                convert_items = float
            krippendorff_alpha = metrics.krippendorff_alpha(annotation_list, metric_type=metric_type, convert_items=convert_items)
            print(f"Krippendorff's alpha for {score_name} is {krippendorff_alpha}.")

    ## Compute correlations between metrics
    if args.compute_cor:
        ## Load corpus config
        from corpora.dd.config import Config
        corpus_config = Config(task="response_eval")

        ## Metrics calculator
        eval_tokenizer = WhiteSpaceTokenizer(
            word_count_path=corpus_config.word_count_path,
            vocab_size=10000
        )
        metrics = SentenceMetrics(corpus_config.eval_word_embedding_path, eval_tokenizer)

        ## Get data
        with open(args.result_path, encoding="utf-8") as f:
            result_json = json.load(f)
        data_points = []
        for dialog_id, dialog in result_json.items():
            context = dialog["context"]
            ref_floor, ref_text = dialog["reference"]

            for model_name, response in dialog["responses"].items():
                if model_name == "ground-truth":
                    continue

                # skip unannotated responses
                if "scores" not in response:
                    continue

                if " " in model_name:
                    model_name, decode_method = model_name.split(" ")
                else:
                    model_name, decode_method = model_name, "human"

                data_point = {
                    "model_name": model_name,
                    "decode_method": decode_method,
                    "context": context,
                    "reference_floor": ref_floor,
                    "reference_text": ref_text,
                    "hypothesis_text": response["uttr"],
                    "metrics": {}
                }

                for score_name in SCORE_NAMES:
                    scores = []
                    for worker_id, worker_annotation in response["scores"].items():

                        # skip outlier annotations
                        if worker_annotation[f"{score_name}_is_outlier"]:
                            continue

                        # convert scores
                        score = worker_annotation[score_name]
                        if score_name == "fact":
                            if score == "na":
                                continue
                            else:
                                score = 1 if score == "true" else 0

                        scores.append(score)
                    average_score = np.mean(scores)
                    data_point["metrics"][score_name] = average_score

                data_points.append(data_point)

        ## Compute automated metrics
        for data_point in data_points:
            ref = data_point["reference_text"]
            hyp = data_point["hypothesis_text"]

            # BLEU
            bleu_scores = metrics.batch_bleu([hyp], [ref])
            bleu = np.mean(bleu_scores)
            # Embedding similarities
            avg_emb_sims, ext_emb_sims, greedy_emb_sims = metrics.batch_sim_bow([hyp], [ref])
            avg_emb_sim = np.mean(avg_emb_sims)
            ext_emb_sim = np.mean(ext_emb_sims)
            greedy_emb_sim = np.mean(greedy_emb_sims)
            # SIF embedding similarity
            # sif_emb_sims = metrics.batch_sif_emb_sim([hyp], [ref])
            # sif_emb_sim = np.mean(sif_emb_sims)
            # Distinct n-grams
            intra_dist1, intra_dist2, _, _, \
                intra_types1, intra_types2, _, _ \
                = metrics.batch_div_distinct([hyp])
            # Average sentence length
            hyp_tokens = eval_tokenizer.convert_string_to_tokens(hyp)
            hyp_len = len(hyp_tokens)

            data_point["metrics"].update({
                "bleu": bleu,
                "avg_emb_sim": avg_emb_sim,
                "ext_emb_sim": ext_emb_sim,
                "greedy_emb_sim": greedy_emb_sim,
                # "sif_emb_sim": sif_emb_sim,
                "distinct1": intra_types1,
                "distinct2": intra_types2,
                "length": hyp_len,
            })

        ## Compute utterance-level correlation
        cor_metrics = CorrelationMetrics()
        uttr_cor_results = {}
        metric_stats = {}
        for metric1 in METRICS_NAMES:
            if metric1 in uttr_cor_results:
                continue
            uttr_cor_results[metric1] = {}
            data1 = []
            for data_point in data_points:
                data1.append(data_point["metrics"][metric1])
            data1 = np.array(data1)
            for metric2 in METRICS_NAMES:
                data2 = []
                for data_point in data_points:
                    data2.append(data_point["metrics"][metric2])
                data2 = np.array(data2)
                pearson_r, pearson_p = cor_metrics.pearson_cor(data1, data2)
                spearman_r, spearman_p = cor_metrics.spearman_cor(data1, data2)
                uttr_cor_results[metric1][metric2] = {
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            metric_stats[metric1] = {
                "mean": np.mean(data1*4+1),
                "std": np.std(data1*4+1)
            }
        print(metric_stats)

        ## Compute system-level correlation
        sys_cor_results = {}
        for metric1 in METRICS_NAMES:
            if metric1 in sys_cor_results:
                continue
            sys_cor_results[metric1] = {}
            for metric2 in METRICS_NAMES:
                data1, data2 = collections.defaultdict(list), collections.defaultdict(list)
                for data_point in data_points:
                    model_name = f'{data_point["model_name"]} {data_point["decode_method"]}'
                    data1[model_name].append(data_point["metrics"][metric1])
                    data2[model_name].append(data_point["metrics"][metric2])
                for model_name in data1.keys():
                    data1[model_name] = np.mean(data1[model_name])
                for model_name in data2.keys():
                    data2[model_name] = np.mean(data2[model_name])
                data1 = np.array(list(data1.values()))
                data2 = np.array(list(data2.values()))
                pearson_r, pearson_p = cor_metrics.pearson_cor(data1, data2)
                spearman_r, spearman_p = cor_metrics.spearman_cor(data1, data2)
                sys_cor_results[metric1][metric2] = {
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }

        ## Plot heatmap of correlation
        if args.cor_type in ["uttr_pearson", "uttr_spearman"]:
            cor_results = uttr_cor_results
        elif args.cor_type in ["sys_pearson", "sys_spearman"]:
            cor_results = sys_cor_results
        matrices = {
            "pearson_r": [],
            "pearson_p": [],
            "spearman_r": [],
            "spearman_p": []
        }
        for metric1 in METRICS_NAMES:
            for key in matrices.keys():
                matrices[key].append([])
            for metric2 in METRICS_NAMES:
                for key in matrices.keys():
                    matrices[key][-1].append(cor_results[metric1][metric2][key])
        for key in matrices:
            matrices[key] = np.array(matrices[key])

        LABEL_NAMES = []
        for metric_name in METRICS_NAMES:
            if metric_name in LABLE_NAME_MAPPINGS:
                LABEL_NAMES.append(LABLE_NAME_MAPPINGS[metric_name])
            else:
                LABEL_NAMES.append(metric_name)

        if args.cor_type in ["uttr_pearson", "sys_pearson"]:
            corr_r = matrices["pearson_r"]
            corr_p = matrices["pearson_p"]
        elif args.cor_type in ["uttr_spearman", "sys_spearman"]:
            corr_r = matrices["spearman_r"]
            corr_p = matrices["spearman_p"]

        def get_annot(r, p):
            r_str = f"{r:.2f}"

            if p <= 0.001:
                return r"${}^{{**}}$".format(r_str)
            elif p <= 0.01:
                return r"${}^*$".format(r_str)
            else:
                return r"${}$".format(r_str)
        corr_annot = np.vectorize(get_annot)(corr_r, corr_p)
        np.fill_diagonal(corr_annot, "")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_r, dtype=np.bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        sns.set(style="white", palette="muted", color_codes=True)
        f, ax = plt.subplots(figsize=(12, 6))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(240, 10, l=55, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            data=corr_r,
            annot=corr_annot, fmt="s", annot_kws={"size": 10, "color": "black", "ha": "center", "va": "center"},
            mask=mask,
            cmap=cmap,
            vmin=-1.0, vmax=1.0, center=0.0,
            linewidths=1.0,
            cbar_kws={"shrink": .5},
        )
        ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticklabels(LABEL_NAMES, rotation=0, va="center", ha="right")
        ax.hlines(y=[7, 10], xmin=[0, 0], xmax=[7, 10])
        ax.vlines(x=[7, 10], ymin=[7, 10], ymax=[14, 14])
        plt.tight_layout()
        if args.cor_plot_path:
            plt.savefig(args.cor_plot_path, format='eps')
        else:
            plt.show()

        ## Save to file
        if args.cor_output_path:
            with open(args.cor_output_path, "w+", encoding="utf-8") as f:
                for title, cor_results in zip(["utterance-level correlation", "system-level correlation", "clustered system-level correlation"], [uttr_cor_results, sys_cor_results, clustered_sys_cor_results]):
                    f.write(f"{title}\n")
                    for cor_metric in ["pearson", "spearman"]:
                        header = '\t'.join([cor_metric]+METRICS_NAMES)
                        f.write(f"{header}\n")
                        for metric_row in METRICS_NAMES:
                            f.write(f"{metric_row}")
                            for metric_col in METRICS_NAMES:
                                r = cor_results[metric_row][metric_col][f"{cor_metric}_r"]
                                p = cor_results[metric_row][metric_col][f"{cor_metric}_p"]
                                f.write(f"\t{r:.2g}")
                            f.write("\n")
                        f.write("\n"*3)
