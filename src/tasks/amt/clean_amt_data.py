# -*- coding: utf-8 -*-
import code
import json
import argparse


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amt_result_path", type=str, required=True, help="path to AMT results")
    parser.add_argument("--output_path", type=str, required=True, help="path to cleaned output")
    args = parser.parse_args()

    # Load results
    with open(args.amt_result_path, encoding="utf-8") as f:
        data = json.load(f)

    clean_data = {}
    hit_cnt = 0
    worker_id2new_worker_id = {}
    for dialog_id, dialog in data.items():
        clean_dialog = {
            "context": dialog["context"],
            "reference": dialog["reference"],
            "responses": {}
        }

        for model_name, response in dialog["responses"].items():
            if "scores" not in response:
                continue
            
            clean_scores = {}
            for hit_id, score in response["scores"].items():
                new_hit_id = hit_cnt
                hit_cnt += 1

                worker_id = score["worker_id"]
                if worker_id not in worker_id2new_worker_id:
                    worker_id2new_worker_id[worker_id] = len(worker_id2new_worker_id)
                new_worker_id = worker_id2new_worker_id[worker_id]
                
                clean_scores[new_hit_id] = {
                    "worker_id": new_worker_id
                }

                for score_name in ["content", "fact", "grammar", "overall", "relevance"]:
                    if score_name in score:
                        clean_scores[new_hit_id][score_name] = score[score_name]
            
            clean_dialog["responses"][model_name] = {
                "uttr": response["uttr"],
                "scores": clean_scores
            }
        
        clean_data[dialog_id] = clean_dialog

    with open(args.output_path, "w+", encoding="utf-8") as f:
        json.dump(clean_data, f)
