# -*- coding: utf-8 -*-
import code
import json
import argparse

from utils.statistics import OutlierDetector


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amt_result_path", type=str, required=True, help="path to annotation results")
    parser.add_argument("--n_dev", type=int, default=1, help="threshold: number of MAD deviations")
    args = parser.parse_args()

    # Names
    SCORE_NAMES = [
        "grammar",
        "fact",
        "content",
        "relevance",
        "overall"
    ]

    # Load results
    with open(args.amt_result_path, encoding="utf-8") as f:
        result_json = json.load(f)

    # Mark outlier
    outlier_detector = OutlierDetector()
    outlier_statistics = {}
    for score_name in SCORE_NAMES:
        outlier_statistics[score_name] = {"num_datapoints": 0, "num_outliers": 0}
        for dialog_id, dialog in result_json.items():
            for model_name, response in dialog["responses"].items():
                if "scores" not in response:
                    continue

                # aggregate scores
                scores = []
                for worker_result in response["scores"].values():
                    if score_name not in worker_result:
                        continue
                    score = worker_result[score_name]
                    if score_name == "fact":
                        if score == "na":
                            continue
                        score = 1 if score == "true" else 0
                    scores.append(score)

                # find outliers
                if len(scores) > 0:
                    outliers = outlier_detector.detect_by_abd_median(scores, n_dev=args.n_dev)

                # mark outliers
                for worker_result in response["scores"].values():
                    if score_name not in worker_result:
                        continue
                    score = worker_result[score_name]
                    if score_name == "fact":
                        score = 1 if score == "true" else 0

                    mark_name = f"{score_name}_is_outlier"
                    if score in outliers:
                        worker_result[mark_name] = True
                        outlier_statistics[score_name]["num_outliers"] += 1
                    else:
                        worker_result[mark_name] = False
                    outlier_statistics[score_name]["num_datapoints"] += 1

    print("Outlier statistics:")
    print(json.dumps(outlier_statistics, indent=2))

    # Save to file
    with open(args.amt_result_path, "w+", encoding="utf-8") as f:
        json.dump(result_json, f)
