import code
import json
import random
import argparse
import os


def are_different_uttrs(tokens1, tokens2):
    token_set1 = set(tokens1)
    token_set2 = set(tokens2)
    intersection = token_set1.intersection(token_set2)
    coverage_score = (len(intersection)/len(token_set1) + len(intersection)/len(token_set2))/2
    if coverage_score < 0.8:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to model outputs json file")
    parser.add_argument("--save_as", help="path resulting json file")
    parser.add_argument("--n_samples", default=100, type=int, help="number of dialog samples")
    parser.add_argument("--seed", default=42, help="randomness seed")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.data_path, encoding="utf-8") as f:
        data_json = json.load(f)
    
    # collecting human sentences for negative response sample
    negative_sample_pool = []
    for dialog in data_json.values():
        context = dialog["context"]
        for speaker, text in context:
            negative_sample_pool.append(text)

    # sample n_sample dialogs
    samples = random.sample(data_json.items(), args.n_samples)
    samples = sorted(samples, key=lambda x: int(x[0]))

    # write n samples to file
    sample_json = {}
    for sample in samples:
        dialog_id, dialog = sample
        context = dialog["context"]
        reference = dialog["reference"]
        responses = dialog["responses"]
        ref_speaker, ref_text = reference
        while True:
            negative_sample = random.choice(negative_sample_pool)
            ref_tokens = ref_text.split(" ")
            sample_tokens = negative_sample.split(" ")
            if are_different_uttrs(ref_tokens, sample_tokens):
                dialog["responses"]["negative-sample"] = {
                    "uttr": negative_sample
                }
                break
        sample_json[dialog_id] = dialog

    # save sampled data
    if args.save_as:
        with open(args.save_as, "w+", encoding="utf-8") as f:
            json.dump(sample_json, f)



            
