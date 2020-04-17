import os
import code
import json
import argparse
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", help="path to directory of test output json files")
    parser.add_argument("--save_as", help="path resulting json file")
    args = parser.parse_args()

    # Retrieve log files
    output_file_paths = []
    assert os.path.isdir(args.model_output_dir)
    for filename in os.listdir(args.model_output_dir):
        if filename.endswith("eval_samples.json"):
            output_file_paths.append(f"{args.model_output_dir}/{filename}")
    output_file_paths = sorted(output_file_paths)

    # Load model output json files
    dialogs = {}
    for model_output_path in output_file_paths:
        with open(model_output_path, encoding="utf-8") as f:
            model_output_json = json.load(f)
            _model_name = model_output_json["meta_information"]["model_name"]
            model_attributes = _model_name.split(".")
            model_base_name, floor_encoder, seed, timestamp, _, _, dec1, dec2, dec3 = model_attributes
            decode_description = f"{dec1}.{dec2}.{dec3}"
            decode_method, temp, top_k, top_p = decode_description.split("_")
            model_name = f"{model_base_name} {decode_description}"

            samples = model_output_json["samples"]
            for sample in samples:
                sample_idx = str(sample["sample_idx"])
                context = sample["context"]
                reference = sample["reference"]
                hypothesis = sample["hypothesis"]
                ref_floor, ref_uttr = reference
                hyp_floor, hyp_uttr = hypothesis

                if sample_idx not in dialogs:
                    dialogs[sample_idx] = {
                        "context": context,
                        "reference": reference,
                        "responses": {
                            "ground-truth": {
                                "uttr": ref_uttr,
                            }
                        }
                    }

                dialogs[sample_idx]["responses"][model_name] = {
                    "uttr": hyp_uttr,
                }

    # Save collected results
    with open(args.save_as, "w+", encoding="utf-8") as f:
        json.dump(dialogs, f)

