import argparse
import glob
import os
from pprint import pprint

import pandas
import sys
pandas.set_option("display.precision", 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="MSRVTT", choices=['Youtube2Text', 'MSRVTT', 'VATEX', 'myV'])
    parser.add_argument("-ss", "--skip_scopes", nargs='+', type=str, default=['test_'],
                        help="skip scopes whose names are exact one of these")
    parser.add_argument("-sm", "--skip_models", nargs='+', type=str, default=[])
    parser.add_argument("-tasks", "--tasks", nargs='+', type=str, default=[])
    parser.add_argument("-sv", "--sort_values", nargs='+', type=str, default=['model_name', 'task_name', 'scope_name'])
    parser.add_argument("-name", "--output_name", type=str, default="merged_all_csv",
                        help="output file name.")
    parser.add_argument("--csv_name", type=str, default="test_result.csv", )
    parser.add_argument("--round", type=int, default=2)

    args = parser.parse_args()

    BASE_PATH = os.path.join("../experiments", args.dataset)
    # find path
    path = os.path.join(BASE_PATH, f"*/*/*/{args.csv_name}")
    models_paths = glob.glob(path)
    models_paths = sorted(models_paths)

    # skip some file
    new_paths = []
    for path in models_paths:
        ps = path.split("/")
        model_name, task_name, scope_name = ps[3:6]
        if model_name in args.skip_models:
            continue
        if scope_name in args.skip_scopes:
            continue
        if len(args.tasks) and task_name not in args.tasks:
            continue
        print(ps)
        new_paths.append(path)
    models_paths = new_paths

    # merge
    csv_data = []
    for i, path in enumerate(models_paths):
        ps = path.split("/")
        model_name, task_name, scope_name = ps[3:6]
        csv_df = pandas.read_csv(path)
        csv_df.insert(0, "model_name", model_name)
        csv_df.insert(1, "task_name", task_name)
        csv_df.insert(2, "scope_name", scope_name)
        csv_data.append(csv_df)
        # print(csv_df)
    assert len(csv_data) > 0, f"No test data in `experiments` dir for dataset `{args.dataset}`"
    all_df = pandas.concat(csv_data).sort_values(args.sort_values)
    # to percentage and .2 precision
    # all_df[all_df.select_dtypes(include=['number']).columns] *= 100
    all_df[["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "Sum", "novel", "unique"]] *= 100
    all_df = all_df.round(args.round)
    all_df["Sum"] = all_df["Bleu_4"] + all_df["METEOR"] + all_df["ROUGE_L"] + all_df["CIDEr"]
    # all_df[all_df.select_dtypes(include=['number']).columns].map(lambda x: '%.2f' % x)
    pprint(all_df)
    # finalnp = finaldf.to_numpy()
    # output
    output_file_name = args.output_name if ".csv" in args.output_name else args.output_name + ".csv"
    output_path = os.path.join(BASE_PATH, f"{output_file_name}")
    all_df.to_csv(output_path, index=False)

