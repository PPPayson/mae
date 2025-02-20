import timm
import json
import pandas as pd
import csv
import os
import subprocess
from tools.eval_alignment import get_args_parser
from tools.eval_alignment import main as eval_alignment


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    log_file = f'outs/alignment_results.json'
    timm_df = pd.read_csv(timm_results_csv)
    num_models = len(timm_df.index)

    for i in range(num_models):
        with open(log_file, 'r') as f:
            alignment_results = json.load(f)
        model_name = timm_df['model'][i]
        input_size = timm_df['img_size'][i]
        if input_size != 224:
            continue
        args.model = model_name
        args.output_dir = os.path.join('outs/linear_probe', model_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print(args.model)
        try:
            if model_name in alignment_results and alignment_results[model_name] != "Failed":
                continue
            outputs = eval_alignment(args)
                # acc = outputs['acc']
                # co3d_align = outputs['co3d_alignment']
                # imgnet_align = outputs['imgnet_alignment']
            alignment_results[model_name] = outputs
        except:
             alignment_results[model_name] = "Failed"
        
        with open(log_file, 'w') as f:
            results_json = json.dumps(alignment_results, indent=4)
            f.write(results_json)    
    