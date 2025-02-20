import json
import csv
import pandas as pd
import os 

if __name__ == "__main__":
    timm_results_csv = os.path.join('assets', 'results-imagenet.csv')
    timm_df = pd.read_csv(timm_results_csv)
    num_models = len(timm_df.index)
    reference_csv = os.path.join('assets', 'model_type_reference.csv')
    reference_df = pd.read_csv(reference_csv)

    alignment_file = 'outs/alignment_results.json'
    with open(alignment_file, 'r') as f:
        results = json.load(f)
    output = [['model', 'model_type', 'imgnet_acc', 'acc', 'full_co3d', 'full_imgnet', 'topk_co3d', 
                'topk_imgnet', 'full_unnorm_co3d', 'full_unnorm_imgnet', 'topk_unnorm_co3d',
                 'topk_unnorm_imgnet']]
    for i in range(num_models):
        model = timm_df['model'][i]
        input_size = timm_df['img_size'][i]
        if model not in results.keys():
            continue
        if 'Failed' in results[model]:
            continue
        model_index = reference_df.index[reference_df['model'] == model].tolist()[0]
        model_type = reference_df.iloc[model_index]['model_type']

        output.append([model, model_type, timm_df['top1'][i], results[model]['acc'], 
                        results[model]['full_co3d_alignment'],
                         results[model]['full_imgnet_alignment'],
                         results[model]['topk_co3d_alignment'],
                         results[model]['topk_imgnet_alignment'],                         
                         results[model]['full_unnorm_co3d_alignment'],
                         results[model]['full_unnorm_imgnet_alignment'],
                         results[model]['topk_unnorm_co3d_alignment'],
                         results[model]['topk_unnorm_imgnet_alignment'],
                         ])
    
    output_file = 'outs/alignment_results.csv'
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)