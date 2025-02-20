import numpy as np
import os 
import shutil
import glob
import json
from test_sliding_window import compute_spearman_correlation

if __name__ == "__main__":
    output_dir = 'outs/collected_maps'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'human'), exist_ok=True)
    ceiling_data = np.load('assets/human_ceiling_split_half_co3d_val.npz', allow_pickle=True)
    img_names = ceiling_data['final_clickmaps'].tolist().keys()
    all_data = np.load('assets/co3d_val_processed.npz', allow_pickle=True)

    target_list = ['611_97356_193537', '581_86253_171231', '412_56308_109311', '590_88868_176181', '433_61324_11945']
    names = []
    full_names = []
    for i, k in enumerate(img_names):
        for t in target_list:
            if t in k and k in all_data.files:
                heatmaps = all_data[k].tolist()['heatmap']
                ceiling = ceiling_data['ceiling_correlations'].tolist()[i]
                print(k, t, ceiling)
                name = k.split('.')[0].split('/')[1]
                np.save(os.path.join(output_dir, 'human', f'{name}.npy'), heatmaps)
                names.append(name)
                full_names.append(k)               
    # all_scores = {}
    # for model in os.listdir('outs/linear_probe'):
    #     all_scores[model] = {}
    #     os.makedirs(os.path.join(output_dir, model), exist_ok=True)
    #     heatmaps = glob.glob(os.path.join('outs/linear_probe', model, 'co3d/*.npy'))
    #     imgs = glob.glob(os.path.join('outs/linear_probe', model, 'co3d/*.png'))
    #     for img in imgs:
    #         for n in names:
    #             if n in img:
    #                 score = img.split("/")[-1].split('_')[0]
    #                 all_scores[model][n] = score
    #                 shutil.copy(img, os.path.join(output_dir, model, f'{n}.png'))
    
    #     for heatmap in heatmaps:
    #         for n in names:
    #             if n in heatmap:
    #                 shutil.copy(heatmap, os.path.join(output_dir, model, f'{n}.npy'))
    
    # with open(os.path.join(output_dir, 'model_score.json'), 'w') as f:
    #     json_output = json.dumps(all_scores, indent=4)
    #     f.write(json_output)