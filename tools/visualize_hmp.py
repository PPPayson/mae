from matplotlib import pyplot as plt
import numpy as np
import os 

if __name__ == '__main__':
    print("Visualize")
    hmp_path = 'assets/jay_imagenet_for_co3d_val_0.1_processed.npz'
    human_results_file = 'assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz'
    os.makedirs('outs/visualize', exist_ok=True)
    all_data = np.load(hmp_path, allow_pickle=True)
    human_data = np.load(human_results_file, allow_pickle=True)

    filtered_imgs = human_data['final_clickmaps'].tolist().keys()
    all_ceiling = human_data['ceiling_correlations'].tolist()
    all_null = human_data['null_correlations'].tolist()
    image_files = []
    heatmaps = []
    image_names = []
    print(len(filtered_imgs))
    print(len(all_data.files))
    # self.ceiling = np.mean(all_ceiling)
    for i, img_name in enumerate(filtered_imgs):
        cat = img_name.split('/')[0]
        if img_name in all_data.files and all_ceiling[i] < 0:
            hmp = all_data[img_name].tolist()['heatmap']
            image_files.append(all_data[img_name].tolist()['image'])
            heatmaps.append(hmp)
            image_names.append(img_name)


    for img_name, hmp, img in zip(image_names, heatmaps, image_files):
        img_name = img_name.split('/')[1]
        hmp = np.mean(hmp, 0)
        img = np.array(img)
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(hmp)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join('outs/visualize/', img_name), bbox_inches='tight')
        plt.close()



