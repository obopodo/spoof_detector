import numpy as np

def color_diversity(img):
    quantized = (img // 8)
    quantized = quantized.reshape(-1, 3)

    colors_distribution = np.unique(quantized, axis=0, return_counts=True)
    n_distinct_colors = colors_distribution[0].shape[0]
    top_100_colors_counts = np.sort(colors_distribution[1])[-100:]
    result = np.append(top_100_colors_counts, n_distinct_colors)
    return result

def color_diversity_hist(img):
    quantized = (img // 8)
    channels = [quantized[:,:,i].flatten() for i in range(3)]
    img_flatten = channels[0] + 32*channels[1] + 32*32*channels[2]
    n_distinct_colors = np.unique(img_flatten).shape[0]
    bins_counts = np.sort(np.histogram(img_flatten, bins=100)[0])
    result = np.append(bins_counts, n_distinct_colors)
    return
