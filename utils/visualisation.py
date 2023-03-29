import random
import numpy as np
import matplotlib.pyplot as plt

def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]).astype(np.float32)

def remove_ceiling(points):
    points_mask = points[..., 2] < np.max(points[..., 2]) - 1
    points = points[points_mask]
    return points

def visualise_dict_counts(counts_dict, title = '', file_name=None):
    class_names = list(counts_dict.keys())
    counts = np.array(list(counts_dict.values()))
    counts = counts.astype(np.float32)
    counts = list(counts)

    fig = plt.figure(figsize = (15, 7.5))
    plt.bar(class_names, counts, color ='#9fb4e3', width = 0.4)
    plt.xticks(rotation=55)
    plt.title(title)
    plt.show()

    if file_name is not None:
        plt.savefig(file_name)
    