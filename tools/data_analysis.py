import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET = {}
image_size = []
obj_ratio = []

def txt_parse(txt_anno_file):
    with open(txt_anno_file, 'r') as f:
        lines = f.readlines()
    
    annos = []
    if len(lines) > 0:
        for line in lines:
            if len(line.strip().split()) > 4:
                x1, y1, x2, y2, x3, y3, x4, y4, label, _ = line.strip().split()
                annos.append([[int(float(x1)), int(float(y1))], [int(float(x2)), int(float(y2))],\
                    [int(float(x3)), int(float(y3))], [int(float(x4)), int(float(y4))], label])
                if label not in DATASET:
                    DATASET[label] = 1
                else:
                    DATASET[label] += 1
                width = np.sqrt((float(y2)-float(y1)) ** 2 + (float(x2)-float(x1)) ** 2)
                height = np.sqrt((float(y3)-float(y2)) ** 2 + (float(x3)-float(x2)) ** 2)
                ratio = (min(width, height), max(width, height))
                print(ratio)
                obj_ratio.append(ratio)

    return annos

def get_class(txt_dir):
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    img_dir = '/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/images'
    img_prefix='.jpg'
    
    for i, txt_file in enumerate(txt_files):
        name = os.path.splitext(os.path.basename(txt_file))[0]
        img_path = os.path.join(img_dir, name + img_prefix)
        name = os.path.splitext(os.path.basename(txt_file))[0]
        annos = txt_parse(txt_file)

        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            image_size.append(img.shape)

    
    print(DATASET)
    data = np.array(image_size)
    # obj_ratio = np.array(obj_ratio)
    import pdb; pdb.set_trace()
    # min_val = np.min(data)
    # max_val = np.max(data)
    # interval_range = (max_val - min_val) / 5

    # # Divide the data into five intervals
    # intervals = [min_val + i * interval_range for i in range(6)]
    # counts, _ = np.histogram(data, bins=intervals)

    # # Plot the histogram
    # plt.bar(range(1, 6), counts)
    # plt.xlabel('Interval')
    # plt.ylabel('Count')
    # plt.title('Histogram of Data Intervals')
    # plt.xticks(range(1, 6))
    # plt.savefig("./resolution.jpg")

if __name__ == "__main__":
    txt_dir = "/mnt/cfs_bj/nihao/data/ICJAI2023/train_track2/annotations_dota"
    # {'battery': 1287, 'pressure': 663, 'umbrella': 451, 'OCbottle': 1180, 'glassbottle': 565, 'lighter': 1839, 'electronicequipment': 1856, 'knife': 2852, 'metalbottle': 987}
    get_class(txt_dir)
