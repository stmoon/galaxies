"""
1. Read file list
2. Read image using cv2
3. Resize image
4. Restore image(64x64 size and gray scale)
"""

import csv, cv2, os
from tqdm import tqdm
import csv
import numpy as np

ORIGIN_TRAIN_DATA_PATH = './data/images_training_rev1'
ORIGIN_TEST_DATA_PATH = './data/images_test_rev1'

RESIZE_TRAIN_DATA_PATH = './data/resized_train'
RESIZE_TEST_DATA_PATH = './data/resized_test'

TRAIN_LABEL_CSV_PATH = './data/training_solutions_rev1.csv'
MODIFIED_TRAIN_LABEL_CSV_PATH = './data/modified_training_solutions.csv'



### Prepare image data for training
def prepre_resize_train():
    ### Create directory
    if os.path.exists(RESIZE_TRAIN_DATA_PATH):
        print "Already exist resize train directory"
    else:
        os.mkdir(RESIZE_TRAIN_DATA_PATH)

        for data in tqdm(os.listdir(ORIGIN_TRAIN_DATA_PATH)):
            ori_img = cv2.imread(ORIGIN_TRAIN_DATA_PATH+'/'+data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            crop_img = ori_img[108:108+207, 108:108+207]  # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

            img = cv2.resize(crop_img, (64, 64))
            cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+data, img)


def prepare_resize_test():
    ### Create directory
    if os.path.exists(RESIZE_TEST_DATA_PATH):
        print "Already exist resize train directory"
    else:
        os.mkdir(RESIZE_TEST_DATA_PATH)

        for data in tqdm(os.listdir(ORIGIN_TEST_DATA_PATH)):
            ori_img = cv2.imread(ORIGIN_TEST_DATA_PATH+'/'+data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            crop_img = ori_img[108:108+207, 108:108+207]  # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

            img = cv2.resize(crop_img, (64, 64))
            cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+data, img)




def prepare_csv():

    with open(TRAIN_LABEL_CSV_PATH) as csvfile, open(MODIFIED_TRAIN_LABEL_CSV_PATH, 'wb') as writefile:    
        
        reader = csv.reader(csvfile, delimiter=',')    
	writer = csv.writer(writefile, delimiter=',')

	next(reader, None)  

        for row in tqdm(reader):                        
            file_path = RESIZE_TRAIN_DATA_PATH + '/' + row[0] + '.jpg'
            # FIXME: add leaf node 
	    #  4,11,12,13,14,17,18,19,27,28,29
            writer.writerow([file_path, row[3], row[10], row[11], row[12], row[13], row[16], row[17], row[18], row[26], row[27], row[28]])


