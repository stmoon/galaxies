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

PREFIX = './data/set100/'

ORIGIN_TRAIN_DATA_PATH = PREFIX + 'train'
ORIGIN_VALID_DATA_PATH =  PREFIX + 'valid'
ORIGIN_TEST_DATA_PATH =  PREFIX + 'test'

RESIZE_TRAIN_DATA_PATH = PREFIX + 'resized_train'
RESIZE_VALID_DATA_PATH = PREFIX + 'resized_valid'
RESIZE_TEST_DATA_PATH = PREFIX + 'resized_test'

TRAIN_LABEL_CSV_FILE = PREFIX + 'train_solutions.csv'
VALID_LABEL_CSV_FILE = PREFIX + 'valid_solutions.csv'
TEST_LABEL_CSV_FILE = PREFIX + 'test_solutions.csv'

MODIFIED_TRAIN_LABEL_CSV_FILE = PREFIX + 'modified_train_solutions.csv'
MODIFIED_VALID_LABEL_CSV_FILE = PREFIX + 'modified_valid_solutions.csv'
MODIFIED_TEST_LABEL_CSV_FILE = PREFIX + 'modified_test_solutions.csv'



### Prepare image data for training
def prepare_image() :
    resize_data(ORIGIN_TRAIN_DATA_PATH, RESIZE_TRAIN_DATA_PATH)
    resize_data(ORIGIN_VALID_DATA_PATH, RESIZE_VALID_DATA_PATH)
    resize_data(ORIGIN_TEST_DATA_PATH, RESIZE_TEST_DATA_PATH)

    modified_label(TRAIN_LABEL_CSV_FILE, MODIFIED_TRAIN_LABEL_CSV_FILE, RESIZE_TRAIN_DATA_PATH)
    modified_label(VALID_LABEL_CSV_FILE, MODIFIED_VALID_LABEL_CSV_FILE, RESIZE_VALID_DATA_PATH)
    modified_label(TEST_LABEL_CSV_FILE, MODIFIED_TEST_LABEL_CSV_FILE, RESIZE_TEST_DATA_PATH)

def resize_data(original_path, modified_path) :
    ### Create directory
    if os.path.exists(modified_path):
        print "Already exist resize train directory"
    else:
        os.mkdir(modified_path)

        for data in tqdm(os.listdir(original_path)):
            ori_img = cv2.imread(original_path+'/'+data)
            crop_img = ori_img[108:108+207, 108:108+207]  # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

            img = cv2.resize(crop_img, (128, 128))
            cv2.imwrite(modified_path+'/'+data, img)

def modified_label(original_file, modified_file, path):

    with open(original_file) as csvfile, open(modified_file, 'wb') as writefile:    
        
        reader = csv.reader(csvfile, delimiter=',')    
        writer = csv.writer(writefile, delimiter=',')

        next(reader, None)  

        for row in tqdm(reader):                        
            file_path = path + '/' + row[0] + '.jpg'
            # FIXME: add leaf node 
	    #  4,11,12,13,14,17,18,19,27,28,29
            #writer.writerow([file_path, row[3], row[10], row[11], row[12], row[13], row[16], row[17], row[18], row[26], row[27], row[28]])
            row[0] = file_path
            writer.writerow(row)



