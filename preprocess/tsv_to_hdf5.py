"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
from six.moves import cPickle
import numpy as np
import utils
import argparse
import json
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

feature_length = 2048
num_fixed_boxes = 36


def count(infile, FIELDNAMES):
    cnt = 0
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            cnt += 1
    return cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/hdd/robik/VQACP')
    parser.add_argument('--split', type=str, choices=['trainval', 'test2015'], default='trainval')
    args = parser.parse_args()

    output_split = 'test' if 'test' in args.split else args.split

    infile = args.data_root + f'/{args.split}_36/{args.split}_resnet101_faster_rcnn_genome_36.tsv'
    train_data_file = f'{args.data_root}/features/{output_split}.hdf5'
    h5_file = h5py.File(train_data_file, "w")
    image_ids_map = {'image_id_to_ix': {}, 'ix_to_image_id': {}}
    num_images = count(infile, FIELDNAMES)
    # num_images = 2/
    print(f"Num images {num_images}")
    img_features = h5_file.create_dataset(
        'image_features', (num_images, num_fixed_boxes, feature_length), 'f')
    bb = h5_file.create_dataset(
        'image_bb', (num_images, num_fixed_boxes, 4), 'f')
    spatial_img_features = h5_file.create_dataset(
        'spatial_features', (num_images, num_fixed_boxes, 6), 'f')

    counter = 0
    print("reading tsv...")
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(bytes(item['boxes'], 'utf-8')),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            # if image_id in train_imgids:
            #     train_imgids.remove(image_id)
            bb[counter, :, :] = bboxes
            img_features[counter, :, :] = np.frombuffer(
                base64.decodestring(bytes(item['features'], 'utf-8')),
                dtype=np.float32).reshape((item['num_boxes'], -1))
            spatial_img_features[counter, :, :] = spatial_features
            counter += 1
            image_ids_map['image_id_to_ix'][str(image_id)] = str(counter)
            image_ids_map['ix_to_image_id'][str(counter)] = str(image_id)
            print(f"Completed image index {counter}\r", end="")

    with open(os.path.join(args.data_root, 'features', f'{output_split}_ids_map.json'), 'w') as f:
        json.dump(image_ids_map, f)

    h5_file.close()
    print("done!")
