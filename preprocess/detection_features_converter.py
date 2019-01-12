"""
Reads in a tsv file with pre-trained bottom up attention features_path and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features_path
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
# import cPickle
# import _pickle as cPickle
import six;
from six.moves import cPickle
import numpy as np
import utils
import argparse
import json
import io

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features_path']
# infile = 'data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
# train_data_file = 'train36.hdf5'
# val_data_file = 'data/val36.hdf5'
# train_indices_file = 'data/train36_imgid2idx.pkl'
# val_indices_file = 'data/val36_imgid2idx.pkl'
# train_ids_file = 'data/train_ids.pkl'
# val_ids_file = 'data/val_ids.pkl'

feature_length = 2048


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--num_fixed_boxes', type=int, required=True)
    parser.add_argument('--tsv_created_in_python3', type=int, default=1)
    parser.add_argument('--is_tsv_binary', type=int, default=1)
    args = parser.parse_args()
    args.tsv_created_in_python3 = bool(args.tsv_created_in_python3)
    args.is_tsv_binary = bool(args.is_tsv_binary)
    return args


if __name__ == '__main__':
    args = parse_args()
    data_file = args.dataroot + '/bottom-up-attention/{}.hdf5'.format(args.split)
    h = h5py.File(data_file, "w")

    with open(args.dataroot + '/image_ids/{}_image_ids.json'.format(args.split), 'r') as f:
        img_ids = json.load(f)['image_ids']

    indices = {}
    indices_file = args.dataroot + '/bottom-up-attention/{}_imgid2idx.pkl'.format(args.split)
    ids_map_file = args.dataroot + '/bottom-up-attention/{}_ids_map.json'.format(args.split)


    img_features = h.create_dataset(
        'image_features', (len(img_ids), args.num_fixed_boxes, feature_length), 'f')
    img_bb = h.create_dataset(
        'image_bb', (len(img_ids), args.num_fixed_boxes, 4), 'f')
    spatial_img_features = h.create_dataset(
        'spatial_features', (len(img_ids), args.num_fixed_boxes, 6), 'f')

    counter = 0

    # print("reading tsv...")
    # with io.TextIOWrapper(fileobj, encoding='utf-8') as text_file:
    #     reader = csv.DictReader(text_file, delimiter=',')
    #
    #     for row in reader:
    #         if row and 'caption' in row.keys():
    #             yield (reader.line_num, '', row['caption'], '')

    if args.is_tsv_binary:
        open_mode = 'rb+'
    else:
        open_mode = 'r'
    with io.TextIOWrapper(open(args.infile, open_mode), encoding='utf-8') as tsv_in_file:

        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)

        # fileobj = open(args.infile, 'rb')
        # with io.TextIOWrapper(fileobj, encoding='utf-8') as text_file:
        # #with io.TextIOWrapper(args.infile, encoding='utf-8') as text_file:
        #     reader = csv.DictReader(text_file, delimiter='\t', fieldnames=FIELDNAMES)
        # with open(args.infile, 'rb') as tsv_file:
        #     reader = csv.DictReader(tsv_file, delimiter='\t', fieldnames=FIELDNAMES)

        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])  # 642
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            boxes = item['boxes']
            features = item['features_path']

            if args.tsv_created_in_python3:
                boxes = boxes[2:-1]
                features = features[2:-1]
                boxes = boxes.encode('utf-8')
                features = features.encode('utf-8')
            bboxes = np.frombuffer(
                base64.decodestring(boxes),
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

            ii = 1
            if image_id in img_ids:
                ii += 1
                img_ids.remove(image_id)
                indices[image_id] = counter
                img_bb[counter, :, :] = bboxes

                img_features[counter, :, :] = np.frombuffer(
                    base64.decodestring(features),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                spatial_img_features[counter, :, :] = spatial_features
                counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(img_ids) != 0:
        print('Warning: image_ids is not empty. len(img_ids) = ', len(img_ids))

    cPickle.dump(indices, open(indices_file, 'wb'))

    image_id_to_ix = {}
    image_ix_to_id = {}

    for id in indices.keys():
        ix = indices[id]
        image_id_to_ix[int(id)] = int(ix)
        image_ix_to_id[int(ix)] = int(id)

    with open(ids_map_file, 'w') as f:
        json.dump({
            'image_id_to_ix': image_id_to_ix,
            'image_ix_to_id': image_ix_to_id
        }, f)

    h.close()
    print("done!")
