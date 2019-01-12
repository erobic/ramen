import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_set', required=True)
    parser.add_argument('--splits', required=True, nargs='+')

    args = parser.parse_args()
    args.data_root = args.root + "/" + args.data_set

    with open(args.data_root + '/image_ids/all_image_ids.json') as f:
        all_image_ids = json.load(f)

    ix_to_id, id_to_ix = {}, {}
    for ix, id in enumerate(all_image_ids):
        ix_to_id[ix] = id
        id_to_ix[id] = ix

    for split in args.splits:
        split_id_to_ix, split_ix_to_id = {}, {}
        with open(args.data_root + '/vqa2/{}_questions.json') as qf:
            qns = json.load(qf)['questions']
            for qn in qns:
                image_id = qn['image_id']
                image_ix = id_to_ix[image_id]
                split_id_to_ix[image_id] = image_ix
                split_ix_to_id[image_ix] = image_id

            with open(args.data_root + '/image_ids/{}_ids_map.json'.format(split), 'w') as f:
                json.dump({
                    'id_to_ix': split_id_to_ix,
                    'ix_to_id': split_ix_to_id
                })
