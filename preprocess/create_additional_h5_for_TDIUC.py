import h5py
import os
import json


def get_all_image_ids(split):
    img_ids = {}
    qns = json.load(open(os.path.join(f'/hdd/robik/TDIUC/questions/{split}_questions.json')))['questions']
    for q in qns:
        img_ids[q['image_id']] = 1
    return img_ids


def get_extra_image_ids(split, img_ids):
    img_id_to_ix = json.load(open(os.path.join(f'/hdd/robik/TDIUC/features/{split}_ids_map.json')))['image_id_to_ix']
    extra_img_ids = {}
    for img_id in img_ids:
        if str(img_id) not in img_id_to_ix:
            extra_img_ids[img_id] = 1
    return extra_img_ids


def create_extra_h5(split, extra_img_ids):
    target_dir = '/media/robik/TDIUC'
    extra_ids_map = {
        'image_id_to_ix': {},
        'ix_to_image_id': {}
    }
    with h5py.File(f'{target_dir}/features/{split}_extra.hdf5', 'w') as h:
        all_features = h5py.File(os.path.join('/media/robik/NATURAL_VQA/bottom-up-attention/all.hdf5'))
        image_features = all_features.get('image_features')
        spatial_features = all_features.get('spatial_features')
        all_ids = json.load(open(os.path.join('/media/robik/NATURAL_VQA/bottom-up-attention/all_ids_map.json')))
        new_ix = 0
        for eid_ix, eid in enumerate(extra_img_ids):
            all_ix = all_ids['image_id_to_ix'][str(eid)]
            img_feat = image_features[all_ix]
            spatial_feat = spatial_features[all_ix]
            if eid_ix == 0:
                h.create_dataset('image_features', shape=(len(extra_img_ids), img_feat.shape[0], img_feat.shape[1]))
                h.create_dataset('spatial_features',
                                 shape=(len(extra_img_ids), spatial_feat.shape[0], spatial_feat.shape[1]))
            h['image_features'][new_ix] = img_feat
            h['spatial_features'][new_ix] = spatial_feat
            extra_ids_map['image_id_to_ix'][str(eid)] = str(new_ix)
            extra_ids_map['ix_to_image_id'][str(new_ix)] = str(eid)
            new_ix += 1
            print(f"new_ix {new_ix}")
    json.dump(extra_ids_map, open(f'{target_dir}/features/{split}_ids_map_extra.json', 'w'))


if __name__ == "__main__":
    train_img_ids = get_all_image_ids('train')
    val_img_ids = get_all_image_ids('val')
    extra_train_img_ids = get_extra_image_ids('train', train_img_ids)
    extra_val_img_ids = get_extra_image_ids('val', val_img_ids)
    print(f"# Extra train img ids {len(extra_train_img_ids)} # Extra val img ids {len(extra_val_img_ids)}")
    create_extra_h5('train', extra_train_img_ids)
    create_extra_h5('val', extra_val_img_ids)
