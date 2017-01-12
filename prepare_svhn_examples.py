import h5py
import numpy as np
import pickle
import sys
import argparse
import Utils


def unpack_int(hdf5_file, bbox, field):
    dataset_or_ref = bbox.get(field)

    if type(dataset_or_ref) is h5py.Dataset:
        dataset = dataset_or_ref
        data = dataset[0][0]
        if type(data) is h5py.Reference:
            return int(np.array(hdf5_file.get(data))[0][0])
        else:
            return int(dataset[0])
    else:
        raise IOError


def deref_array(hdf5_file, array):
    result = []
    for ref in array:
        ref = ref[0]
        digit = int(np.array(hdf5_file.get(ref))[0][0])
        if digit == 10:
            digit = 0
        result.append(digit)
    return np.array(result)


def label_from_dataset(hdf5_file, dataset):
    array = np.array(dataset)
    if array.dtype == 'float64':
        return np.array([int(array[0][0])])
    else:
        return deref_array(hdf5_file, array)


def get_bbox_data_single(dataset, key):
    return dataset.get(key)[0][0]


def get_bbox_data(hdf5_file, dataset, key, index):
    ref = dataset.get(key)[index][0]
    return np.array(hdf5_file.get(ref))[0][0]


def unpack_bboxes(hdf5_file, dataset):
    array = np.array(dataset)
    hdata = dataset.get('height')
    num_bboxes = hdata.shape[0]
    bboxes = []

    if num_bboxes == 1:
        h = get_bbox_data_single(dataset, 'height')
        w = get_bbox_data_single(dataset, 'width')
        x = get_bbox_data_single(dataset, 'left')
        y = get_bbox_data_single(dataset, 'top')
        l = get_bbox_data_single(dataset, 'label')
        l -= (10 if l == 10 else 0)
        bboxes.append([int(x), int(y), int(w), int(h), int(l)])
    else:
        for i in range(0,num_bboxes):
            h = get_bbox_data(hdf5_file, dataset, 'height', i)
            w = get_bbox_data(hdf5_file, dataset, 'width', i)
            x = get_bbox_data(hdf5_file, dataset, 'left', i)
            y = get_bbox_data(hdf5_file, dataset, 'top', i)
            l = get_bbox_data(hdf5_file, dataset, 'label', i)
            l -= (10 if l == 10 else 0)
            bboxes.append([int(x), int(y), int(w), int(h), int(l)])

    return bboxes

def unpack_label(hdf5_file, bbox):
    dataset_or_ref = bbox.get('label')
    dataset = None
    if type(dataset_or_ref) is h5py.Dataset:
        dataset = dataset_or_ref
    else:
        ref = dataset_or_ref
        dataset = hdf5_file[ref]

    return label_from_dataset(hdf5_file, dataset)


def join_labels(bbox_array):
    label = ""
    for bbox in bbox_array:
        label += str(int(bbox[4]))

    return label


parser = argparse.ArgumentParser()
parser.add_argument('dir', action="store")
args = parser.parse_args()
f = h5py.File(args.dir + '/digitStruct.mat', 'r')

refs = f.get('#refs#')
name_refs = f['/digitStruct/name']
bbox_refs = f['/digitStruct/bbox']
num_data = len(name_refs)
labels = {}
bboxes = {}

print "Reading digitStruct.mat..."
for i in range(0,num_data):
    Utils.progress_bar(i+1,num_data)
    name_ref = name_refs[i][0]
    bbox_ref = bbox_refs[i][0]
    file_name_array = np.array(f.get(name_ref))
    file_name = "".join([chr(c) for c in file_name_array])
    bbox_data = f.get(bbox_ref)
    bbox_array = unpack_bboxes(bbox_data, bbox_data)
    id = file_name.replace('.png', '')
    labels[id] = join_labels(bbox_array)
    bboxes[id] = bbox_array

print
file = open(args.dir + '/labels.pickle', 'wb')
print ("Writing labels.pickle ...")
pickle.dump(labels, file, -1)

file = open(args.dir + '/bboxes.pickle', 'wb')
print ("Writing bboxes.pickle ...")
pickle.dump(bboxes, file, -1)
