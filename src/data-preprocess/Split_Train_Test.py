from __future__ import absolute_import
from __future__ import print_function
import os
import shutil
import argparse
from utils import *

from sklearn.model_selection import train_test_split

def split_train_test(folders):
    train_id, test_id = train_test_split(folders, test_size=0.20, random_state=42)
    return train_id, test_id

def create_train_test_folder(subject_path, patients, foldername):
    print("Strat creating {} set".format(foldername))
    if not os.path.exists(os.path.join(subject_path, foldername)):
        os.mkdir(os.path.join(subject_path, foldername))
    for patient in patients:
        from_folder = os.path.join(subject_path, patient)
        to_folder = os.path.join(subject_path, foldername, patient)
        shutil.move(from_folder, to_folder)
    print("Finished!")



def main():
    parser = argparse.ArgumentParser(description='Split patients into training set or test set.')
    parser.add_argument('task', type=str, help='Can choose using test data or original data.')

    args, _ = parser.parse_known_args()

    if args.task == 'test':
        subjects_path = '../../data/test_data/data2'
    elif args.task == 'raw':
        subjects_path = '../../data/root'




    folders = os.listdir(subjects_path)
    folders = list((filter(str.isdigit, folders)))
    train_id, test_id = split_train_test(folders)

    create_train_test_folder(subjects_path, train_id, "train")
    create_train_test_folder(subjects_path, test_id, "test")


if __name__ == '__main__':
    main()
