from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd
import random
import argparse
from utils import *
from sklearn.model_selection import train_test_split
random.seed(6250)

def train_test_valid_for_modeling(from_path, to_path, folder):

    to_folder = os.path.join(to_path, folder)
    from_folder = os.listdir(os.path.join(from_path, folder))

    if not os.path.exists(to_folder):
        os.mkdir(to_folder)

    data_ytrue_pair = []
    eps = 1e-6
    ids = list(filter(str.isdigit, from_folder))

    for (i, patient_id) in enumerate(ids):
        patient_data = os.path.join(from_path, folder, patient_id)
        data_files = os.listdir(patient_data)
        timeseries_files = list(filter(lambda x: x.find("timeseries") != -1, data_files))

        for timeseries in timeseries_files:
            with open(os.path.join(patient_data, timeseries)) as tsfile:
                episode_information = pd.read_csv(os.path.join(patient_data, timeseries.replace("_timeseries", "")))

                if episode_information.shape[0] == 0:
                    continue

                mortality = int(episode_information.iloc[0]["Mortality"])
                length_of_stay = episode_information.iloc[0]['Length of Stay'] * 24.0
                if pd.isnull(length_of_stay) or length_of_stay < 48 - eps:
                    print("The length of stay for patient{0} in {1} is missing or less than 48 hours".format(patient_id,
                                                                                                             timeseries))
                    continue

                ts_record = tsfile.readlines()
                header = ts_record[0]
                ts_record = ts_record[1:]
                hour = [float(line.split(',')[0]) for line in ts_record]

                ts_record = [line for (line, t) in zip(ts_record, hour) if -eps < t < 48 + eps]

                if len(ts_record) == 0:
                    print("Patient {0} has no events in ICU for {1}) ".format(patient_id, timeseries))
                    continue

                with open(os.path.join(to_folder, patient_id + "_" + timeseries), "w") as outfile:
                    outfile.write(header)
                    for line in ts_record:
                        outfile.write(line)

                data_ytrue_pair.append((patient_id + "_" + timeseries, mortality))


    print("\n", len(data_ytrue_pair))
    if folder == "train":
        random.shuffle(data_ytrue_pair)
        train_data_ytrue_pair, valid_data_ytrue_pair = train_test_split(data_ytrue_pair, test_size=0.50, random_state=42)

        with open(os.path.join(to_path, "train_listfile.csv"), "w") as listfile:
            listfile.write('stay,y_true\n')
            for (data, y) in train_data_ytrue_pair:
                listfile.write('{},{:d}\n'.format(data, y))


        with open(os.path.join(to_path,"val_listfile.csv"), "w") as listfile:
            listfile.write('stay,y_true\n')
            for (data, y) in valid_data_ytrue_pair:
                listfile.write('{},{:d}\n'.format(data, y))

    if folder == "test":
        test_data_ytrue_pair = sorted(data_ytrue_pair)
        with open(os.path.join(to_path, folder+"_listfile.csv"), "w") as listfile:
            listfile.write('stay,y_true\n')
            for (data, y) in test_data_ytrue_pair:
                listfile.write('{},{:d}\n'.format(data, y))


def main():

    parser = argparse.ArgumentParser(description='Generate train and test datasets for later modeling.')
    parser.add_argument('task', type=str, help='Can choose using test data or original data.')

    args, _ = parser.parse_known_args()

    if args.task == 'test':
        from_path = '../../data/test_data/data2'
    elif args.task == 'raw':
        from_path = '../../data/root'

    to_path = '../../data/preprocessed_data'

    try:
        os.makedirs(to_path)
    except:
        pass

    folders = ['train', 'test']
    for folder in folders:
        print('Start processing data for {}'.format(folder))
        train_test_valid_for_modeling(from_path, to_path, folder)
        print('We have processed data for {}'.format(folder))

if __name__ == '__main__':
    main()
