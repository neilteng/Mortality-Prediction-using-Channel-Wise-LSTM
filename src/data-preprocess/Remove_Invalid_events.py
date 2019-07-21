from __future__ import absolute_import
from __future__ import print_function
import os
import pandas as pd
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Remove invalid events.')
    parser.add_argument('task', type=str, help='Can choose using test data or original data.')

    args, _ = parser.parse_known_args()

    if args.task == 'test':
        subject_path = '../../data/test_root/'
    elif args.task == 'raw':
        subject_path = '../../data/root/'


    sub_dir= os.listdir(subject_path)
    subjects = list(filter(str.isdigit, sub_dir))

    for (i, subject) in enumerate(subjects):
        print('We are removing invalid events for Subject:{}'.format(subject))
        stays_df = pd.read_csv(os.path.join(subject_path+subject, 'stays.csv'),index_col=False,dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        stays_df.columns = stays_df.columns.str.upper()
        events_df = pd.read_csv(os.path.join(subject_path, subject, 'events.csv'),
                                index_col=False,dtype={'HADM_ID': str, "ICUSTAY_ID": str})
        events_df.columns = events_df.columns.str.upper()
        begin_event_num = events_df.shape[0]
        events_df = events_df.dropna(subset=['HADM_ID'])
        tmp = events_df.merge(stays_df, left_on=['HADM_ID'], right_on=['HADM_ID'],how='left', suffixes=['', '_r'], indicator=True)
        tmp = tmp[tmp['_merge'] == 'both']
        tmp.loc[:, 'ICUSTAY_ID'] = tmp['ICUSTAY_ID'].fillna(tmp['ICUSTAY_ID_r'])
        tmp = tmp.dropna(subset=['ICUSTAY_ID'])
        tmp = tmp[(tmp['ICUSTAY_ID'] == tmp['ICUSTAY_ID_r'])]
        '''
        Export new events.csv
        '''
        cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
        new_events_df = tmp[cols]
        new_events_df.to_csv(os.path.join(subject_path, subject, 'events.csv'), index=False)
        after_event_num = new_events_df.shape[0]

        print("We have processed {0} subjects of {1}, we have removed {2} invalid events.".format(i+1, len(subjects), begin_event_num-after_event_num))


if __name__ == "__main__":
    main()
