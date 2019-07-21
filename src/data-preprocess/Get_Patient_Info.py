from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import sys
import csv
import argparse
import pandas as pd
from utils import *


'''
Step 1:

Read csv: PATIENTS, ADMISSIONS, ICUSTAYS.
'''

def read_patients(data_path):
    patients = csv_to_df(os.path.join(data_path, 'PATIENTS.csv'))
    patients = patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    patients.DOB = pd.to_datetime(patients.DOB)
    patients.DOD = pd.to_datetime(patients.DOD)
    return patients


def read_admissions(data_path):
    admissions = csv_to_df(os.path.join(data_path, 'ADMISSIONS.csv'))
    admissions = admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admissions.ADMITTIME = pd.to_datetime(admissions.ADMITTIME)
    admissions.DISCHTIME = pd.to_datetime(admissions.DISCHTIME)
    admissions.DEATHTIME = pd.to_datetime(admissions.DEATHTIME)
    return admissions


def read_icustays(data_path):
    icustays = csv_to_df(os.path.join(data_path, 'ICUSTAYS.csv'))
    icustays.INTIME = pd.to_datetime(icustays.INTIME)
    icustays.OUTTIME = pd.to_datetime(icustays.OUTTIME)
    return icustays


def read_codes_diagnoses(data_path):
    codes = csv_to_df(os.path.join(data_path, 'D_ICD_DIAGNOSES.csv'))
    diagnoses = csv_to_df(os.path.join(data_path, 'DIAGNOSES_ICD.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = pd.merge(diagnoses, codes, on='ICD9_CODE', how='inner')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)

    return diagnoses



'''

we remove patients who were transfered
only patients who stay in the same location and care unit during ICU stays are pertained.

'''
def exclude_transfers(icustays):
    cond1 = (icustays.FIRST_CAREUNIT == icustays.LAST_CAREUNIT)
    cond2 = (icustays.FIRST_WARDID == icustays.LAST_WARDID)
    icustays = icustays.ix[cond1 & cond2]
    cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']
    icustays= icustays[cols]
    return icustays


def exclude_multi_stays(icustays):
    single_stay = icustays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    single_stay = single_stay.ix[(single_stay.ICUSTAY_ID == 1)][['HADM_ID']]
    icustays = pd.merge(icustays,single_stay, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return icustays

'''

inner join icu stays table and patients table on key 'SUBJECT_ID'.

'''
def merge_stays_patients(icustays, patients):
    return pd.merge(icustays, patients, how='inner', left_on='SUBJECT_ID', right_on='SUBJECT_ID')

'''

inner join icu stays table and admissions table on key 'SUBJECT_ID' and 'HADM_ID'.

'''
def merge_stays_admissions(icustays, admissions):
    on_key = ['SUBJECT_ID', 'HADM_ID']
    return pd.merge(icustays, admissions, how='inner', left_on=on_key, right_on=on_key)

'''
compute patients' age when he/ she enter icu.
'''
def compute_age(icustays):
    icustays['AGE'] = (icustays.INTIME - icustays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60 / 24 / 365
    icustays.ix[icustays.AGE < 0, 'AGE'] = 90
    return icustays

'''
remove patients whose age < 18.
'''
def remove_less18(icustays):
    icustays = icustays.ix[(icustays.AGE >= 18) & (icustays.AGE <= np.inf)]
    return icustays

'''
get patients in unit mortality.
'''
def inunit_mortality(icustays):
    inunit_mortality = icustays.DOD.notnull() & ((icustays.INTIME <= icustays.DOD) & (icustays.OUTTIME >= icustays.DOD))
    inunit_mortality = inunit_mortality | (
                icustays.DEATHTIME.notnull() & ((icustays.INTIME <= icustays.DEATHTIME) & (icustays.OUTTIME >= icustays.DEATHTIME)))
    icustays['MORTALITY_INUNIT'] = inunit_mortality.astype(int)
    return icustays

'''
get patients in hospital mortality.
'''
def inhospital_mortality(icustays):
    inhospital_mortality = icustays.DOD.notnull() & ((icustays.ADMITTIME <= icustays.DOD) & (icustays.DISCHTIME >= icustays.DOD))
    inhospital_mortality = inhospital_mortality | (icustays.DEATHTIME.notnull() & (
                (icustays.ADMITTIME <= icustays.DEATHTIME) & (icustays.DISCHTIME >= icustays.DEATHTIME)))
    icustays['MORTALITY'] = inhospital_mortality.astype(int)
    icustays['MORTALITY_INHOSPITAL'] = inhospital_mortality.astype(int)
    return icustays


def merge_diag_stays(diagnoses, icustays):
    icustays = icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates()
    on_key = ['SUBJECT_ID', 'HADM_ID']
    return pd.merge(diagnoses, icustays, how='inner',left_on=on_key, right_on=on_key)

'''
count number of icd9 codes and keep those counts > 0.
'''
def filter_icd9_codes(diagnoses):
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes['COUNT'] = codes['COUNT'].fillna(0).astype(int)
    codes = codes.ix[codes['COUNT'] > 0]
    return codes


'''
For each subject, we create a file to with subject ID as the name,
and store each patient's icu stay information in his/ her file.
'''
def create_stays_for_each_subject(icustays, subjects, output_path):

    for i, id in enumerate(subjects):
        id_file = os.path.join(output_path, str(id))
        try:
            os.makedirs(id_file)
        except:
            pass
        sys.stdout.write('\r Writing stays data for SUBJECT {}...'.format(id))
        stays = icustays.ix[icustays.SUBJECT_ID == id].sort_values(by='INTIME')
        stays.to_csv(os.path.join(id_file, 'stays.csv'),index=False)

'''
For each subject, we create a file to with subject ID as the name,
and store each patient's diagnoses information in his/ her file.
'''


def create_diag_for_each_subject(diagnoses, subjects, output_path):
    for _, id in enumerate(subjects):
        id_file = os.path.join(output_path, str(id))
        sys.stdout.write('\r Writing diagnoses data for SUBJECT {}...'.format(id))
        diag = diagnoses.ix[diagnoses.SUBJECT_ID == id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])
        diag.to_csv(os.path.join(id_file, 'diagnoses.csv'), index=False)

def read_events_table_by_row(data_path, table, task):
    if task =='raw':
        nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    elif task =='test':
        nb_rows = {'chartevents': 11473, 'labevents': 2476, 'outputevents': 0}
    reader = csv.DictReader(open(os.path.join(data_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]

def create_event_for_each_subject(data_path, table, output_path, subjects_to_keep, task):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.last_write_no = 0
            self.last_write_nb_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        data_stats.last_write_no += 1
        data_stats.last_write_nb_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    for row, row_no, nb_rows in read_events_table_by_row(data_path, table,task):
        if data_stats.last_write_no != '':
            sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                                 '({3}) {4} rows for subject {5}'.format(table, row_no, nb_rows,
                                                                         data_stats.last_write_no,
                                                                         data_stats.last_write_nb_rows,
                                                                         data_stats.last_write_subject_id))
        else:
            sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, row_no, nb_rows))

        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue


        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()


def main():
    parser = argparse.ArgumentParser(description='Generate stays, diagnoses, events information for each patient.')
    parser.add_argument('task', type=str, help='Can choose using test data or original data.')

    args, _ = parser.parse_known_args()

    if args.task == 'test':
        data_path = '../../data/test_data/data1'
        output_path = '../../data/test_root/'
        task = 'test'
    elif args.task == 'raw':
        data_path = '../../data/'
        output_path = '....//data/root/'
        task = 'raw'

    try:
        os.makedirs(output_path)
    except:
        pass

    print('Start loading patients data, admissions data, ICU stays data...')
    patients = read_patients(data_path)
    admissions = read_admissions(data_path)
    icustays = read_icustays(data_path)
    diagnoses = read_codes_diagnoses(data_path)
    print('Done!')
    print('The number of unique SUBJECT_ID (patient) is',icustays.SUBJECT_ID.unique().shape[0],
          'The number of unique HADM_ID (admissions) is', icustays.HADM_ID.unique().shape[0],
          'The number of unique ICUSTAY_ID (ICU satys) is', icustays.ICUSTAY_ID.unique().shape[0])

    '''
    Step1 :
    
    Remove records with ICU transfers.
    
    We only keep records which didn't change their icu unit or ward,
    to avoid the effect that related to hospital admissions.
    '''

    print('Start removing icu stays with icu transfers...')
    icustays = exclude_transfers(icustays)
    print('After step1, we remove records with ICU transfers!')
    print('The number of unique SUBJECT_ID (patient) is',icustays.SUBJECT_ID.unique().shape[0],
          'The number of unique HADM_ID (admissions) is', icustays.HADM_ID.unique().shape[0],
          'The number of unique ICUSTAY_ID (ICU satys) is', icustays.ICUSTAY_ID.unique().shape[0])

    print('Start merging icu stays table and admissions table by SUBJECT_ID and HADM_ID')
    icustays = merge_stays_admissions(icustays, admissions)
    icustays = merge_stays_patients(icustays, patients)

    print('Merging Done!')


    '''
    Step 2:
    
    Remove admissions with multiple icu stays records,
    since we hope one admission maps to one icu stay.

    '''

    print('Start removing records with multiple icu stays...')
    icustays = exclude_multi_stays(icustays)
    print('After Step 2, we remove records with multiple ICU stays, now one admission maps to one icu stays')
    print('The number of unique SUBJECT_ID (patient) is',icustays.SUBJECT_ID.unique().shape[0],
          'The number of unique HADM_ID (admissions) is', icustays.HADM_ID.unique().shape[0],
          'The number of unique ICUSTAY_ID (ICU satys) is', icustays.ICUSTAY_ID.unique().shape[0])


    '''
    Step 3:
    
    Remove patients whose age < 18 and add mortality information for each patient.
    '''

    icustays = compute_age(icustays)
    icustays = remove_less18(icustays)

    icustays = inunit_mortality(icustays)
    icustays = inhospital_mortality(icustays)

    print('After Step 3, we remove patients whose age < 18 and add mortality information for each patient.')
    print('The number of unique SUBJECT_ID (patient) is',icustays.SUBJECT_ID.unique().shape[0],
          'The number of unique HADM_ID (admissions) is', icustays.HADM_ID.unique().shape[0],
          'The number of unique ICUSTAY_ID (ICU satys) is', icustays.ICUSTAY_ID.unique().shape[0])




    '''
    Step 4. For each patient,
    create a file to store the information about stays, diagnoses, and events.
    '''
    subjects = icustays.SUBJECT_ID.unique()
    diagnoses = merge_diag_stays(diagnoses, icustays)


    '''
    stays:
    '''
    print('Start create icu stays file for each patient...')
    create_stays_for_each_subject(icustays, subjects, output_path)
    print('For each patient, we have create a file to store the stays information!')


    '''
    diags:
    '''

    print('Start create diagnoses file for each patient...')
    create_diag_for_each_subject(diagnoses,subjects,output_path)
    print('For each patient, we have create a file to store the diagnoses information!')

    '''
    events:
    '''
    print('Start create events file for each patient...')
    event_tables=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS']
    for table in event_tables:
        create_event_for_each_subject(data_path, table, output_path,subjects,task)
    print('For each patient, we have create a file to store the event information!')

if __name__ == '__main__':
    main()