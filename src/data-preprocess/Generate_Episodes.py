from __future__ import absolute_import
from __future__ import print_function
import argparse
from utils import *
import os
import sys
import numpy as np
import pandas as pd
import re


g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}

diag_list = ['4019', '4280', '41401', '42731', '25000', '5849', '2724', '51881', '53081', '5990', '2720',
                    '2859', '2449', '486', '2762', '2851', '496', 'V5861', '99592', '311', '0389', '5859', '5070',
                    '40390', '3051', '412', 'V4581', '2761', '41071', '2875', '4240', 'V1582', 'V4582', 'V5867',
                    '4241', '40391', '78552', '5119', '42789', '32723', '49390', '9971', '2767', '2760', '2749',
                    '4168', '5180', '45829', '4589', '73300', '5845', '78039', '5856', '4271', '4254', '4111',
                    'V1251', '30000', '3572', '60000', '27800', '41400', '2768', '4439', '27651', 'V4501', '27652',
                    '99811', '431', '28521', '2930', '7907', 'E8798', '5789', '79902', 'V4986', 'V103', '42832',
                    'E8788', '00845', '5715', '99591', '07054', '42833', '4275', '49121', 'V1046', '2948', '70703',
                    '2809', '5712', '27801', '42732', '99812', '4139', '3004', '2639', '42822', '25060', 'V1254',
                    '42823', '28529', 'E8782', '30500', '78791', '78551', 'E8889', '78820', '34590', '2800', '99859',
                    'V667', 'E8497', '79092', '5723', '3485', '5601', '25040', '570', '71590', '2869', '2763', '5770',
                    'V5865', '99662', '28860', '36201', '56210']


def add_diag_data(diagnoses):
    global diag_list
    diagnoses['VALUE'] = 1
    labels = diagnoses[['ICUSTAY_ID', 'ICD9_CODE', 'VALUE']].drop_duplicates().pivot(index='ICUSTAY_ID',
                                                                                     columns='ICD9_CODE',
                                                                                     values='VALUE').fillna(0).astype(int)
    for diag in diag_list:
        if diag not in labels:
            labels[diag] = 0

    labels = labels[diag_list]
    return labels.rename(dict(zip(diag_list, ['Diagnosis ' + d for d in diag_list])), axis=1)

'''
we transform gender label using g_map
'''
def update_gender(gender):
    global g_map
    return {'Gender': gender.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}

'''
we transform ethnicity using e_map
'''

def update_ethnicity(ethnicity):
    global e_map

    def clean_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity = ethnicity.apply(clean_ethnicity)
    return {'Ethnicity': ethnicity.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

'''
create patient's basic information in one episodic
information contains:
Icustay	Ethnicity Gender Age Height	Weight	Length of Stay	Mortality and Value of each Diagnose
'''

def create_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.ICUSTAY_ID, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY}
    data['Height'] = np.nan
    data['Weight'] = np.nan
    data.update(update_gender(stays.GENDER))
    data.update(update_ethnicity(stays.ETHNICITY))
    data = pd.DataFrame(data).set_index('Icustay')
    col=['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']
    data = data[col]
    return data.merge(add_diag_data(diagnoses), left_index=True, right_index=True)


'''
This function creates a mapping relation between ITEMID and VARIABLE.
'''
def itemid_to_variable_map(file_name):
    var = 'LEVEL2'

    var_map = read_csv(file_name, index_col=None).fillna('').astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)

    var_map = var_map.ix[(var_map[var] != '') & (var_map.COUNT > 0)]
    var_map = var_map.ix[(var_map.STATUS == 'ready')]

    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[var, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')

    return var_map.rename({var: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1)

def itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)




'''
There are ambigous values in the column 'VALUE' in events for different 'VARIABLE',
for each 'VARIABLE', we use different methods to clean.
'''


def crr_clean(events):
    val = pd.Series(np.zeros(events.shape[0]), index=events.index)
    val[:] = np.nan
    events.VALUE = events.VALUE.astype(str)

    val.ix[(events.VALUE == 'Normal <3 secs') | (events.VALUE == 'Brisk')] = 0
    val.ix[(events.VALUE == 'Abnormal >3 secs') | (events.VALUE == 'Delayed')] = 1
    return val

def sbp_clean(events):
    val = events.VALUE.astype(str)
    idx = val.apply(lambda s: '/' in s)
    val.ix[idx] = val[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return val.astype(float)

def dbp_clean(events):
    val = events.VALUE.astype(str)
    idx = val.apply(lambda s: '/' in s)
    val.ix[idx] = val[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return val.astype(float)


def fio2_clean(events):
    val = events.VALUE.astype(float)

    is_str = np.array(map(lambda x: type(x) == str, list(events.VALUE)), dtype=np.bool)
    idx = events.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (val > 1.0)))

    val.ix[idx] = val[idx] / 100.
    return val

def o2satu_clean(events):
    val = events.VALUE
    idx_error = val.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    val.ix[idx_error] = np.nan

    val = val.astype(float)
    idx = (val <= 1)
    val.ix[idx] = val[idx] * 100.
    return val

def gluph_clean(events):
    val = events.VALUE
    idx = val.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    val.ix[idx] = np.nan
    return val.astype(float)

def temp_clean(events):
    val = events.VALUE
    val = val.astype(float)

    idx = events.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | events.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (val >= 79)
    val.ix[idx] = (val[idx] - 32) * 5. / 9
    return val

def weight_clean(events):
    val = events.VALUE.astype(float)

    idx = events.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | events.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    val.ix[idx] = val[idx] / 16.

    idx = idx | events.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | events.MIMIC_LABEL.apply(
        lambda s: 'lb' in s.lower())
    val.ix[idx] = val[idx] * 0.453592
    return val


def height_clean(events):
    val = events.VALUE.astype(float)
    idx = events.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | events.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    val.ix[idx] = np.round(val[idx] * 2.54)
    return val

def clean_event_var_value(events):
    crr_idx = (events.VARIABLE == 'Capillary refill rate')
    dbp_idx = (events.VARIABLE == 'Diastolic blood pressure')
    sbp_idx = (events.VARIABLE == 'Systolic blood pressure')
    fio2_idx = (events.VARIABLE == 'Fraction inspired oxygen')
    o2satu_idx = (events.VARIABLE == 'Oxygen saturation')
    glu_idx = (events.VARIABLE == 'Glucose')
    ph_idx = (events.VARIABLE == 'pH')
    temp_idx = (events.VARIABLE == 'Temperature')
    w_idx = (events.VARIABLE == 'Weight')
    h_idx = (events.VARIABLE == 'Height')

    idx_list = [crr_idx, dbp_idx, sbp_idx, fio2_idx, o2satu_idx, glu_idx, ph_idx, temp_idx, w_idx, h_idx]
    func_list = [crr_clean, dbp_clean, sbp_clean, fio2_clean, o2satu_clean, gluph_clean, gluph_clean, temp_clean, weight_clean, height_clean]

    var_num = len(idx_list)

    for i in range(var_num):
        var_idx = idx_list[i]
        clean_func = func_list[i]
        events.ix[var_idx, 'VALUE'] = clean_func(events.ix[var_idx])

    return events.ix[events.VALUE.notnull()]

def read_stays(subject_path):
    stays = read_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays

def read_diagnoses(subject_path):
    return read_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)

def read_events(subject_path):
    events = read_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    events = events.ix[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    return events


def extract_event_for_stay(events, stay_id, intime, outtime):
    idx = (events.ICUSTAY_ID == stay_id)
    if intime is not None and outtime is not None: idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events.ix[idx]
    del events['ICUSTAY_ID']
    return events


def compute_hours_from_intime_to_charttime(events, intime):
    events['HOURS'] = (events.CHARTTIME - intime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    del events['CHARTTIME']
    return events


def create_timeseries(events, variables):

    data_to_keep = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID']).drop_duplicates(keep='first').set_index('CHARTTIME')
    timeseries = events[['CHARTTIME', 'VARIABLE', 'VALUE']].sort_values(by=['CHARTTIME', 'VARIABLE', 'VALUE'], axis=0).drop_duplicates(subset=['CHARTTIME', 'VARIABLE'], keep='last')
    timeseries = timeseries.pivot(index='CHARTTIME', columns='VARIABLE', values='VALUE').merge(data_to_keep, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for var in variables:
        if var not in timeseries:
            timeseries[var] = np.nan
    return timeseries


def first_valid_value_in_timeseries(timeseries, var_name):
    if var_name in timeseries:
        idx = timeseries[var_name].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[var_name].iloc[loc]
    return np.nan




def main():
    parser = argparse.ArgumentParser(description='Generate time series data for each patient.')
    parser.add_argument('task', type=str, help='Can choose using test data or original data.')

    args, _ = parser.parse_known_args()

    if args.task == 'test':
        subjects_path = '../../data/test_root'
    elif args.task == 'raw':
        subjects_path = '../../data/root'

    map_file = '../../resources/itemid_to_variable_map.csv'

    var_map = itemid_to_variable_map(map_file)
    variables = var_map.VARIABLE.unique()

    sub_dir = os.listdir(subjects_path)
    subjects = list(filter(str.isdigit, sub_dir))

    for sub_num, subject in enumerate(subjects):

        subject_id = int(subject)
        subjects_file = os.path.join(subjects_path, subject)
        print('We are processing episode data for Subject {}: '.format(subject_id))
        print('Creating episodic data...')
        stays = read_stays(subjects_file)
        diagnoses = read_diagnoses(subjects_file)
        events = read_events(subjects_file)

        print('For Subject{0}, there are {1} stays, {2} diagnoses, {3} events...'.format(subject_id, stays.shape[0],
                                                                                         diagnoses.shape[0],
                                                                                         events.shape[0]))
        episodic_data = create_episodic_data(stays, diagnoses)

        print('Creating time series data for patient {}...'.format(subject_id))

        events = itemids_to_variables(events, var_map)
        events = clean_event_var_value(events)
        timeseries = create_timeseries(events, variables)

        for i in range(stays.shape[0]):
            stay_id = stays.ICUSTAY_ID.iloc[i]
            intime = stays.INTIME.iloc[i].to_datetime64()
            outtime = stays.OUTTIME.iloc[i].to_datetime64()

            episode = extract_event_for_stay(timeseries, stay_id, intime, outtime)
            if episode.shape[0] == 0:
                print("no data found for stay_id :{} in given time interval".format(stay_id))
                continue

            episode = compute_hours_from_intime_to_charttime(episode, intime).set_index('HOURS').sort_index(axis=0)
            episodic_data.Weight.ix[stay_id] = first_valid_value_in_timeseries(episode, 'Weight')
            episodic_data.Height.ix[stay_id] = first_valid_value_in_timeseries(episode, 'Height')

            episodic_data.ix[episodic_data.index == stay_id].to_csv(
                os.path.join(subjects_file, 'episode{}.csv'.format(i + 1)), index_label='Icustay')

            columns = list(episode.columns)
            episode = episode[sorted(columns)]
            episode.to_csv(os.path.join(subjects_file, 'episode{}_timeseries.csv'.format(i + 1)), index_label='Hours')
        print("We have processed {0} times series data for {1} subjects.".format(sub_num+1, len(subjects)))

        sys.stdout.write(' We have generated episodes for all subjects!\n')

if __name__ == '__main__':
    main()