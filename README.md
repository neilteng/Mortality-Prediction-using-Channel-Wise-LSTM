# mortality_prediction
member: Liyan Fang, Shimeng Jiang, Xujian Liang, Zhiyong Deng

## Data preprocessing

> Since the raw datasets from mimic3 database are too large, we have created test dataset in the directory: `data/test_data/`, which contains `data1.zip` and `data2.zip` that are sub datasets for testing codes for data preprocess.

The following instructions for this part are based on test data, if you want to run on original mimic3 datasets, just change test to raw in the command line, and make sure the raw csv tables are put under `data/`.

To successfully test the code, please unzip the `data1.zip` and `data2.zip` firstly, then access the folder `src/data-preprocess` in the terminal to continue the following steps. 

### Note:
If you want to run the code on raw datasets from mimic3 database, you need to put 'ADMISSION.csv', 'PATIENTS.csv', 'ICUSTAYS.csv', 'D_ICD_DIAGNOSES.csv', 'DIAGNOSES_ICD', 'CHARTEVENTS.csv', 'LABEVENTS.csv', 'OUTPUTEVENTS.csv' from original mimic3 database into the directory `data/`. It will take hours to process raw data. No matter using test data or raw data, preprocessed data is in the directory: `data/preprocessed_data/`.

1. For each 'SUBJECT_ID' (patient), we want to integrate his/ her information during ICU stays from different tables. After running this code, we extract information for each patient during his/ her icustays in the directory `data/test_root/`, each folder is the name of patient's 'SUBJECT_ID', in which we can see stays.csv, events.csv, diagnoses.csv. 

<pre><code>python Get_Patient_Info.py test</code></pre>

2. We hope that one HADM_ID maps to one ICUSTAY_ID and there are invalid events that we need to remove. In this part, we will remove records that satisfies:

- HADM_ID is empty in events.csv.
- HADM_ID does not appear in stays.csv, since there is no way to know the mortality of that stay.

If ICUSTAY_ID is empty in stays.csv, we use the HADM_ID of that record to recorver this empty ICUSTAY_ID. We drop those records if we can not recover the ICUSTAY_ID.

Finally we check if each unique HADM_ID maps to a unique ICUSTAY_ID, and drop those don't satisfy that.

<pre><code>python Remove_Invalid_events.py test</code></pre>

3. Each icu stay is an episode, in this step, we generate a time series of events for each episode
<pre><code>python Generate_Episodes.py test</code></pre>

4. In this step, we split all patients data into training set and test set(80% : 20%).
<pre><code>python Split_Train_Test.py test</code></pre>

5.Finally, we generate datasets for modeling use. We will generate `train_listfile.csv`, `test_listfile.csv`, `val_listfile.csv` in `data/preprocessed_data/`, these listfile.csv contain (stay data file, true mortality) pairs. `test` and `train` contain corresponding stay data files.

<pre><code>python Dataset_For_Model.py test</code></pre>

### Note:
The reason we use two sub datasets for testing data proprocess codes is that, in the first step for integrating patients' information during ICU stays from different tables is very time-consuming, the raw `CHARTEVENTS.csv` contains 330712484 rows record, we have to scan the whole table to extract certain patient's information. Hence, for convenience, we only use 3 unique patient in `data1` for testing 1 to 4 step above. The processed data after 1 to 4 step are located in `test_data/test_root/`.

For testing step 5 and 6, which is to split datasets into train and test sets, we need to use a larger dataset. Hence for step 5 and 6, we use `data2` which contains 195 unique patients. They have the same type of structure as data in `test_data/test_root/`.

## Preprocessed data for testing models
We provided a preprocessed_data.zip file in the directory `data/`. Unzip the zip file directly in this directory so that models could use the data and run.

## Machine learning models
To run Logistic Regression, KNN, SVM, Decision Tree, Random Forest and Adaboost on preprocessed data, please run the following command in terminal in the following directory: `src/` and make sure preprocessed data is in the directory: `data/preprocessed_data/`. The results will be printed in the terminal. The Anaconda base environment is enough for these models.

    python ml_models.py --model LR

LR can be substitute with following parameters: LR, KNN, SVM, DT, RF or ADA.

Best parameters are already integrated as part of the code. If you want to change the parameters of models, please directly change them in the code.

## LSTM Model
base_lstm.py is wrapper class for trainning, testing base_lstm model.

base_LSTM_model.py construct the base_LSTM model.

lstm_utils.py and utils.py provide utils for data loader, metrics and evaluation.

To run LSTM, LSTM for DS on preprocess data, please run the following command in terminal in the following directory: `src/` and make sure preprocessed data is in the directory: `data/`. The results will be printed in the terminal. The Anaconda base environment is enough for these models.
 
    python base_lstm.py --target_repl_coef 0.0/0.5(>0.0 means start deep_supervision)
   
Best parameters are already integrated as part of the code. If you want to change the parameters, please type -h to get all parameters

Test a network please use following api, set parameter 'load_state' to 

	python base_lstm.py --load_state ./output/keras_states/k_lstm.n16.d0.3.rd0.3.dep2.bs8.ts1.0.trc0.5.epoch10.test0.2959162076295791.state

The best model for DS one and normal one has been stored in ./src/output named as `base_lstm_.h5` and `base_lstm_DS.h5`
## Channel-wise LSTM MODEL
channel_wise_LSTM.py is wrapper class for trainning, testing channel_wise_LSTM model.

channel_wise_lstms_model.py construct the channel_wise_LSTM model.

lstm_utils.py provide utils for data loader, metrics and evaluation.

(P.S.: the first  few time you call the model may rise errorlike:'OSError: [Errno 5] Input/output error: '../data/preprocessed_data/train/15656_episode1_timeseries.csv'"]', you can just try to call the model several time.)

#to use channel_wise_LSTM model please refer to code:

For train and test channel_wise_LSTM model, use the following common line.

    python channel_wise_LSTM.py

To use channel_wise_LSTM in other python script, one can initial with following api

    cw_lstm = cw_lstm(batch_norm=False, batch_size=512, depth=3, hid_dim=32, dropout=0.3, epochs=100,
                      learning_rate=0.05, rec_dropout=0.0, save_period=1, model_size_coef=4.0, timestep=1.0,
                      data='../data/preprocessed_data/',
                      output_dir='./output/')
              
And train a network please use following api

    cw_lstm.train()

Test a network please use following api, set parameter 'load_state' to 
 
    cw_lstm.test(load_state='./resources/channel_wise_lstm_best.state')


## Deep Supervision Channel-wise LSTM MODEL
deep_supervision_channel_wise_lstm.py.py is wrapper class for trainning, testing channel_wise_LSTM model.

deep_supervision_channel_wise_lstm_model.py construct the channel_wise_LSTM model.

lstm_utils.py provide utils for data loader, metrics and evaluation.

(P.S.: the first  few time you call the model may rise errorlike:'OSError: [Errno 5] Input/output error: '../data/preprocessed_data/train/15656_episode1_timeseries.csv'"]', you can just try to call the model several time.)

#to use channel_wise_LSTM model please refer to code:

For train and test channel_wise_LSTM model, use the following common line.

    python deep_supervision_channel_wise_lstm.py

To use channel_wise_LSTM in other python script, one can initial with following api

    ds_cw_lstm=ds_cw_lstm(batch_size=512, depth=2, hid_dim=16, dropout=0.3, epochs=100, target_repl_coef=0.5,
                    learning_rate=0.05,rec_dropout=0.0,save_period=1, model_size_coef=4.0, timestep=1.0,
              data=os.path.join(os.path.dirname(__file__),'../data/preprocessed_data/'),
              output_dir='./output/')
              
And train a network please use following api

    cw_lstm.train()

Test a network please use following api, set parameter 'load_state' to 
 
    ds_cw_lstm.test(load_state='./resources/ds_channel_wise_lstm_best.state')
    
**Load and test saved best model, we need to initial the class and call test on the best saved model stated like above.**
'./resources/channel_wise_lstm_best.state' is the best model for channel_wise_lstm.
'./resources/ds_channel_wise_lstm_best.state' is the best model for ds_channel_wise_lstm.
    

