import numpy as np
import os
import re

from deep_supervision_channel_wise_lstm_model import ds_channel_wise_lstms
import lstm_utils
import utils

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau


class ds_cw_lstm():
    def __init__(self, batch_norm=False, batch_size=128, data='', depth=2, hid_dim=16, dropout=0.3, epochs=100,
                target_repl_coef=0.5,
                 imputation='previous', l1=0, l2=0, load_state='', learning_rate=0.05, input_dim=76,
                 num_classes=1, normalizer_state=None, optimizer='adam', adam_beta_1=0.9, output_dir='',
                 rec_dropout=0.0, save_period=1, model_size_coef=4.0, timestep=1.0, verbose=2,
                 small_part=False):

        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.data_path = data
        self.depth = depth
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.epochs = epochs
        self.target_repl_coef = target_repl_coef
        self.imputation = imputation
        self.l1 = l1
        self.l2 = l2
        self.load_state = load_state
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalizer_state = normalizer_state
        self.optimizer = optimizer
        self.adam_beta_1 = adam_beta_1
        self.output_dir = output_dir
        self.recurrent_dropout = rec_dropout
        self.save_period = save_period
        self.model_size = model_size_coef
        self.timestep = timestep
        self.verbose = verbose
        self.small_part = small_part

        # check datapath
        if self.data_path == '':
            self.data_path = os.path.join(os.path.dirname(__file__), '../data/preprocessed_data/')
        if output_dir == '':
            self.output_dir = os.path.join(os.path.dirname(__file__), '../output/')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

    def train(self):
        if self.small_part:
            self.save_period = 2 ** 30
        # Build readers, discretizers, normalizers
        train_reader = lstm_utils.InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'),
                                                 listfile=os.path.join(self.data_path, 'train_listfile_lstm.csv'),
                                                 period_length=48.0)
        val_reader = lstm_utils.InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'),
                                               listfile=os.path.join(self.data_path, 'val_listfile.csv'),
                                               period_length=48.0)
        discretizer = lstm_utils.Discretizer(timestep=float(self.timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        normalizer = lstm_utils.Normalizer(fields=channels)
        if self.normalizer_state is None:
            self.normalizer_state = os.path.join(os.path.dirname(__file__), './resources/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(self.timestep,
                                                                                                          self.imputation) )
            self.normalizer_state = os.path.join(os.path.dirname(__file__), self.normalizer_state)
        normalizer.load_params(self.normalizer_state)
        # get channel
        ch_names = set()
        for ch in discretizer_header:
            if ch.find("mask->") != -1:
                continue
            pos = ch.find("->")
            if pos != -1:
                ch_names.add(ch[:pos])
            else:
                ch_names.add(ch)
        ch_names = sorted(list(ch_names))

        channels = []
        for ch in ch_names:
            idx = range(len(discretizer_header))
            idx = list(filter(lambda i: discretizer_header[i].find(ch) != -1, idx))
            channels.append(idx)

        args_dict = {'batch_norm': self.batch_norm, 'depth': self.depth, 'hid_dim': self.hid_dim,
                     'dropout': self.dropout,
                      'task': 'ihm',
                     'recurrent_dropout': self.recurrent_dropout,
                     'model_size': self.model_size, 'input_dim': self.input_dim, 'channels': channels,
                     'num_classes': self.num_classes}
        # Build the model
        print("==> using model {}".format('deep_supervision_channel_wise_lstm'))
        model = ds_channel_wise_lstms(**args_dict)
        model_name = "{}.n{}.szc{}{}{}{}.dep{}.bs{}{}{}.ts{}.trc{}".format('ds_channel_wise_lstms',
                                                                     self.hid_dim,
                                                                     self.model_size,
                                                                     ".bn" if self.batch_norm else "",
                                                                     ".d{}".format(
                                                                         self.dropout) if self.dropout > 0 else "",
                                                                     ".rd{}".format(
                                                                         self.recurrent_dropout) if self.recurrent_dropout > 0 else "",
                                                                     self.depth,
                                                                     self.batch_size,
                                                                     ".L1{}".format(self.l1) if self.l1 > 0 else "",
                                                                     ".L2{}".format(self.l2) if self.l2 > 0 else "",
                                                                     self.timestep,
                                                                    self.target_repl_coef)
        print("==> model.final_name:", model_name)
        # Compile the model
        print("==> compiling the model")
        optimizer_config = {'class_name': self.optimizer,
                            'config': {'lr': self.learning_rate,
                                       'beta_1': self.adam_beta_1}}
        model.compile(optimizer=optimizer_config,
                      loss=['binary_crossentropy','binary_crossentropy'],
                      loss_weights=[1 - self.target_repl_coef, self.target_repl_coef])
        model.summary()
        # Load model weights
        trained_epoch = 0
        if self.load_state != "":
            model.load_weights(self.load_state)
            trained_epoch = int(re.match(".*epoch([0-9]+).*", self.load_state).group(1))
        # Read data
        train_raw = lstm_utils.load_data(train_reader, discretizer, normalizer, self.small_part)
        val_raw = lstm_utils.load_data(val_reader, discretizer, normalizer, self.small_part)

        train_raw = lstm_utils.extend_labels(train_raw)
        val_raw = lstm_utils.extend_labels(val_raw)

        # Prepare training
        path = os.path.join(self.output_dir, model_name + '.epoch{epoch}.test{val_loss}.state')
        metrics_callback = lstm_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                            val_data=val_raw,
                                                            target_repl=True,
                                                            batch_size=self.batch_size,
                                                            verbose=self.verbose)
        # make sure save directory exists
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        saver = ModelCheckpoint(path, verbose=1, period=self.save_period, save_best_only=True, monitor='val_loss',
                                mode='min')

        keras_logs = os.path.join(self.output_dir, 'keras_logs')
        if not os.path.exists(keras_logs):
            os.makedirs(keras_logs)
        csv_logger = CSVLogger(os.path.join(keras_logs, model_name + '.csv'),
                               append=True, separator=';')

        earlyStopping = EarlyStopping(monitor='val_loss', patience=11, verbose=0, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4,
                                           mode='min')

        print("==> training")
        model.fit(x=train_raw[0],
                  y=train_raw[1],
                  validation_data=val_raw,
                  epochs=trained_epoch + self.epochs,
                  initial_epoch=trained_epoch,
                  callbacks=[metrics_callback, saver, csv_logger, earlyStopping, reduce_lr_loss],
                  shuffle=True,
                  verbose=self.verbose,
                  batch_size=self.batch_size)

    def test(self, load_state=''):
        self.load_state = load_state
        if self.small_part:
            self.save_period = 2 ** 30
        # Build readers, discretizers, normalizers
        train_reader = lstm_utils.InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'),
                                                 listfile=os.path.join(self.data_path, 'train_listfile_lstm.csv'),
                                                 period_length=48.0)
        val_reader = lstm_utils.InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'),
                                               listfile=os.path.join(self.data_path, 'val_listfile.csv'),
                                               period_length=48.0)
        discretizer = lstm_utils.Discretizer(timestep=float(self.timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        normalizer = lstm_utils.Normalizer(fields=channels)
        # normalizer_state = self.normalizer_state
        if self.normalizer_state is None:
            self.normalizer_state = os.path.join(os.path.dirname(__file__), './resources/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(self.timestep,
                                                                                                          self.imputation) )
            self.normalizer_state = os.path.join(os.path.dirname(__file__), self.normalizer_state)
        normalizer.load_params(self.normalizer_state)

        # get channel
        ch_names = set()
        for ch in discretizer_header:
            if ch.find("mask->") != -1:
                continue
            pos = ch.find("->")
            if pos != -1:
                ch_names.add(ch[:pos])
            else:
                ch_names.add(ch)
        ch_names = sorted(list(ch_names))

        channels = []
        for ch in ch_names:
            idx = range(len(discretizer_header))
            idx = list(filter(lambda i: discretizer_header[i].find(ch) != -1, idx))
            channels.append(idx)

        args_dict = {'batch_norm': self.batch_norm, 'depth': self.depth, 'hid_dim': self.hid_dim,
                     'dropout': self.dropout,
                      'task': 'ihm',
                     'recurrent_dropout': self.recurrent_dropout,
                     'model_size': self.model_size, 'input_dim': self.input_dim, 'channels': channels,
                     'num_classes': self.num_classes}
        # Build the model
        print("==> testing model {}".format('deep_supervision_channel_wise_lstm'))
        model = ds_channel_wise_lstms(**args_dict)

        # Compile the model
        print("==> compiling the model")
        optimizer_config = {'class_name': self.optimizer,
                            'config': {'lr': self.learning_rate,
                                       'beta_1': self.adam_beta_1}}

        model.compile(optimizer=optimizer_config,
                      loss=['binary_crossentropy','binary_crossentropy'],
                      loss_weights=[1 - self.target_repl_coef, self.target_repl_coef])
        model.summary()
        # Load model weights
        if self.load_state != "":
            model.load_weights(self.load_state)
        # Read data
        del train_reader
        del val_reader
        test_reader = lstm_utils.InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'test'),
                                                listfile=os.path.join(self.data_path, 'test_listfile.csv'),
                                                period_length=48.0)
        # utils
        ret = lstm_utils.load_data(test_reader, discretizer, normalizer, self.small_part,
                              return_names=True)

        data = ret["data"][0]
        labels = ret["data"][1]
        names = ret["names"]

        predictions = model.predict(data, batch_size=self.batch_size, verbose=1)
        print (predictions[0])
        with open('your_file.txt', 'w') as f:
            for item in predictions[0]:
                f.write("%s\n" % item)

        predictions = np.array(predictions[0])[:, 0]
        utils.print_metrics_binary(labels, predictions)

        path = os.path.join(self.output_dir, "test_predictions", os.path.basename(self.load_state)) + ".csv"
        # utils
        lstm_utils.save_results(names, predictions, labels, path)


if __name__ == "__main__":
    ds_cw_lstm=ds_cw_lstm(batch_size=512, depth=1, hid_dim=16, dropout=0.3, epochs=100, target_repl_coef=0.5,
                    learning_rate=0.05,rec_dropout=0.0,save_period=1, model_size_coef=4.0, timestep=1.0,
              data=os.path.join(os.path.dirname(__file__),'../data/preprocessed_data/'),
              output_dir='./output/')
    # train a network please use following lines
    ds_cw_lstm.train()

    # # test a network please use following lines
    ds_cw_lstm.test(load_state=os.path.join(os.path.dirname(__file__),'./resources/ds_channel_wise_lstm_best.state'))
