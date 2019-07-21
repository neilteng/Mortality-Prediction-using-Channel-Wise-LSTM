from __future__ import print_function
from __future__ import absolute_import


import numpy as np
import os
import re
import argparse

from base_LSTM_model import base_LSTM_m
from utils import Reader, load_data, print_metrics_binary, save_results
import utils
from keras.callbacks import ModelCheckpoint, CSVLogger
from lstm_utils import InHospitalMortalityMetrics, Discretizer, Normalizer
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn import metrics
import keras_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--dim', type=int, default=16,
                        help='number of hidden units')
parser.add_argument('--depth', type=int, default=2,
                    help='number of bi-LSTMs')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of chunks to train')
parser.add_argument('--load_state', type=str, default="",
                    help='state file path')
parser.add_argument('--mode', type=str, default="train",
                    help='mode: train or test')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
parser.add_argument('--save_every', type=int, default=1,
                    help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="",
                    help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--rec_dropout', type=float, default=0.0,
                    help="dropout rate for recurrent connections")
parser.add_argument('--batch_norm', type=bool, default=False,
                    help='batch normalization')
parser.add_argument('--timestep', type=float, default=1.0,
                    help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--whole_data', dest='small_part', action='store_false')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9,
                    help='beta_1 param for Adam optimizer')
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--size_coef', type=float, default=4.0)
parser.add_argument('--normalizer_state', type=str, default=None,
                    help='Path to a state file of a normalizer. Leave none if you want to '
                         'use one of the provided ones.')
parser.add_argument('--deep_supervision', type=bool, default=False,
                    help='set deep supervision for the model')
parser.set_defaults(small_part=False)

parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../data/preprocessed_data/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default=os.path.join(os.path.dirname(__file__), '../output/'))


args = parser.parse_args()
print(args)

drop_out = args.dropout
depth = args.depth
batch_norm = args.batch_norm
mode = args.mode
dim = args.dim
target_repl_coef = args.target_repl_coef
data = args.data
rec_dropout = args.dropout
timestep = args.timestep
normalizer_state = args.normalizer_state
imputation=args.imputation
l1 = args.l1
l2 = args.l2
batch_size=args.batch_size
optimizer=args.optimizer
lr=args.lr
beta_1=args.beta_1
load_state=args.load_state
verbose=args.verbose
epochs=args.epochs
output_dir=args.output_dir
small_part=args.small_part
save_every=args.save_every
deep_supervision=args.deep_supervision
# mode = 'train'
# target_repl_coef = 0.0
# target_repl = (target_repl_coef > 0.0 and mode == 'train')
# data = os.path.join(os.path.dirname(__file__), '../data/in-hospital-mortality/')
# timestep = 1.0
# normalizer_state = None
# imputation='previous'
# l1 = 0
# l2 = 0
# batch_size=8
# optimizer='adam'
# lr=0.001
# beta_1=0.9
# load_state=''
# verbose=2
# epochs=10
# output_dir=os.path.join(os.path.dirname(__file__), '../output/')
# small_part=False
# save_every=1

target_repl = (target_repl_coef > 0.0 and mode == 'train')

if small_part:
    save_every = 2**30


# Build readers, discretizers, normalizers
train_reader = Reader(dataset_dir=os.path.join(data, 'train'),
                                         listfile=os.path.join(data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = Reader(dataset_dir=os.path.join(data, 'train'),
                                       listfile=os.path.join(data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = normalizer_state
if normalizer_state is None:
    # normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(timestep, imputation)
    normalizer_state = './resources/ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(
        timestep, imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

# Build the model
print("==> using model base_Lstm")
model = base_LSTM_m(dim=dim, batch_norm=batch_norm, dropout=drop_out, rec_dropout=rec_dropout,deep_supervision=deep_supervision,num_classes=1,depth=depth,target_repl=target_repl)
suffix = ".bs{}{}{}.ts{}{}".format(batch_size,
                                   ".L1{}".format(l1) if l1 > 0 else "",
                                   ".L2{}".format(l2) if l2 > 0 else "",
                                   timestep,
                                   ".trc{}".format(target_repl_coef) if target_repl_coef > 0 else "")
model.final_name = model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': optimizer,
                    'config': {'lr': lr,
                               'beta_1': beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss, loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if load_state != "":
    model.load_weights(load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", load_state).group(1))


# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part)

if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if mode == 'train':

    # Prepare training
    path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(target_repl_coef > 0),
                                                              batch_size=batch_size,
                                                              verbose=verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=save_every)

    keras_logs = os.path.join(output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    history = model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=verbose,
              batch_size=batch_size)

    precision, recall = metrics_callback.getPrecisionRecall()






elif mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = utils.Reader(dataset_dir=os.path.join(data, 'test'),
                                            listfile=os.path.join(data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]



    predictions = model.predict(data, batch_size=batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]

    ret = utils.print_metrics_binary(labels, predictions)

    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)

    auprc = ret['auprc']

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('LSTM Precision-Recall curve: auprc={0:0.3f}'.format(auprc))
    plt.show()



    path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)
    if target_repl_coef > 0.0:
        model.save('./output/base_lstm_DS.h5')
    else:
        model.save('./output/base_lstm_.h5')

else:
    raise ValueError("Wrong value for args.mode")
