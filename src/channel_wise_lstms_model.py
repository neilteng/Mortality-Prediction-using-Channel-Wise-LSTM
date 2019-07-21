from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from lstm_utils import Slice


class channel_wise_lstms(Model):

    def __init__(self, hid_dim, batch_norm, dropout, recurrent_dropout, channels,
                 num_classes=1,
                 depth=1, input_dim=76, model_size=4, **kwargs):

        self.hid_dim = hid_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.depth = depth
        self.model_size = model_size

        # Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mask_X = Masking()(X)


        # Preprocess each channel
        channels_X = []
        for ch in channels:
            channels_X.append(Slice(ch)(mask_X))
        channel_lstm__X = []  # LSTM processed version of channels_X
        for cx in channels_X:
            lstm_cx = cx
            for i in range(depth):
                units = hid_dim//2

                lstm = LSTM(units=units,
                            activation='tanh',
                            return_sequences=True,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout)

                lstm_cx = Bidirectional(lstm)(lstm_cx)

            channel_lstm__X.append(lstm_cx)

        # Concatenate processed channels
        CWX = Concatenate(axis=2)(channel_lstm__X)

        # LSTM for time series lstm processed data
        for i in range(depth-1):    # last layer is left for manipulation of output.
            units = int(model_size*hid_dim)//2

            lstm = LSTM(units=units,
                        activation='tanh',
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout)

            CWX = Bidirectional(lstm)(CWX)

        # Output module of the network
        last_layer = LSTM(units=int(model_size*hid_dim),
                 activation='tanh',
                 return_sequences=False,
                 dropout=dropout,
                 recurrent_dropout=recurrent_dropout)(CWX)

        if dropout > 0:
            last_layer = Dropout(dropout)(last_layer)


        y = Dense(num_classes, activation='sigmoid')(last_layer)
        outputs = [y]

        super(channel_wise_lstms, self).__init__(inputs=inputs, outputs=outputs)
