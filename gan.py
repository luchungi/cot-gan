import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float32')

class SimpleGenerator(tf.keras.Model):
    '''
    Generator for creating fake time series data (y_1, y_2,...,y_T) from the latent variable Z.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN model has shape
                 [batch_size, time_step, z_hidden_dims]
    Returns:
          output of RNN generator
    '''
    def __init__(self, batch_size, seq_len, time_steps, sub_seq_len, dx, state_size, z_dims, training_scheme,
                 rnn_activation='sigmoid', output_activation='linear'):
        super(SimpleGenerator, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.time_steps = time_steps
        self.sub_seq_len = sub_seq_len
        self.dx = dx
        dense_layer_unit = max(10, self.sub_seq_len*2)
        self.state_size = state_size
        self.z_dims = z_dims
        self.training_scheme = training_scheme

        self.rnn_activation = rnn_activation
        self.output_activation = output_activation
        self.l2_regularisation = None
        self.counter = 1
        # last lstm output as the input to dense layer
        self.dense_layer = tf.keras.layers.Dense(units=self.state_size, activation='relu', use_bias=True)
        self.dense_layer2 = tf.keras.layers.Dense(units=self.state_size, activation='relu', use_bias=True)
        self.output_layer = tf.keras.layers.Dense(units=self.seq_len, activation=self.output_activation,
                                                  use_bias=True)

    def call(self, inputs, training=True, mask=None):
        y = self.dense_layer(inputs)
        y = self.dense_layer2(y)
        y = self.output_layer(y)
        y = tf.reshape(tensor=y, shape=[self.batch_size, self.seq_len//self.dx, self.dx])
        return y

class ToyGenerator(tf.keras.Model):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z.
    Args:
        inputs: (numpy array) latent variables as inputs to the RNN model has shape
                [batch_size, time_step, sub_sequence_hidden_dims]
    Returns:
        output of generator
    '''
    def __init__(self, batch_size, time_steps, Dz, Dx, state_size, filter_size, output_activation='sigmoid', bn=False,
                 nlstm=1, nlayer=2, Dy=0, rnn_bn=False):
        super().__init__()

        self.Dz = Dz
        self.Dy = Dy
        self.Dx = Dx
        self.batch_size = batch_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.rnn = tf.keras.Sequential()
        k_init = None
        self.rnn.add(layers.LSTM(self.state_size, return_sequences=True, recurrent_initializer=k_init, kernel_initializer=k_init))
        if rnn_bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())

        for i in range(nlstm-1):
            self.rnn.add(layers.LSTM(self.state_size, return_sequences=True, recurrent_initializer=k_init, kernel_initializer=k_init))
            if rnn_bn:
                self.rnn.add(tf.keras.layers.BatchNormalization())

        self.fc = tf.keras.Sequential()
        for i in range(nlayer-1):
            self.fc.add(layers.Dense(units=filter_size, activation=None, use_bias=True))
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(units=Dx, activation=output_activation, use_bias=True))

    def call(self, inputs, y=None, training=True, mask=None):

        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.time_steps, self.Dz])
        if y is not None:
            y = tf.broadcast_to(y[:, None, :], [self.batch_size, self.time_steps, self.Dy])
            z = tf.concat([z, y], -1)

        lstm = self.rnn(z, training=training)

        x = self.fc(lstm, training=training)
        x = tf.reshape(tensor=x, shape=[self.batch_size, self.time_steps, self.Dx])
        return x

class GenLSTM(tf.keras.Model):
    def __init__(self, noise_dim, seq_dim, seq_len, hidden_size=64, n_lstm_layers=1, activation='relu', log_series=True):
        super().__init__()
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.log_series = log_series

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim, seq_len), units=hidden_size, return_sequences=True, return_state=True)
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(seq_dim)
        ])

    def call(self, noise, training=True, mask=None):
        batch_size = tf.shape(noise)[0]
        if self.log_series:
            x = tf.zeros([batch_size, 1, self.seq_dim])
        else:
            x = tf.ones([batch_size, 1, self.seq_dim])
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        seq = [x]
        for i in range(self.seq_len-1):
            input = tf.concat([x, noise[:,i:i+1,:]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c], training=training)
            x = self.net(output, training=training)
            seq.append(x)
        output_seq = tf.concat(seq, axis=1)
        output_seq = tf.cumsum(output_seq, axis=1)
        return output_seq

class GenLSTMp(tf.keras.Model):
    def __init__(self, noise_dim, seq_dim, seq_len, hidden_size=64, n_lstm_layers=1, activation='relu', log_series=True):
        super().__init__()
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.activation = activation
        self.log_series = log_series

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim, seq_len), units=hidden_size, return_sequences=True, return_state=True)
        self.mean_net = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(seq_dim)
        ])
        self.var_net = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(seq_dim)
        ])

    def call(self, noise, training=True, mask=None):
        batch_size = tf.shape(noise)[0]
        if self.log_series:
            x = tf.zeros([batch_size, 1, self.seq_dim])
        else:
            x = tf.ones([batch_size, 1, self.seq_dim])
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        seq = [x]
        for i in range(self.seq_len-1):
            input = tf.concat([x, noise[:,i:i+1,:]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c], training=training)
            mu = self.mean_net(output, training=training)
            logvar = self.var_net(output, training=training)
            self.std = tf.exp(0.5 * logvar)
            out_dist = tfp.distributions.Normal(mu, self.std)
            x = out_dist.sample()
            seq.append(x)
        output_seq = tf.concat(seq, axis=1)
        output_seq = tf.cumsum(output_seq, axis=1)
        return output_seq

class GenLSTMpdt(tf.keras.Model):
    def __init__(self, noise_dim, seq_dim, seq_len, dt, hidden_size=64, n_lstm_layers=1, activation='relu', log_series=True):
        super().__init__()
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.activation = activation
        self.dt = dt
        self.log_series = log_series

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim, seq_len), units=hidden_size, return_sequences=True, return_state=True)
        self.mean_net = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(seq_dim)
        ])
        self.var_net = tf.keras.Sequential([
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(hidden_size, activation=activation),
            layers.Dense(seq_dim)
        ])

    def call(self, noise, training=True, mask=None):
        batch_size = tf.shape(noise)[0]
        if self.log_series:
            x = tf.zeros([batch_size, 1, self.seq_dim])
        else:
            x = tf.ones([batch_size, 1, self.seq_dim])
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        seq = [x]
        for i in range(self.seq_len-1):
            input = tf.concat([x, noise[:,i:i+1,:]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c], training=training)
            mu = self.mean_net(output, training=training)
            logvar = self.var_net(output, training=training)
            self.std = tf.exp(0.5 * logvar) * self.dt
            out_dist = tfp.distributions.Normal(mu, self.std)
            x = out_dist.sample()
            seq.append(x)
        output_seq = tf.concat(seq, axis=1)
        output_seq = tf.cumsum(output_seq, axis=1)
        return output_seq

class GenLSTMd(tf.keras.Model):
    def __init__(self, noise_dim, seq_dim, seq_len, hist_len, hidden_size=64):
        super().__init__()
        assert hist_len != 1, 'Historical length must be 0 or greater than 1'
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hist_len = hist_len
        self.hidden_size = hidden_size

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim+1, seq_len), units=hidden_size, return_sequences=True, return_state=True, unroll=True)
        self.output_net = layers.Dense(seq_dim)

    def _condition_lstm(self, noise, hist_x, t):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        seq = tf.zeros([batch_size, 1, 1])

        dts = t[:, 1:, :] - t[:, :-1, :]
        if self.hist_len > 1: # feed in the historical data to get the hidden state
            diff_x = hist_x[:, 1:, :] - hist_x[:, :-1, :]
            input = tf.concat([diff_x, noise[:, :self.hist_len-1, :], dts[:, :self.hist_len-1, :]], axis=-1) # hist_len-1 returns for seq of length hist_len
            output, h, c = self.rnn(input, initial_state=[h, c])
            noise = noise[:,self.hist_len-1:,:] # set the noise to start from the end of the historical data
            dts = dts[:,self.hist_len-1:,:] # continue from the last dt
            seq = tf.concat([seq, diff_x], axis=1)
        else:
            diff_x = tf.zeros([batch_size, 1, self.seq_dim])
            input = tf.concat([diff_x, noise[:, :1, :], dts[:, :1, :]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c])
            noise = noise[:,1:,:]
            dts = dts[:,1:,:]
        # print(seq.shape, output.shape)
        return seq, output[:,-1:,:], noise, dts, h, c

    def _generate_sequence(self, seq, output, noise, dts, h, c):
        gen_seq = []
        for i in range(self.seq_len-seq.shape[1]): # iterate over the remaining time steps
            x = self.output_net(output)
            # print(f'Generated single step output shape: {x.shape}')
            gen_seq.append(x)
            if i < noise.shape[1]:
                input = tf.concat([x, noise[:,i:i+1,:], dts[:,i:i+1,:]], axis=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, h, c = self.rnn(input, initial_state=[h, c])
        # print(f'Historical sequence shape: {seq.shape}')
        # print(seq)
        output_seq = tf.concat(gen_seq, axis=1)
        # print(f'Generated sequence shape: {output_seq.shape}')
        # print(output_seq)
        output_seq = tf.concat([seq, output_seq], axis=1)
        return output_seq

    def call(self, noise, x, training=True, mask=None):
        t = x[:,:,0:1]
        hist_x = x[:,:self.hist_len,1:] if self.hist_len > 1 else None
        # print(f'call x:{x.shape} / noise:{noise.shape} / t:{t.shape} / hist_x:{hist_x.shape if hist_x is not None else []}')
        seq, output, noise, dts, h, c = self._condition_lstm(noise, hist_x, t)
        # print(f'call noise:{noise.shape} / dts:{dts.shape}')
        output_seq = self._generate_sequence(seq, output, noise, dts, h, c)
        output_seq = tf.cumsum(output_seq, axis=1)
        output_seq = tf.concat([t, output_seq], axis=2)
        return output_seq

class GenLSTMd_v2(tf.keras.Model):
    def __init__(self, noise_dim, seq_dim, seq_len, hist_len, hidden_size=64):
        super().__init__()
        assert hist_len != 1, 'Historical length must be 0 or greater than 1'
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hist_len = hist_len
        self.hidden_size = hidden_size

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim+1, seq_len), units=hidden_size, return_sequences=True, return_state=True)
        # self.rnn.build(input_shape=(seq_dim+noise_dim+1, seq_len))
        self.output_net = layers.Dense(seq_dim)

    def _condition_lstm(self, noise, hist_x, t):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        seq = tf.zeros([batch_size, 1, 1])

        dts = t[:, 1:, :] - t[:, :-1, :]
        if self.hist_len > 1: # feed in the historical data to get the hidden state
            diff_x = hist_x[:, 1:, :] - hist_x[:, :-1, :]
            input = tf.concat([diff_x, noise[:, :self.hist_len-1, :], dts[:, :self.hist_len-1, :]], axis=-1) # hist_len-1 returns for seq of length hist_len
            output, h, c = self.rnn(input, initial_state=[h, c])
            noise = noise[:,self.hist_len-1:,:] # set the noise to start from the end of the historical data
            dts = dts[:,self.hist_len-1:,:] # continue from the last dt
            seq = tf.concat([seq, diff_x], axis=1)
        else:
            diff_x = tf.zeros([batch_size, 1, self.seq_dim])
            input = tf.concat([diff_x, noise[:, :1, :], dts[:, :1, :]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c])
            noise = noise[:,1:,:]
            dts = dts[:,1:,:]
        # print(seq.shape)
        return seq, output[:,-1:,:], noise, dts, h, c

    def _generate_sequence(self, seq, output, noise, dts, h, c):
        gen_seq = []
        for i in range(self.seq_len-seq.shape[1]): # iterate over the remaining time steps
            x = self.output_net(output)
            # print(f'Generated single step output shape: {x.shape}')
            gen_seq.append(x)
            if i < noise.shape[1]:
                input = tf.concat([x, noise[:,i:i+1,:], dts[:,i:i+1,:]], axis=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, h, c = self.rnn(input, initial_state=[h, c])
        # print(f'Historical sequence shape: {seq.shape}')
        # print(seq)
        output_seq = tf.concat(gen_seq, axis=1)
        # print(f'Generated sequence shape: {output_seq.shape}')
        # print(output_seq)
        output_seq = tf.concat([seq, output_seq], axis=1)
        return output_seq

    def call(self, inputs, training=True, mask=None):
        noise, x = inputs
        t = x[:,:,0:1]
        hist_x = x[:,:self.hist_len,1:] if self.hist_len > 1 else None
        # print(f'call x:{x.shape} / noise:{noise.shape} / t:{t.shape} / hist_x:{hist_x.shape if hist_x is not None else []}')
        seq, output, noise, dts, h, c = self._condition_lstm(noise, hist_x, t)
        # print(f'call noise:{noise.shape} / dts:{dts.shape}')
        output_seq = self._generate_sequence(seq, output, noise, dts, h, c)
        output_seq = tf.cumsum(output_seq, axis=1)
        output_seq = tf.concat([t, output_seq], axis=2)
        return output_seq

class LSTMusic(tf.keras.Model):
    '''
    LSTM model that generates a sequence of delta pitch values
    The output of the LSTM is passed through a linear layer followed by a tanh activation to ensure that the delta pitch values are within the range [-1, 1]
    This value is then multiplied by the dpitch_range (input) to get the actual delta pitch value
    The gap and pitch duration are given as input from the reference sample
    Previous solution tried to generate these values by taking the exponential of the output of the LSTM
    However, the gap ended getting too large towards the end of the sequence
    '''

    def __init__(self, noise_dim:int, seq_dim: int, seq_len: int,
                 dpitch_range: int=24, scale: float=1.0,
                 hidden_size:int =64, n_lstm_layers: int=1, activation: str='Tanh'):
        super().__init__()
        self.gen_type = 'LSTMusic'
        self.seq_dim = seq_dim # dimension of the time series
        self.noise_dim = noise_dim # dimension of the noise vector -> vector of (noise_dim, 1) concatenated with the seq value of dimension seq_dim at each time step
        self.seq_len = seq_len # length of the time series
        self.dpitch_range = dpitch_range
        self.scale = scale
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        self.rnn = layers.LSTM(input_shape=(seq_dim+noise_dim, seq_len), units=hidden_size, return_sequences=True, return_state=True, unroll=True)
        self.output_net = layers.Dense(1, activation='tanh')

    def _condition_lstm(self, noise, hist_x):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        dist = tf.cumsum(hist_x[:,:,-1:], axis=1) # distance from the initial note
        # print(hist_x[0,:,-1], dist[0,:,-1])

        if hist_x is not None: # feed in the historical data to get the hidden state
            input = tf.concat([hist_x, dist, noise[:, :hist_x.shape[1], :]], axis=-1)
            output, h, c = self.rnn(input, initial_state=[h, c])
            noise = noise[:,hist_x.shape[1]:,:] # set the noise to start from the end of the historical data
        else:
            output = tf.zeros(batch_size, 1, self.hidden_size, requires_grad=False, device=noise.device)
        return output[:,-1:,:], noise, h, c, dist[:,-1:,:]

    def _generate_sequence(self, output, noise, h, c, gap_duration, dist):
        gen_seq = []
        for i in range(noise.shape[1]+1): # +1 for the first note which is using the output passed in
            z = self.output_net(output)
            # gap_duration = torch.exp(z[:,:,:2]) # ensure that the duration and pause duration are positive
            deltapitch = z * self.dpitch_range * self.scale
            x = tf.concat([gap_duration[:,i:i+1,:], deltapitch], axis=-1)
            gen_seq.append(x)
            dist = dist + deltapitch
            if i < noise.shape[1]:
                input = tf.concat([x, dist, noise[:,i:i+1,:]], axis=-1) # len=1, batch_size, input_size=X.shape[-1]+noise_dim+1 for dt
                output, h, c = self.rnn(input, initial_state=[h, c])
        output_seq = tf.concat(gen_seq, axis=1)
        output_seq = tf.cumsum(output_seq, axis=1)
        return output_seq

    def call(self, noise, hist_x=None, gap_duration=None, training=True, mask=None):
        output, noise, h, c, dist = self._condition_lstm(noise, hist_x)
        output_seq = self._generate_sequence(output, noise, h, c, gap_duration, dist)
        if hist_x is None:
            return output_seq
        else:
            return tf.concat([hist_x, output_seq], axis=1)

class ToyDiscriminator(tf.keras.Model):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, Dz, Dx, state_size, filter_size, bn=False, kernel_size=5, strides=1,
                 output_activation="tanh", nlayer=2, nlstm=0):
        super().__init__()

        self.batch_size = batch_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.Dz = Dz
        self.Dx = Dx

        self.fc = tf.keras.Sequential()
        self.fc.add(tf.keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size,
                                           padding="causal", strides=strides))

        for i in range(nlayer-1):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.ReLU())
            self.fc.add(tf.keras.layers.Conv1D(filters=state_size, kernel_size=kernel_size,
                                               activation=output_activation if i == nlayer-2 else None,
                                               padding="causal", strides=strides))
        for i in range(nlstm):
            if bn:
                self.fc.add(tf.keras.layers.BatchNormalization())
            self.fc.add(layers.LSTM(state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):

        x = tf.reshape(tensor=inputs, shape=[self.batch_size, self.time_steps, self.Dx])
        z = self.fc(x)
        return z

class VideoDCG(tf.keras.Model):
    '''
    Generator for creating fake video sequence (y_1, y_2,...,y_T) from the latent variable Z.
    Args:
         inputs: (numpy array) latent variables as inputs to the RNN layers has shape
                 [batch_size, time_step, z_weight*z_height]
    Returns:
          output of generator: fake video sequence (y_1, y_2,...,y_T)
          of shape [batch_size, x_height, x_weight*time_step, channel]
    '''
    def __init__(self, batch_size, time_steps, x_width, x_height, z_width, z_height, state_size,
                 filter_size=64, bn=False, output_activation="sigmoid", nlstm=1, cat=False, nchannel=3):
        super(VideoDCG, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.nlstm = nlstm
        self.cat = cat
        self.nchannel = nchannel

        # last lstm output as the input to dense layer
        self.last_lstm_h = None
        self.bn = bn

        self.lstm_layer1 = tf.keras.layers.LSTM(self.state_size, return_sequences=True)
        if self.bn:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.lstm_layer2 = tf.keras.layers.LSTM(self.state_size*2, return_sequences=True)

        if self.bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*self.filter_size*4, use_bias=False))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, self.filter_size*4)))
        # assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(self.filter_size * 4, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        # assert model.output_shape == (None, 16, 16, 128)
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.filter_size*2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        # assert model.output_shape == (None, 16, 16, 128)
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.x_width == 64:
            model.add(layers.Conv2DTranspose(self.filter_size, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            # assert model.output_shape == (None, 32, 32, 64)
            if self.bn:
                model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.nchannel, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         activation=output_activation))

        self.deconv = model

    def call_all(self, inputs_z, inputs_y, training=True, mask=None):
        # for RNN, z has shape of [batch_size, time_step, sub_sequence_hidden_dims]
        z = tf.reshape(tensor=inputs_z, shape=[self.batch_size, self.time_steps, self.z_width*self.z_height])
        y = tf.broadcast_to(inputs_y[:, None, :], [self.batch_size, self.time_steps, inputs_y.shape[-1]])
        zy = tf.concat([z, y], -1)

        lstm_h = self.lstm_layer1(zy)
        if self.cat:
            lstm_h = tf.concat([lstm_h, y], -1)

        if self.bn:
            lstm_h = self.bn1(lstm_h)

        lstm_h = self.lstm_layer2(lstm_h)
        if self.bn:
            lstm_h = self.bn2(lstm_h)

        # input shape for conv3D: (batch, depth, rows, cols, channels)
        conv_inputs = tf.reshape(lstm_h, [self.batch_size * self.time_steps, -1])
        y = self.deconv(conv_inputs)

        y = tf.reshape(y, [self.batch_size, self.time_steps, self.x_height, self.x_width, self.nchannel])
        y = tf.transpose(y, (0, 2, 1, 3, 4))
        y = tf.reshape(tensor=y, shape=[self.batch_size, self.x_height, self.x_width*self.time_steps, self.nchannel])
        return zy, lstm_h, y

    def call(self, *args, **kwargs):
        return self.call_all(*args, **kwargs)[-1]

class VideoDCD(tf.keras.Model):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, x_width, x_height, z_width, z_height, state_size,
                 filter_size=64, bn=False, nchannel=3):
        super(VideoDCD, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.x_width = x_width
        self.x_height = x_height
        self.state_size = state_size
        self.z_width = z_width
        self.z_height = z_height
        self.filter_size = filter_size
        self.bn = bn
        self.nchannel = nchannel

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(self.filter_size, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[x_width, x_height, nchannel]))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(self.filter_size*2, (5, 5), strides=(2, 2), padding='same'))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if x_width == 64:
            model.add(layers.Conv2D(self.filter_size*4, (5, 5), strides=(2, 2), padding='same'))
            if self.bn:
                model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        self.conv = model

        self.rnn = tf.keras.Sequential()
        if x_width == 64:
            self.rnn.add(tf.keras.layers.LSTM(self.filter_size*4, return_sequences=True))
        elif x_width == 32:
            self.rnn.add(tf.keras.layers.LSTM(self.filter_size*2, return_sequences=True))

        if self.bn:
            self.rnn.add(tf.keras.layers.BatchNormalization())
        self.rnn.add(tf.keras.layers.LSTM(self.state_size, return_sequences=True))

    def call(self, inputs, training=True, mask=None):
        # permute original data shape [batch_size, h, timesteps, w, channels]
        # to [batch_size, timesteps, h, w, channels] as convnet inputs
        z = tf.reshape(tensor=inputs, shape=[self.batch_size, self.x_height, self.time_steps,
                                             self.x_width, self.nchannel])
        z = tf.transpose(z, (0, 2, 1, 3, 4))
        z = tf.reshape(tensor=z, shape=[self.batch_size * self.time_steps, self.x_height, self.x_width, self.nchannel])

        z = self.conv(z)
        z = tf.reshape(z, shape=[self.batch_size, self.time_steps, -1])
        z = self.rnn(z)

        return z