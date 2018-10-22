class DC_GAN():
    def __init__(self, sess, real_img_dim, gen_img_dim, noise_dim, batch_size, num_channels=3, ):
        self.sess = sess
        self.noise_dim = noise_dim
        self.D_input_dim = real_img_dim
        self.G_output_dim = gen_img_dim
        self.batch_size = batch_size
        self.channels = num_channels
        self.saver = tf.train.Saver()
        
    def noise(self, batch_size=self.batch_size, noise_dim=self.noise_dim, pdf='uniform'): 
        '''
        Inputs: batch size while training, dimension of noise
        returns random sample from the distribution - pdf (uniform by default)
        '''
        if pdf='uniform':
            return tf.random_uniform(minval=-1, maxval=1, shape=(batch_size, noise_dim), name='z')
        if pdf='normal':
            return tf.random_normal(mean=0, stddev=1, shape=(batch_size, noise_dim), name='z')
        
    def add_conv_layer(self, inputs, filter_size, in_channels, out_channels, scope_name):
        with tf.variable_scope(scope_name):
            shape = [filter_size, filter_size, in_channels, out_channels]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
            biases = tf.Variable(tf.constant(0.05, shape=[out_channels]))
            
            layer = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, 2, 2, 1],
                padding='SAME') + biases
            
            layer = tf.nn.leaky_relu(layer, alpha=0.2)
            return layer, weights, biases
            
    def add_conv_transpose_layer(self,inputs,filter_size,in_channels,out_channels,scope_name,activation):
        with tf.variable_scope(scope_name):
            shape = [filter_size, filter_size, out_channels, in_channels]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
            biases = tf.Variable(tf.constant(0.05, shape=[out_channels]))

            layer = tf.nn.conv2d_transpose(
                input=inputs,
                filter=weights,
                strides=[1, 2, 2, 1],
                padding='SAME') + biases

            if activation == 'relu': layer = tf.nn.relu(layer)
            else if activation = 'tanh': layer = tf.nn.tanh(layer)    

            return layer, weights, biases

    def discriminator(self, img):
        scope_name='discriminator'
        h1, w1, b1 = self.add_conv_layer(
            inputs = img,
            filter_size=5,
            in_channels=3,
            out_channels=64,
            scope_name=scope_name,
        )
        h2, w2, b2 = self.add_conv_layer(
            inputs=h1,
            filter_size=5,
            in_channels=64,
            num_filters=128,
            scope_name=scope_name,
        )
        h3, w3, b3 = self.add_conv_layer(
            inputs=h2,
            filter_size=5,
            in_channels=128,
            out_channels=256,
            scope_name=scope_name,
        )
        h4, w4, b4 = self.add_conv_layer(
            inputs=h3,
            filter_size=5,
            in_channels=256,
            out_channels=512,
            scope_name=scope_name,
        )
        D_out = tf.layers.dense(
            inputs=tf.reshape(h4, shape=[self.batch_size, -1]),
            units=1,
            activation=tf.nn.sigmoid,
            use_bias=True,
            kernel_initalizer=tf.contrib.layers.xavier_initializer()
        )

        return D_out
        
    def generator(self, z):
        scope_name = 'generator'
        _z = tf.layers.dense(
            inputs=z,
            units=4*4*1024,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initalizer=tf.contrib.layers.xavier_initializer()
        )
        h1 = tf.reshape(_z, [-1, 4,4,1024])
        h2, w2, b2 = self.add_conv_transpose_layer(
            inputs=h1,
            filter_size=5,
            in_channels=1024,
            out_channels=512,
            scope_name=scope_name,
            activation='relu'
        )
        h3, w3, b3 = self.add_conv_transpose_layer(
            inputs=h2,
            filter_size=5,
            in_channels=512,
            out_channels=256,
            scope_name=scope_name,
            activation='relu'
        )
        h4, w4, b4 = self.add_conv_transpose_layer(
            inputs=h4,
            filter_size=5,
            in_channels=256,
            out_channels=128,
            scope_name=scope_name,
            activation='relu'
        )
        G_out, w5, b5 = self.add_conv_transpose_layer(
            inputs=h4,
            filter_size=5,
            in_channels=128,
            num_filters=3,
            scope_name=scope_name,
            activation='tanh'
        )
        return G_out
    
    