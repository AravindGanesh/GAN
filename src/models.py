import tensorflow as tf

class DC_GAN():
    def __init__(self, sess, real_img_dim, gen_img_dim, noise_dim, batch_size, num_channels=3):
        self.sess = sess
        self.noise_dim = noise_dim
        self.D_input_dim = real_img_dim
        self.G_output_dim = gen_img_dim
        self.batch_size = batch_size
        self.channels = num_channels
        self.saver = tf.train.Saver()
        
    def noise(self, batch_size, noise_dim, pdf='uniform'): 
        '''
        Inputs: batch size while training, dimension of noise
        returns random sample from the distribution - pdf (uniform by default)
        '''
        batch_size = self.batch_size
        noise_dim = self.noise_dim
        if pdf=='uniform':
            return tf.random_uniform(minval=-1, maxval=1, shape=(batch_size, noise_dim), name='z')
        if pdf=='normal':
            return tf.random_normal(mean=0, stddev=1, shape=(batch_size, noise_dim), name='z')
        
    def batch_norm(self, batch):
        return tf.contrib.layers.batch_norm(inputs=batch,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            is_training=True
            )
    
    def add_conv_layer(self, inputs, filter_size, in_channels, out_channels, scope_name, _batch_norm=True):
        with tf.variable_scope(scope_name):
            shape = [filter_size, filter_size, in_channels, out_channels]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.02), trainable=True)
            biases = tf.Variable(tf.constant(0.05, shape=[out_channels]), trainable=True)
            
            layer = tf.nn.conv2d(
                input=inputs,
                filter=weights,
                strides=[1, 2, 2, 1],
                padding='SAME') + biases
            if _batch_norm: layer = self.batch_norm(layer)
            layer = tf.nn.leaky_relu(layer, alpha=0.2)
            return layer, weights, biases
            
    def add_conv_transpose_layer(self,inputs,filter_size,in_channels,out_channels,scope_name,activation,_batch_norm=True):
        with tf.variable_scope(scope_name):
            shape = [filter_size, filter_size, out_channels, in_channels]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.02), trainable=True)
            biases = tf.Variable(tf.constant(0.05, shape=[out_channels]), trainable=True)

            layer = tf.nn.conv2d_transpose(
                input=inputs,
                filter=weights,
                strides=[1, 2, 2, 1],
                padding='SAME') + biases
            if _batch_norm: layer = self.batch_norm(layer)
            if activation == 'relu': layer = tf.nn.relu(layer)
            elif activation == 'tanh': layer = tf.nn.tanh(layer)

            return layer, weights, biases

    def discriminator(self, img):
        scope_name='discriminator'
        h1, w1, b1 = self.add_conv_layer(
            inputs = img,
            filter_size=5,
            in_channels=self.num_channels,
            out_channels=64,
            _batch_norm=False,
            scope_name=scope_name
        )
        h2, w2, b2 = self.add_conv_layer(
            inputs=h1,
            filter_size=5,
            in_channels=64,
            num_filters=128,
            scope_name=scope_name
        )
        h3, w3, b3 = self.add_conv_layer(
            inputs=h2,
            filter_size=5,
            in_channels=128,
            out_channels=256,
            scope_name=scope_name,
        )
        logits = tf.layers.dense(
            inputs=tf.reshape(h3, shape=[self.batch_size, -1]),
            units=1,
            use_bias=True,
            kernel_initalizer=tf.contrib.layers.xavier_initializer()
        )
        return logits
        
    def generator(self, z):
        scope_name = 'generator'
        _z = tf.layers.dense(
            inputs=z,
            units=4*4*1024,
            use_bias=True,
            kernel_initalizer=tf.contrib.layers.xavier_initializer()
        ) # projection of z to 4x4x1024 layer (linear)
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
            num_filters=self.num_channels,
            _batch_norm=False,
            scope_name=scope_name,
            activation='tanh'
        )
        return G_out
    
    def generate_imgs(self, num_imgs):
        z = self.noise(num_imgs, self.noise_dim)
        gen_imgs = self.generator(z)
        return gen_imgs
        
    def compile_model(self):
        self.real_imgs = tf.placeholder(dtype=tf.float32,shape=[self.batch_size]+ self.real_img_dim,name='real')
        self.z = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.noise_dim], name='noise')
        self.fake_imgs = self.generator(self.z)
        self.logits_real = self.discriminator(self.real_imgs)
        self.logits_fake = self.discriminator(self.fake_imgs)
        self.D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_real, labels=tf.ones_like(self.logits_real))
        self.D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_fake, labels=tf.zeros_like(self.logits_fake))
        self.D_loss = tf.reduce_mean(self.D_loss_real) + tf.reduce_mean(self.D_loss_fake)
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_fake, labels=tf.ones_like(self.logits_fake)))
        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        
    def train(self, real_data, learning_rate=0.0002, beta1=0.5, epochs):
        # create tf.data.Dataset() instance
        train_data = tf.data.Dataset.from_tensor_slices(real_data) 
        # D train step
        self.D_train = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1).minimize(self.D_loss, var_list=self.D_vars)
        # G train step
        self.G_train = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1).minimize(self.G_loss, var_list=self.G_vars)
        
        tf.global_variables_initializer().run()

        for epoch in range(epochs):
            