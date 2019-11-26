import tensorflow as tf

class NeuralNetwork(object):
    
    def __init__(self,args,n_user,n_item):
        self._parse_args(args,n_user,n_item)
        self._build_inputs()
        self._build_params()
        self._build_embeddings()
        self.predict()
        self._build_loss()
        self._build_train()
        
        
    def _parse_args(self,args,n_user, n_item):
        self.n_user = n_user
        self.n_item = n_item
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.l2_weight = args.l2_weight
        self.lr = args.learning_rate
        
        
    
    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='item_indices')
        self.ratings = tf.placeholder(dtype=tf.float32, shape=[None], name='ratings')
        
        
    def _build_params(self):
        self.user_feature_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.n_user, self.embed_dim]),
                                               name='user_feature_matrix')
        self.item_feature_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[self.n_item, self.embed_dim]),
                                               name='item_feature_matrix')
        self.W_1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.embed_dim * 2, self.hidden_dim], name='W_1'))
        self.b_1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.hidden_dim], name='b_1'))
        self.W_2 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[self.hidden_dim, 1], name='W_2'))
        self.b_2 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[1], name='b_2'))
        
        
    def _build_embeddings(self):
        self.user_features = tf.nn.embedding_lookup(params=self.user_feature_matrix, ids=self.user_indices)
        self.item_features = tf.nn.embedding_lookup(params=self.item_feature_matrix, ids=self.item_indices)
   
     
    def predict(self):
        self.concat = tf.concat([self.user_features, self.item_features], axis=-1)
        self.hidden = tf.nn.relu(tf.matmul(self.concat, self.W_1) + self.b_1)
        self.output = tf.matmul(self.hidden, self.W_2) + self.b_2
        
        
    def _build_loss(self):
        self.pred_loss = tf.reduce_mean(tf.square(self.ratings - self.output))
        self.l2_loss = self.l2_weight * (tf.nn.l2_loss(self.user_features) + tf.nn.l2_loss(self.item_features) + 
                                         tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2))
        self.loss = self.pred_loss + self.l2_loss
        
        
    def _build_train(self):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.rmse = tf.sqrt(self.pred_loss)
        
        
    def train(self, sess, feed_dict):
        return sess.run(self.optimizer,feed_dict)
    
    
    def validation(self, sess, feed_dict):
        return sess.run([self.loss, self.rmse], feed_dict)