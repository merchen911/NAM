import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import L2
from tensorflow.keras import regularizers

from .modules import ExU, feature_nns

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class NAM(Model):
    def __init__(
        self,
        n_feature,
        exu_unit,
        linear_unit,
        mode = 'cls',
        n_class = 1,
        dropout_ratio = .1,
        layer_regularization = .1,
        feature_regularization = .1,
        first_layer_type = 'ExU',
    ):
        super(NAM,self).__init__()
        
        assert mode in ['cls','reg'], "Choose one of mode 'cls' or 'reg'"
        assert first_layer_type in ['ExU','Linear'], "Choose one of first layer type 'ExU' or 'Linear'"
        
        self.first_layer_type = first_layer_type
        self.mode = mode
        self.n_feature = n_feature
        self.exu_unit = exu_unit
        self.linear_unit = linear_unit
        self.dropout_ratio = dropout_ratio
        self.n_class = n_class
        self.layer_regularization = layer_regularization
        self.feature_regularization = feature_regularization
        
        if mode == 'cls':
            assert n_class > 1, "The number of class must be bigger than one"
            
        
        self.b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=self.b_init(shape=(self.n_class,), 
                                      dtype='float32',),
            name = 'Bias',
            trainable=True
        )
        
        self.features_list = [
            feature_nns(
                self.linear_unit, 
                self.exu_unit, 
                self.n_feature,
                n, 
                self.dropout_ratio,
                layer_regularization = layer_regularization,
                n_output = n_class,
                first_layer_type = first_layer_type
            ) for n in range(self.n_feature)
        ]
        self.feature_drop = Dropout(dropout_ratio)
        

    def extract(self, inputs, training = True):
        output_list = []
        for n,feature_unit in enumerate(self.features_list):
            output_list.append(feature_unit(inputs[:,n][...,tf.newaxis],training))
        return tf.stack(output_list,1)
        

    def __call__(self, inputs, training = True):
        weights = self.extract(inputs,training)
        
        ### added feature regularization
        self.losses.append(
            tf.reduce_mean( 
                tf.reduce_sum(weights ** 2, 1) / self.n_feature
            ) * self.feature_regularization
        )
        
        if training:
            weights = self.feature_drop(weights,training)
            
        logits = tf.reduce_sum(weights,1) + self.b
            
        if self.mode == 'cls':
            return tf.math.softmax(logits,1)
        else:
            return logits
        
    
    def loss_fn(self, logits, targets):
        ### general loss
        if self.mode == 'cls':
            loss = tf.keras.losses.categorical_crossentropy(targets,logits,from_logits=True)
        else:
            loss = tf.losses.mean_squared_error(targets,logits)
        ### added L2 regularization
        loss += tf.reduce_sum(self.losses)
        return loss