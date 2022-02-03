import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import L2
from tensorflow.keras import regularizers


class ExU(Layer):

    def __init__(self, units, n_feature, feature_number = 0, mean = 0, std = .05, 
                 kernel_regularizer = None):
        super(ExU, self).__init__()

        self.units = units
        self.feature_number = feature_number
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        self.w = self.add_weight(
            shape=(1, self.units),
            initializer="random_normal",
            name = 'Feature_{}_ExU_w'.format(str(self.feature_number)),
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            name = 'Feature_{}_ExU_b'.format(str(self.feature_number)),
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul( (inputs - self.b) ,tf.exp(self.w))

    
    
class feature_nns(Model):
    def __init__(self, 
                 linear_unit_list, exu_unit, n_feature,
                 feature_number = 0,
                 dropout_ratio = .1,
                 layer_regularization = .1,
                 n_output = 1,
                 first_layer_type = 'ExU',
                ):
        super(feature_nns,self).__init__()
        
        
        self.n_feature = n_feature
        self.linear_unit_list = linear_unit_list
        self.feature_number = feature_number

        


        
        if first_layer_type == 'ExU':
            self.layer_list = [
                Dense(
                    linear_unit,name = 'Feature_{}_linear_{}'.format(feature_number,n),
                    kernel_regularizer = L2(layer_regularization),
                ) for n, linear_unit in enumerate(linear_unit_list)
            ]
            
            self.layer_list = [
                ExU(exu_unit, n_feature, feature_number,kernel_regularizer = L2(layer_regularization))
            ] + self.layer_list
        else:
            self.layer_list = [
                Dense(
                    linear_unit,name = 'Feature_{}_linear_{}'.format(feature_number,n),
                    kernel_regularizer = L2(layer_regularization),
                ) for n, linear_unit in enumerate([exu_unit] + linear_unit_list)
            ]
        
        self.output_linear = Dense(n_output)    
        self.dropout = Dropout(dropout_ratio)
    
    def __call__(self, x, training = True):
        for linear in self.layer_list:
            x = relu(linear(x), .01)
            if training:
                x = self.dropout(x,training)
        else:
            return self.output_linear(x)