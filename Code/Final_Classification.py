
from keras import backend as K, regularizers, constraints, initializers
from keras.engine.topology import Layer

class FinalLayer(Layer):

    def __init__(self,Wl_regularizer=None, Wl_constraint=None, b_regularizer=None, b_constraint=None,  **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.Wl_regularizer = regularizers.get(Wl_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        
        self.Wl_constraint = constraints.get(Wl_constraint)
        self.b_constraint  = constraints.get(b_constraint)

        
        super(FinalLayer, self).__init__(**kwargs)

    def build(self,input_shape):
       
        self.Wl = self.add_weight((600,1),
                                 initializer=self.init,
                                 name='{}_Wl'.format(self.name),
                                 regularizer=self.Wl_regularizer,
                                 constraint=self.Wl_constraint)
        self.b = self.add_weight((1,),  # CHECK
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        super(FinalLayer, self).build(input_shape)  # Be sure to call this somewhere!



    def call(self,d, mask=None):
        
        
        x =  K.dot(d,self.Wl)
        
        x = x + self.b
        
        return K.tanh(x)

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0],1)  # Check

