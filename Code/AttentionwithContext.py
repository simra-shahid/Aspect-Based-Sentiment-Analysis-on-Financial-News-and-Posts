
# coding: utf-8

# In[3]:


from keras import backend as K, regularizers, constraints, initializers
from keras.engine.topology import Layer
class AttentionWithContext(Layer):

    def __init__(self,
                 W_regularizer=None, wT_regularizer=None, W_constraint=None, wT_constraint=None,  **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.wT_regularizer = regularizers.get(wT_regularizer)



        self.W_constraint = constraints.get(W_constraint)
        self.wT_constraint = constraints.get(wT_constraint)


        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self,input_shape):

        self.W = self.add_weight((600,600,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
       

        self.wT = self.add_weight((600,1,),
                                 initializer=self.init,
                                 name='{}_wT'.format(self.name),
                                 regularizer=self.wT_regularizer,
                                 constraint=self.wT_constraint)
        


    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
              raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
                
        H_all = inputs[0]   #(batch_size,300)
        concat = inputs[1]  #(batch_size,600)
        
        
        M1 = K.dot(concat,self.W)    # output = (batch_size,600)

        M1=K.tanh(M1)
        print("M1",M1.shape)

        M = K.dot(M1,self.wT)    # output=(batch_size,1)
        print("M",M.shape)

        
        a = K.exp(M)
        

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        
        print("alpha", a.shape)
        
        #a = K.expand_dims(a)
        
        #print("alpha", a.shape)
        
        weighted_input = H_all * a
        
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (300,1)