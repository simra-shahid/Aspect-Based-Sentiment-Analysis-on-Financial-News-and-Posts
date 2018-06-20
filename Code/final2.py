
# coding: utf-8

# In[ ]:



from keras import backend as K, regularizers, constraints, initializers
from keras.engine.topology import Layer

class Final2(Layer):

    def __init__(self,
                 Wp_regularizer=None, Wx_regularizer=None,Ws_regularizer=None, Ws_constraint=None,
                 Wp_constraint=None, Wx_constraint=None,b_regularizer=None, b_constraint=None, **kwargs):
        
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.Wp_regularizer = regularizers.get(Wp_regularizer)
        self.Wx_regularizer = regularizers.get(Wx_regularizer)

        self.Wp_constraint = constraints.get(Wp_constraint)
        self.Wx_constraint =constraints.get(Wx_constraint)
         
        self.Ws_regularizer = regularizers.get(Ws_regularizer)
        self.Ws_constraint = constraints.get(Ws_constraint)

        self.b_constraint  = constraints.get(b_constraint)
        self.b_regularizer  = constraints.get(b_regularizer)

        super(Final2, self).__init__(**kwargs)

    def build(self, input_shape ):
        #print(input_shape , "Input")
        #print(input_shape[1][-1] , "Input_1")
        #print(input_shape[1][0] , "Input")

    
        self.Wx = self.add_weight((300,300,),
                                 initializer=self.init,
                                 name='{}_Wx'.format(self.name),
                                 regularizer=self.Wx_regularizer,
                                 constraint=self.Wx_constraint)
       

        
        self.Wp = self.add_weight((300,300,),
                                 initializer=self.init,
                                 name='{}_Wp'.format(self.name),
                                 regularizer=self.Wp_regularizer,
                                 constraint=self.Wp_constraint)
        
         
        self.Ws = self.add_weight((300,27,),
                                 initializer=self.init,
                                 name='{}_Ws'.format(self.name),
                                 regularizer=self.Ws_regularizer,
                                 constraint=self.Ws_constraint)
         
        self.b = self.add_weight((27,),  
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

    

    def call(self, inputs):
        if type(inputs) is not list or len(inputs) <= 1:
              raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
        x = inputs[0]
        t = inputs[1]
        print("r",x.shape)     # (64,300)
       
   
        print("H_all",t.shape)  #(64,11,300)
    
        
       
        
        m_2 = K.dot(t , self.Wx)     # (64,11,300)
        print("m2.shape",m_2.shape)
        
        
        
        m_1 = K.dot(x,self.Wp)  # output= ( 64,300)
        print("m1 ",m_1.shape)
        
        h_final = K.tanh(m_1+m_2)
        
        print(h_final.shape,"h_final" )
        
        out= K.dot(h_final,self.Ws)  # output = (batch_size,1)
        print("OUT",out.shape)
        
        out=out+self.b    
        
        #out=K.tanh(out)
        
        
        #print("Final", out.shape)
        
        return out

    
    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[1][0],27)