import numpy as np

class LinearRegressionValidator():
    def __init__(self, modelImpl):            
        self.modelImpl = modelImpl
        
    def validate(self):
        self._validate_interface()
        self._validate_without_noise()
        self._validate_with_noise()
        
    def _validate_interface(self):
        if self.modelImpl is None:
            raise ValueError("model cannot be None")
        if not hasattr(self.modelImpl, 'fit'):
            raise ValueError("model must have a fit method")            
        if not hasattr(self.modelImpl, 'predict'):
            raise ValueError("model must have a predict method")        
        model = self.modelImpl()
        model.fit(np.array([1,2,3]), np.array([1,2,3]))                
        if not hasattr(model, 'w_'):
            raise ValueError("model must expose a w_ attribute after fitting")
        if not hasattr(model, 'b_'):
            raise ValueError("model must expose a b_ attribute after fitting")
        
    def _validate_without_noise(self):
        A=np.arctan(np.random.uniform(-np.pi/2,np.pi/2,10))
        B=np.random.uniform(-20,20,10)
        
        for(a,b) in zip(A,B):
            X=np.random.uniform(-100,100,100)
            y=a*X+b
            
            model=self.modelImpl()
            
            model.fit(X,y)
            y_pred=model.predict(X)
            assert np.allclose(y,y_pred,atol=1e-5), "Failed Test Without Noise"        
        
    def _validate_with_noise(self):
        w=5
        b=2
        
        training_x = np.random.uniform(-20,20,5000)
        training_y = w*training_x + b

        testing_x = np.random.uniform(-20,20,5000)
        testing_y = w * testing_x + b

        np.random.normal()




        raise NotImplementedError()
        
        

        #print(np.arctan(np.pi/4))        
        #test_model.validate()
        