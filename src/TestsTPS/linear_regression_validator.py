import numpy as np

class LinearRegressionValidator():
    def __init__(self, modelImpl):            
        self.modelImpl = modelImpl
        
    def validate(self):
        self._validate_interface()
        #self._validate_without_noise()
        self._validate_with_noise()
        
    def _validate_interface(self):              
        model = self.modelImpl()
        model.fit(np.array([1,2,3]), np.array([1,2,3]))                
        model.w_
        model.b_
        _ = model.predict(np.array([4,5,6]))
        
    # def _validate_without_noise(self):
    #     A=np.arctan(np.random.uniform(-np.pi/3,np.pi/3,10))
    #     B=np.random.uniform(-20,20,10)
        
    #     for(a,b) in zip(A,B):
    #         X=np.random.uniform(-100,100,100)
    #         a=5
    #         b=2
    #         y=a*X+b
            
    #         model=self.modelImpl()
            
    #         model.fit(X,y)
    #         y_pred=model.predict(X)
            
    #         assert np.allclose(y,y_pred,atol=1e-2), (y - y_pred) #"Failed Test Without Noise %.3e" % np.max(np.abs(y - y_pred))

    def _validate_without_noise(self):
        w=5
        b=2

        N=5000
        sigma = 1
        
        training_x = np.random.uniform(-20,20,N)
        training_y = w*training_x + b + np.random.normal(0, sigma, N)

        model = self.modelImpl()
        model.fit(training_x,training_y)        
        assert (w-model.w_)**2 + (b-model.b_)**2 < 1e-2, "Failed Test With Noise w: %f, b: %f" % (model.w_, model.b_)

        
    def _validate_with_noise(self):
        w=5
        b=2

        N=5000
        sigma = 1
        
        training_x = np.random.uniform(-20,20,N)
        training_y = w*training_x + b + np.random.normal(0, sigma, N)

        model = self.modelImpl()
        model.fit(training_x,training_y)        
        assert (w-model.w_)**2 + (b-model.b_)**2 < 1e-2, "Failed Test With Noise w: %f, b: %f" % (model.w_, model.b_)

        

        
        #print(np.arctan(np.pi/4))        
        #test_model.validate()
        