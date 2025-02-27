import unittest
import numpy as np

from TestsTPS.linear_regression_validator import LinearRegressionValidator


from solutions import LinearRegression

class TestLinearRegressionValidator(unittest.TestCase):
    def test_init_crocks_without_model(self):
        self.assertRaises(TypeError, lambda: LinearRegressionValidator())
        
    def test_validation_crocks_on_model_without_fit(self):
    
        class __badLinearRegressionModel_NoFit:
            def predict(self):
                pass

        self.assertRaises(ValueError, lambda: LinearRegressionValidator(__badLinearRegressionModel_NoFit).validate(),"model must have a fit method")
        
    def test_validation_crocks_on_model_without_predict(self):        
        class __badLinearRegressionModel_NoPredict:
            def fit(self):
                pass

        self.assertRaises(ValueError, lambda: LinearRegressionValidator(__badLinearRegressionModel_NoPredict).validate(),"model must have a predict method")
        
        
    def test_validation_crocks_on_model_without_w_after_fit(self):        
        class __badLinearRegressionModel_NoW_AfterFit:
            def fit(self,X,Y):
                self.w_= None
                self.b_= None
            def predict(self,X):
                pass
                            
        self.assertRaises(ValueError, lambda: LinearRegressionValidator(__badLinearRegressionModel_NoW_AfterFit).validate())         

    def test_validation_crocks_on_bad_model(self):
        class badLinearRegressionModel_SomePredict:                        
            def fit(self,X,Y):
                self.w_ = 3
                self.b_ = 4
                
            def predict(self,X):
                return np.array(X)*self.w_ + self.b_
            
        validator = LinearRegressionValidator(badLinearRegressionModel_SomePredict)
        validator.validate()
        
    def test_round_validation(self):
        validator = LinearRegressionValidator(LinearRegression)
        validator.validate()        
        
if __name__ == '__main__':
    unittest.main()
        