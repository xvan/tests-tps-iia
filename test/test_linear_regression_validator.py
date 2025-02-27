import unittest
import numpy as np

from TestsTPS.linear_regression_validator import LinearRegressionValidator


from  obf.linear_regression import LinearRegression

class TestLinearRegressionValidator(unittest.TestCase):
    def test_init_croaks_without_model(self):
        with self.assertRaises(TypeError):
            LinearRegressionValidator()
        
    def test_validation_croaks_on_model_without_fit(self):    
        class __badLinearRegressionModel_NoFit:
            def predict(self):
                pass        
        
        with self.assertRaises(Exception):
            LinearRegressionValidator(__badLinearRegressionModel_NoFit).validate()
        
    def test_validation_croaks_on_model_without_predict(self):        
        class __badLinearRegressionModel_NoPredict:
            def fit(self):
                pass

        with self.assertRaises(Exception):
            LinearRegressionValidator(__badLinearRegressionModel_NoPredict).validate()
        
        
    def test_validation_croaks_on_model_without_w_after_fit(self):        
        class __badLinearRegressionModel_NoW_AfterFit:
            def fit(self,X,Y):                
                self.b_= None
            def predict(self,X):
                pass
        
        with self.assertRaises(Exception):
            LinearRegressionValidator(__badLinearRegressionModel_NoW_AfterFit).validate()

    def test_validation_croaks_on_model_without_b_after_fit(self):        
        class __badLinearRegressionModel_NoB_AfterFit:
            def fit(self,X,Y):                
                self.w_= None
            def predict(self,X):
                pass
        
        with self.assertRaises(Exception):
            LinearRegressionValidator(__badLinearRegressionModel_NoB_AfterFit).validate()   

    def test_validation_croaks_on_bad_predicton(self):
        class badLinearRegressionModel_SomePredict:                        
            def fit(self,X,Y):
                self.w_ = 3
                self.b_ = 4
                
            def predict(self,X):
                return np.array(X)*self.w_ + self.b_

        with self.assertRaisesRegex(AssertionError, "Failed Test Without Noise"):
            LinearRegressionValidator(badLinearRegressionModel_SomePredict).validate()

    def test_validation_croaks_on_bad_fit(self):
        class badLinearRegressionModel_SomePredict:                        
            def fit(self,X,Y):
                self.w_ = 5.1
                self.b_ = 2
                
            def predict(self,X):
                return np.array(X)*self.w_ + self.b_

        with self.assertRaisesRegex(AssertionError, "Failed Test Without Noise"):
            LinearRegressionValidator(badLinearRegressionModel_SomePredict).validate()

        with self.assertRaisesRegex(AssertionError, "Failed Test Without Noise"):
            LinearRegressionValidator(badLinearRegressionModel_SomePredict).validate()

    def test_round_validation(self):
        validator = LinearRegressionValidator(LinearRegression)
        validator.validate()
        
if __name__ == '__main__':
    unittest.main()
        