#!/bin/sh
PyObfuscator --output-filename obf/linear_regression.py  solutions/linear_regression.py --level 6 -n "LinearRegression:LinearRegression" "LinearRegressionGD:LinearRegressionGD" "fit:fit" "predict:predict" "w_:w_" "b_:b_"
