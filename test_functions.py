# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:47:48 2023

@author: amrma
"""


import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show


# Here we setup a user defined exception if there is any loading error, 
#while reading from excel into pandas dataframe.
class Mycustomized_exception(Exception):
    def __init__(self, message='this is a customexception'):
        self.message=message
        super().__init__(self.message)


#Loading files
path='C:/Users/amrma/Downloads/iu/python/dataset/raw/'

ideal_path=path+'ideal.csv'
train_path=path+'train.csv'
test_path=path+'test.csv'

try:
    ideal = pd.read_csv(ideal_path)
    train = pd.read_csv(train_path)
    testdata = pd.read_csv(test_path)
except:
    raise Mycustomized_exception('check the path and file name again')


class Twodimentional_data_x_and_y():
    """I created a class that can separately retrieve or visualize any function from any table,
    just specify the type(or table) and the name of the function (or column), this is important for 
    manual check and see function trend."""
    def __init__(self, _type, function):
        self._type=_type
        self.function=function
    
    def retrieve(self):
        if self._type=='train':
            return train[['x', self.function]]
        if self._type=='test':
            return testdata[['x', 'y']]
        if self._type=='ideal':
            return ideal[['x', self.function]]

            
    def visualize(self):
        x=self.retrieve()['x']
        y=(self.retrieve()).iloc[:,-1]

        # Create a scatter plot
        p = figure(title="Scatter Plot of " + self.function + " from "+ self._type +" table",
                   x_axis_label="X-axis", y_axis_label=self.function)

        # Add data points to the plot
        p.circle(x, y, size=10, color="blue", legend_label= self.function +" Data Points")

        p.legend.title = "Legend"
        p.legend.label_text_font_size = "13pt"
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "14pt"

        # Show the plot
        return show(p)


def calculate_mean_square_error(y1, y2):
    """y1 and y2 are columns or series of values"""
    return ((y1-y2)**2).mean()

def deviation(y3, y4):
    """y1 and y2 are columns or series of values"""
    return np.sqrt((y3-y4)**2)

class Testcalc(unittest.TestCase):   #inherits its properties

    def test_calculate_mean_square_error(self):
        y1=pd.Series([1,2,3])
        y2=pd.Series([2,3,4])
        result=calculate_mean_square_error(y1,y2)
        self.assertEqual(result, 1.0)
        
        
    def test_deviation(self):
        y3=pd.Series([5,2,3])
        y4=pd.Series([2,3,4])    
        result=list(deviation(y3,y4))
        self.assertEqual(result, list(pd.Series([3.0,1.0,1.0])))

        
def detect_match(train, ideal):
    for train_y in train.columns[1:]:# y1, y2 ...
        di={}                           #make pairs of ideal 
        for ideal_y in ideal.columns[1:]:
            di[ideal_y]=calculate_mean_square_error(train[train_y], ideal[ideal_y])
        min_value=min(di.values())
        ideal_function = [key for key, value in di.items() if value == min_value]
        print(f"function {ideal_function} best fit {train_y} with minimum value equal to {min_value}")        
        
      

def visualize_2functions(functiontrain, functionideal):
    """takes the first argument the name of the function from tain table 
    and the second argument the name of the function from the ideal table
    (both are strings) """
    x=ideal[['x']]
    y_train=train[functiontrain]
    y_ideal=ideal[functionideal]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_train, color='darkorange', label=f'{functiontrain} from train')
    plt.scatter(x, y_ideal, color='navy', marker='.', label=f'{functionideal} from ideal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'matchine between function {functiontrain}, and {functionideal}')
    plt.legend()
    plt.show()   
 

# deviation visualization to see deviation change against oredered data points
def visualize_deviation(modelname, maximum_limits_for_x_and_y):
    """visualize deviation value against sorted data point, is it smooth increase or
    there may has an abrupt change? 
    the first argument takes string, the second takes list"""
    column='deviation_from_'+modelname+'_model' #like deviation_from_y13_model
    series=(testdata.sort_values(column)[[column]]).reset_index() #picking the column, sort it

    #visualize
    ax=sns.lineplot(x=range(100), y=series['deviation_from_'+modelname+'_model'])
    plt.xlabel('order of sorted data point')
    plt.ylabel('deviation_from_'+modelname+'_model')
    plt.title('Deviation change')
    ax.set_xlim(1, maximum_limits_for_x_and_y[0])  # Set x-axis limits
    ax.set_ylim(0, maximum_limits_for_x_and_y[1])  # Set y-axis limits
    #the function visualize the deviation change of given model

def visualize_test_points_to_function(func):
    """ the ideal function that points mapped to (string) like 'y13'   """
    sns.scatterplot(x='x', y='y',  data=testdata[testdata['function']==func], palette='Set1')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title(f'Scatter Plot of points mapped to function {func}')
    plt.show()


def visualize_test_points_to_function(func):
    """ the ideal function that points mapped to (string) like 'y13'   """
    sns.scatterplot(x='x', y='y',  data=testdata[testdata['function']==func], palette='Set1')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title(f'Scatter Plot of points mapped to function {func}')
    plt.show()


def visualize_all_test_points_together(hue, data):
    sns.scatterplot(x='x', y='y', hue='function', data=testdata, palette='Set1')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter all points together')
    plt.show()     
    
    
if __name__ == "__main__":
    unittest.main()
    # test_case_one()
    # print('all tests passed')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    