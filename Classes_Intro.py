######Introduction to classes########################
class example:
  def __init__(self, x, y):
    self.num1 = x
    self.num2 = y

  def sum(self):
     s = self.num1 + self.num2
     return(s)##or print

  def diff(self):
     d = self.num1-self.num2
     return(d)

  def sum_diff(self):
      k = self.diff()+self.sum()
      return(k)

p1 = example(2, 36)

p1.sum()
p1.diff()
p1.sum_diff()

######Create a class to estimate a regression model#############
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
x , y , coef  = datasets.make_regression(n_samples=100,#number of samples
                                      n_features=1,#number of features
                                      n_informative=1,#number of useful features
                                      noise=10,#bias and standard deviation of the guassian noise
                                      coef=True,#true coefficient used to generated the data
                                      random_state=0) #set for same data points for each run
coef
data_x = x.flatten()
data_y = y.flatten()
plt.scatter(data_x,data_y,c ="blue")


class regression:
    def __init__(self,response,covariate):
        self.x = covariate
        self.y = response
    ######Estimate the parameters######### y = a + bx
    def coefficients(self):
        b = np.sum((self.x-np.mean(self.x))*(self.y-np.mean(self.y)))/np.sum(np.power(self.x-np.mean(self.x),2))
        a = np.mean(self.y) - b*np.mean(self.x)
        return([a,b])
    #####Estimate the fitted values########## y_hat = a_hat + b_hat*x
    def fitted(self):
        y_hat = self.coefficients()[0] + self.coefficients()[1]*self.x
        return(y_hat)
    ####Estimate the residuels############# e = y_hat-y
    def residuals(self):
        res = self.fitted() - self.y
        return(res)
    ####Plot standardized residuals vs fitted values#######
    def diagnostic_plot(self):
        std_res = self.residuals()/np.std(self.residuals())
        print(plt.scatter(r.fitted(),std_res , c="blue"))
    ####Predict function###################################
    def predict(self,new_x):
        y_pred = self.coefficients()[0] + self.coefficients()[1]*new_x
        return(y_pred)


r  = regression(data_y,data_x)
r.coefficients()
r.fitted()
r.residuals()
r.diagnostic_plot()
r.predict(np.array([5,4,-1]))


