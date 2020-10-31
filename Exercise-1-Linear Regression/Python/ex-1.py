#----------------------------------------------------------------------------------------
## Linear Regression with one variable
#----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#Reading data
datafile = 'data/ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True)
# X-->inputs Y-->outputs
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
#Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10)
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

#hypothesis function without victorization
def h2(theta,X):
    arr = np.zeros((X.shape[0],1))
    for i in range (X.shape[0]):
        arr[i][0] = theta[0][0]*X[i][0] + theta[1][0]*X[i][1]
    print(arr)

#hypothesis function vicorized
def h(theta,X):
    return np.dot(X,theta)

#Cost function
def computeCost(theta, X, y):
    m = y.size # number of training examples
    squareSum = np.dot ((h(theta,X)-y).T , (h(theta,X)-y) )
    J = float( (1/(2*m)) * squareSum ) 
    return J

#testing
initial_theta = np.zeros((X.shape[1],1))
print( computeCost(initial_theta,X,y) )

#Gradiant Descent
iterations = 1500
alpha = 0.01
def Gradiantdescent(X, initial_theta):
    m = y.size
    theta = initial_theta
    jvec = [] #all cost functions
    thetahistory = [] #theta history
    for _ in range(iterations):
        tmptheta = theta
        jvec.append(computeCost(theta,X,y))

        thetahistory.append(list(theta[:,0]))
        #Simultaneously updating theta values
        for j in range(len(tmptheta)):
            tmptheta[j] = theta[j] - (alpha/m) * np.sum( (h(theta,X) - y)*np.array(X[:,j]).reshape(m,1) )
        theta = tmptheta
    return theta, thetahistory, jvec

#testing
initial_theta = np.zeros((X.shape[1],1))
theta, thetahistory, jvec = Gradiantdescent(X,initial_theta)

def plotConvergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot( range(len(jvec)) ,jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*iterations,1.05*iterations])
    #dummy = plt.ylim([4,8])


plotConvergence(jvec)
dummy = plt.ylim([4,7])
plt.show()

#Plot the line on top of the data to ensure it looks correct
def myfit(xval):
    return theta[0] + theta[1]*xval
plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
plt.plot(X[:,1],myfit(X[:,1]),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()

#plotting 3D

#Import necessary matplotlib tools for 3d plots
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

xvals = np.arange(-10,10,.5)
yvals = np.arange(-1,4,.1)
myxs, myys, myzs = [], [], []
for a in xvals:
    for b in yvals:
        myxs.append(a)
        myys.append(b)
        myzs.append( computeCost(np.array([[a], [b]]),X,y) )

scat = ax.scatter(myxs,myys,myzs,c=np.abs(myzs),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel(r'$\theta_0$',fontsize=30)
plt.ylabel(r'$\theta_1$',fontsize=30)
plt.title('Cost (Minimization Path Shown in Blue)',fontsize=30)
plt.plot( [x[0] for x in thetahistory] , [x[1] for x in thetahistory] , jvec , 'bo-' )
plt.show()

#----------------------------------------------------------------------------------------
## Linear Regression with multiple variables
#----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

datafile = 'data/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size 
X = np.insert(X,0,1,axis=1)

#Plotting data
plt.grid(True)
plt.xlim([-100,5000]) #to show how much area of houses data far from each value
dummy = plt.hist(X[:,0],label = 'col1')
dummy = plt.hist(X[:,1],label = 'col2')
dummy = plt.hist(X[:,2],label = 'col3')
plt.title('Clearly we need feature normalization.')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

#Feature normalizing --> (value-mean)/std
XNormalized = X.copy()
means, stds = [], []
for i in range(XNormalized.shape[1]):
    means.append( np.mean(XNormalized[:,i]) )
    stds.append( np.std(XNormalized[:,i]) )
    #Skip the first column we don't need to normalize this
    if not i: continue
    XNormalized[:,i] = (XNormalized[:,i] - means[-1])/stds[-1]

#Plotting data after Features normalized
plt.grid(True)
plt.xlim([-5,5])
dummy = plt.hist(XNormalized[:,0],label = 'col1')
dummy = plt.hist(XNormalized[:,1],label = 'col2')
dummy = plt.hist(XNormalized[:,2],label = 'col3')
plt.title('Features Normalized')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

#Run gradiant descent on multiple variables
initial_theta = np.zeros((XNormalized.shape[1],1))
theta, thetahistory, jvec = Gradiantdescent(XNormalized,initial_theta)

#Plot convergence of cost function:
plotConvergence(jvec)
plt.show()

#checking on values
print("Final result theta parameters: \n",theta)
print("Check of result: What is price of house with 1650 square feet and 3 bedrooms?")
test = np.array([1650.,3.])
ytestscaled = [ (test[x]-means[x+1])/stds[x+1]     for x in range( len(test) ) ]        #Apply feature normalization
ytestscaled.insert(0,1)
print("$%0.2f" % float(h(theta,ytestscaled)))

#Normal Equation
from numpy.linalg import inv
def normalEquation(X,y):
    return np.dot( np.dot( inv(np.dot(X.T,X)), X.T ) , y)

#Test on normal equation
print("Normal equation prediction for price of house with 1650 square feet and 3 bedrooms")
print("$%0.2f" % float( h( normalEquation(X,y),[1,1650.,3] )))