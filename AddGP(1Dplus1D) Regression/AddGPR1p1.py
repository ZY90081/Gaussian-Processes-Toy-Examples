
import numpy as np
import matplotlib.pyplot as plt
import math

# Functions
def kernel(x,xstar,para):
    sigmaf = para[0]
    l = 10**para[1]
    K = sigmaf**2 * math.exp(-1/(2*(l**2))*(x-xstar)**2)
    return K

def partial_sigmaf_kernel(x,xstar,para):
    sigmaf = para[0]
    l = 10**para[1]
    PK = 2*sigmaf * math.exp(-1/(2*(l**2))*(x-xstar)**2)
    return PK

def partial_l_kernel(x,xstar,para):
    sigmaf = para[0]
    l = 10**para[1]
    PK = sigmaf**2 * (x-xstar)**2 / (l**3) * math.exp(-1/(2*(l**2))*(x-xstar)**2)
    return PK


def progress(iter,x,fvalue):
    print("iter = %3d: x = %-10s, neg_log_likelihood = %f" %(iter,str(x),fvalue))

def neglog_marginal_likelihood(x,y,para,sigma,N):
    num_tr = len(y)

    para = np.reshape(para, (N, 2))

    K = np.zeros((num_tr, num_tr))
    for n in range(N):
        Ptemp = np.zeros((num_tr, num_tr))
        for i in range(num_tr):
            for j in range(num_tr):
                Ptemp[i][j] = kernel(x[n][i], x[n][j], para[n])
        K = K + Ptemp

    Ky = K + sigma**2*np.eye(num_tr)
    output = 0.5* y @ np.linalg.inv(Ky) @ np.transpose([y]) + 0.5*math.log(np.linalg.det(Ky)) + 0.5*num_tr*math.log(2*np.pi)
    return output



def neg_partiallog_marginal_likelihood(x,y,para,sigma,N):

    num_tr = len(y)
    para = np.reshape(para, (N, 2))

    K = np.zeros((num_tr, num_tr))
    for n in range(N):
        Ptemp = np.zeros((num_tr, num_tr))
        for i in range(num_tr):
            for j in range(num_tr):
                Ptemp[i][j] = kernel(x[n][i], x[n][j], para[n])
        K = K + Ptemp
    Ky = K + sigma ** 2 * np.eye(num_tr)
    alpha = np.linalg.inv(Ky) @ np.transpose([y])

    outputs = np.zeros(2*N)
    for idx in range(2*N):
        if ((idx+1) % 2) == 1:
            #print("{0} is Odd".format(idx+1))
            PK = np.zeros((num_tr, num_tr))
            for i in range(num_tr):
                for j in range(num_tr):
                    PK[i][j] = partial_sigmaf_kernel(x[(idx+2)//2-1][i], x[(idx+2)//2-1][j], para[(idx+2)//2-1])
            PKy = PK + sigma ** 2 * np.eye(num_tr)
            outputs[idx] = -0.5*np.matrix.trace( (alpha @ np.transpose(alpha) - np.linalg.inv(Ky)) @ PKy )
        else:
            #print("{0} is Even".format(idx+1))
            PK = np.zeros((num_tr, num_tr))
            for i in range(num_tr):
                for j in range(num_tr):
                    PK[i][j] = partial_l_kernel(x[(idx+1)//2-1][i], x[(idx+1)//2-1][j], para[(idx+1)//2-1])
            PKy = PK + sigma ** 2 * np.eye(num_tr)
            outputs[idx] = -0.5*np.matrix.trace( (alpha @ np.transpose(alpha) - np.linalg.inv(Ky)) @ PKy )

    return outputs


def marginal_likelihood(x,y,sigma,N):

    GAMMA = 0.001  # step size(learning rate)
    MAX_ITER = 1000 # % maximum number of iterations
    FUNC_TOL = 0.001  # termination tolerance for F(x)
    fvals = [0] # store F(x) values across iterations
    iter = 1  # iterations counter
    para = np.tile(np.array([1,0]), N)  # initial guess

    fvals.append(neglog_marginal_likelihood(x, y, para, sigma,N))
    progress(iter, para, fvals[-1])
    while (iter < MAX_ITER) and (  abs(fvals[-1]-fvals[-2]) > FUNC_TOL):
        iter += 1
        para = para - GAMMA * neg_partiallog_marginal_likelihood(x,y,para,sigma,N)  # gradient descent
        fvals.append(neglog_marginal_likelihood(x,y,para,sigma,N))   # evaluate objective function
        progress(iter, para, fvals[-1])

    return para


'''
# objective function
def objective(x):
    y1 = np.sin(x[0])
    y2 = np.sin((10.0 / 3.0) * x[1])
    #y3 = np.cos((6.0 / 8.0)* x[2] - (np.pi / 3.0))
    return y1+y2
'''


# define range for input
r_min, r_max = -2.6, 7.4
# sample input range uniformly at 0.1 increments
inputs = np.arange(r_min, r_max, 0.2)
xa,xb = np.meshgrid(inputs,inputs)
y = np.sin(xa) + np.sin((10.0 / 3.0) * xb)
fig1 = plt.figure(figsize=(6,6))
cont = plt.contourf(xa,xb,y)
fig1.colorbar(cont, shrink=0.5, aspect=5)

fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='3d')
surf = ax2.plot_surface(xa, xb, y, cmap='viridis', edgecolor='none')
fig2.colorbar(surf, shrink=0.5, aspect=5)

# create training set and test set
total = 50
num_tr = 50
num_te = total - num_tr

# observation error
en = 0.1
# number of functions
N = 2

index_xa = np.random.choice(len(inputs),total,replace=False)
index_xb = np.random.choice(len(inputs),total,replace=False)

index_tr_xa = index_xa[:num_tr]
index_tr_xb = index_xb[:num_tr]
index_te_xa = index_xa[num_tr:]
index_te_xb = index_xb[num_tr:]

TrainingInputs = np.array([inputs[index_tr_xa-1].tolist(),inputs[index_tr_xb-1].tolist()])
TrainingOutputs = np.sin(TrainingInputs[0]) + np.sin((10.0 / 3.0) * TrainingInputs[1]) + np.random.normal(0, en, num_tr)
TestInputs = np.array([inputs[index_te_xa-1].tolist(),inputs[index_te_xb-1].tolist()])
TestOutputs = np.sin(TestInputs[0]) + np.sin((10.0 / 3.0) * TestInputs[1])

#print(TrainingInputs)
print(TrainingOutputs)
ax2.scatter(TrainingInputs[0],TrainingInputs[1],TrainingOutputs,c='r')

fig3 = plt.figure(figsize=(6,6))
ax3 = fig3.add_subplot(111, projection='3d')
train_dots = ax3.scatter(TrainingInputs[0],TrainingInputs[1],TrainingOutputs,c='royalblue')
test_dots = ax3.scatter(TestInputs[0],TestInputs[1],TestOutputs,c='r')

# AddGP

# fitting model
para = marginal_likelihood(TrainingInputs,TrainingOutputs,en,N)
para = np.reshape(para, (N, 2))

# create total test data
TestTotalInputs = []
for i in range(len(inputs)):
    for j in range(len(inputs)):
        TestTotalInputs.append([xa[i][j],xb[i][j]])

TestTotalInputs = np.transpose(np.array(TestTotalInputs))
num_total_test = TestTotalInputs.shape[1]
print(num_total_test)

Pcov = np.zeros((num_total_test, num_total_test))
for n in range(N):
    Ptemp = np.zeros((num_total_test, num_total_test))
    for i in range(num_total_test):
        for j in range(num_total_test):
            Ptemp[i][j] = kernel(TestTotalInputs[n][i], TestTotalInputs[n][j], para[n])
    Pcov = Pcov + Ptemp

fig4 = plt.figure(figsize=(6,6))
[axisX,axisY] = np.meshgrid(np.arange(num_total_test),np.arange(num_total_test))
fig4pcov = plt.contourf(axisX,axisY,Pcov)
plt.axis('equal')
fig4.colorbar(fig4pcov, shrink=0.5, aspect=5)


ktrte = np.zeros((num_tr,num_total_test))
ktrtr = np.zeros((num_tr,num_tr))
for n in range(N):
    Ptemp = np.zeros((num_tr,num_total_test))
    for i in range(num_tr):
        for j in range(num_total_test):
            Ptemp[i][j] = kernel(TrainingInputs[n][i],TestTotalInputs[n][j],para[n])
    ktrte = ktrte + Ptemp


for n in range(N):
    Ptemp = np.zeros((num_tr,num_tr))
    for i in range(num_tr):
        for j in range(num_tr):
            Ptemp[i][j] = kernel(TrainingInputs[n][i],TrainingInputs[n][j],para[n])
    ktrtr = ktrtr + Ptemp

Qavg = np.transpose(ktrte).dot(np.linalg.inv(ktrtr+en**2*np.eye(num_tr))).dot(np.transpose(TrainingOutputs))
Qcov = Pcov - np.transpose(ktrte).dot(np.linalg.inv(ktrtr+en**2*np.eye(num_tr))).dot(ktrte)

fig5 = plt.figure(figsize=(6,6))
[axisX,axisY] = np.meshgrid(np.arange(num_total_test),np.arange(num_total_test))
fig5pcov = plt.contourf(axisX,axisY,Qcov)
plt.axis('equal')
fig5.colorbar(fig5pcov, shrink=0.5, aspect=5)


TestAvg = np.zeros((len(inputs),len(inputs)))
temp = num_total_test
for i in range(len(inputs)):
    for j in range(len(inputs)):
        TestAvg[i][j] = Qavg[num_total_test-temp]
        temp -= 1

fig6 = plt.figure(figsize=(6,6))
ax6 = fig6.add_subplot(111, projection='3d')
ax6.plot_surface(xa, xb, y, rstride=3, cstride=3, linewidth=1,antialiased=True,cmap='viridis', alpha = 0.5)
ax6.plot_surface(xa, xb, TestAvg, rstride=3, cstride=3, linewidth=1,antialiased=True,cmap='inferno', alpha = 1)
plt.show()







