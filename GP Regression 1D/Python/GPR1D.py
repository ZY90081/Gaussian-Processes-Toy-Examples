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

def neglog_marginal_likelihood(x,y,para,sigma):
    N = len(y)
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i][j] = kernel(x[i],x[j],para)

    Ky = K + sigma**2*np.eye(N)
    output = 0.5* y @ np.linalg.inv(Ky) @ np.transpose([y]) + 0.5*math.log(np.linalg.det(Ky)) + 0.5*N*math.log(2*np.pi)
    return output

def neg_partiallog_marginal_likelihood(x,y,para,sigma):
    N = len(y)
    K = np.zeros((N, N))
    PK1 = np.zeros((N, N))
    PK2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = kernel(x[i],x[j],para)
            PK1[i][j] = partial_sigmaf_kernel(x[i],x[j],para)
            PK2[i][j] = partial_l_kernel(x[i],x[j],para)

    Ky =  K + sigma**2*np.eye(N)
    PKy1 = PK1 + sigma**2*np.eye(N)
    PKy2 = PK2 + sigma**2*np.eye(N)
    alpha = np.linalg.inv(Ky) @ np.transpose([y])
    output = np.zeros(2)
    output[0] = -0.5*np.matrix.trace( (alpha @ np.transpose(alpha) - np.linalg.inv(Ky)) @ PKy1 )
    output[1] = -0.5*np.matrix.trace( (alpha @ np.transpose(alpha) - np.linalg.inv(Ky)) @ PKy2 )

    return output

def progress(iter,x,fvalue):
    print("iter = %3d: x = %-10s, neg_log_likelihood = %f" %(iter,str(x),fvalue))

def marginal_likelihood(x,y,sigma):

    GAMMA = 0.001  # step size(learning rate)
    MAX_ITER = 1000 # % maximum number of iterations
    FUNC_TOL = 0.1  # termination tolerance for F(x)

    fvals = [] # store F(x) values across iterations

    iter = 1  # iterations counter
    para = [1, 0] # initial guess
    fvals.append(neglog_marginal_likelihood(x,y,para,sigma))
    progress(iter, para, fvals[-1])
    while (iter < MAX_ITER) and (fvals[-1] > FUNC_TOL):
        iter += 1
        para = para - GAMMA * neg_partiallog_marginal_likelihood(x,y,para,sigma)  # gradient descent
        fvals.append(neglog_marginal_likelihood(x,y,para,sigma))   # evaluate objective function
        progress(iter, para, fvals[-1])

    return para





xo = np.arange(-4, 4, 0.1)
yo = np.sin(3*xo)-2*np.sin(xo+np.pi/2) # Real function.
en = 0.001 # standard derivation of error.
x = np.array([-3.5,-3.1,-3,-2.5,-1.5,-1.1,1,1.1,1.3,1.9,2.1,2.9,3.3,3.8])
y = np.sin(3*x)-2*np.sin(x+np.pi/2) + np.random.normal(0, en, len(x)) # Training data.
N = len(x) # number of training data.

fig1 = plt.figure(figsize=(6,6))
plt.plot(xo,yo)
plt.scatter(x,y,c='r')
plt.show()

para = marginal_likelihood(x,y,en)

itv = np.arange(-10, 10, 0.1)
n = len(itv)
Pcov = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Pcov[i][j] = kernel(itv[i],itv[j],para)
fig2 = plt.figure(figsize=(6,6))
[X,Y] = np.meshgrid(itv,itv)
fig2cont = plt.contourf(X,Y,Pcov)
plt.axis('equal')
fig2.colorbar(fig2cont, shrink=0.5, aspect=5)
plt.show()

Pvar = kernel(0,0,para) # prior variance of f*
Pstd = math.sqrt(Pvar)  #prior standard deviation
Pavg = np.zeros(n)  # prior mean.
Pgp = np.random.multivariate_normal(Pavg,Pcov,3) # draw three random function from prior gp.

fig3 = plt.figure(figsize=(6,6))
plt.plot(itv,Pavg,'k-')
plt.fill_between(itv, Pavg+2*Pstd, Pavg-2*Pstd)
plt.plot(itv,Pgp[0],'r')
plt.plot(itv,Pgp[1],'y')
plt.plot(itv,Pgp[2],'m')
plt.show()

ktrte = np.zeros((N,n))
ktrtr = np.zeros((N,N))
for i in range(n):
    for j in range(N):
        ktrte[j][i] = kernel(itv[i],x[j],para) # cov of training data and testing data
for i in range(N):
    for j in range(N):
        ktrtr[i][j] = kernel(x[i],x[j],para)   # cov of training data



Qavg = np.transpose(ktrte).dot(np.linalg.inv(ktrtr+en**2*np.eye(N))).dot(np.transpose(y))
Qcov = Pcov - np.transpose(ktrte).dot(np.linalg.inv(ktrtr+en**2*np.eye(N))).dot(ktrte)

fig4 = plt.figure(figsize=(6,6))
fig4cont = plt.contourf(X,Y,Qcov)
plt.axis('equal')
fig4.colorbar(fig4cont, shrink=0.5, aspect=5)
plt.show()

Qvar = np.diag(Qcov)  # Posterior variances of each testing points.
Qstd = np.sqrt(Qvar)
Qgp = np.random.multivariate_normal(Qavg,Qcov,3) # draw three random function from posterior gp.

fig5 = plt.figure(figsize=(6,6))
plt.plot(itv,Qavg,'k-')
plt.fill_between(itv, Qavg+2*Qstd, Qavg-2*Qstd)
plt.plot(x,y,'ko')
plt.plot(itv,Qgp[0],'r')
plt.plot(itv,Qgp[1],'y')
plt.plot(itv,Qgp[2],'m')
plt.show()
