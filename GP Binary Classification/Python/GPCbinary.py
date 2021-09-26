import numpy as np
import matplotlib.pyplot as plt
import math

# FUNCTIONS
def ARDkernel(x1,x2,para):
    sf = para[0]
    dim = len(x1)
    s = 0
    for i in range(dim):
        s = s + para[i+1]*(x1[i]-x2[i])**2
    k = sf**2 * math.exp(-0.5*s)
    return k

def PDe_ARDkernel_sf(x1,x2,para):
    sf = para[0]
    dim = len(x1)
    s = 0
    for i in range(dim):
        s = s + para[i + 1] * (x1[i] - x2[i]) ** 2
    k = 2 * sf * math.exp(-0.5 * s)
    return k

def PDe_ARDkernel_alpha(x1,x2,para,q):
    sf = para[0]
    dim = len(x1)
    s = 0
    for i in range(dim):
        s = s + para[i + 1] * (x1[i] - x2[i]) ** 2
    k = -0.5 * sf ** 2 * math.exp(-0.5 * s) * (x1[q-1] - x2[q-1])**2
    return k

def K_CovMatrix(N,trainx,para):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = ARDkernel(trainx[:, [i]],trainx[:, [j]], para)
    return K

def Kstar_CovMatrix(N,M,trainx,testx,para):
    K = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            K[i][j] = ARDkernel(trainx[:, [i]],testx[:, [j]], para)
    return K

def Kstar2_CovMatrix(M,testx,para):
    K = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            K[i][j] = ARDkernel(testx[:, [i]],testx[:, [j]], para)
    return K

def PDeK_CovMatrix_sf(N,trainx,para):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = PDe_ARDkernel_sf(trainx[:, [i]],trainx[:, [j]], para)
    return K

def PDeK_CovMatrix_alpha(N,trainx,para,q):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = PDe_ARDkernel_alpha(trainx[:, [i]],trainx[:, [j]], para,q)
    return K


def Logistic(y,f):
    return 1/(1+math.exp(-y*f))

def LogLogistic(y,f):
    return -math.log(1+math.exp(-y*f))

def De1_LogLogistic(y,f):
    return (y+1)/2 - 1/(1+math.exp(-f))

def De2_LogLogistic(f):
    return -1/(1+math.exp(-f))*(1-1/(1+math.exp(-f)))

def De3_LogLogistic(f):
    return 2*math.exp(-f)/(1+math.exp(-f))**3 - math.exp(-f)/(1+math.exp(-f))**2

def Neg_Log_Posterior(F,args):
    (trainy, trainx, Para) = args
    N = len(trainy)
    K = K_CovMatrix(N, trainx, Para)
    loglike = 0
    for i in range(N):
        loglike = loglike + LogLogistic(trainy[i], F[i])

    return 0.5*np.transpose(F)@np.linalg.inv(K)@F + 0.5*math.log(np.linalg.det(K)) + 0.5*N*math.log(2*np.pi) - loglike

def PDe1_Neg_Log_Posterior(F,args):
    (trainy, trainx, Para) = args
    N = len(trainy)
    K = K_CovMatrix(N, trainx, Para)
    de1_log_like_list = []
    for i in range(N):
        de1_log_like_list.append(De1_LogLogistic(trainy[i], F[i]))
    de1_log_like_list = np.array(de1_log_like_list)
    return np.linalg.inv(K)@F - np.transpose(de1_log_like_list)

def PDe2_Neg_Log_Posterior(F,args):
    (trainy, trainx, Para) = args
    N = len(trainy)
    K = K_CovMatrix(N, trainx, Para)
    de2_log_like_list = []
    for i in range(N):
        de2_log_like_list.append(De2_LogLogistic(F[i]))
    W = - np.diag(de2_log_like_list)
    return np.linalg.inv(K) + W


def Neg_LogMar_Likelihood(Para,args):
    (trainy, trainx, F) = args

    N = len(trainy)
    K = K_CovMatrix(N, trainx, Para)
    de2_log_like_list = []
    for i in range(N):
        de2_log_like_list.append(De2_LogLogistic(F[i]))
    W = - np.diag(de2_log_like_list)
    B = np.identity(N) + np.sqrt(W) @ K @ np.sqrt(W)
    loglike = 0
    for i in range(N):
        loglike = loglike + LogLogistic(trainy[i], F[i])

    output = 0.5*np.transpose(F)@np.linalg.inv(K)@F - loglike + 0.5*math.log(np.linalg.det(B))
    '''
    de2_log_like_list = []
    N = len(trainy)
    for i in range(N):
        de2_log_like_list.append(De2_LogLogistic(F[i]))
    W = - np.diag(de2_log_like_list)
    K = K_CovMatrix(N, trainx, Para)
    B = np.identity(N) + np.sqrt(W) @ K @ np.sqrt(W)
    L = np.linalg.cholesky(B)
    de1_log_like_list = []
    for i in range(N):
        de1_log_like_list.append(De1_LogLogistic(trainy[i], F[i]))
    de1_log_like_list = np.array(de1_log_like_list)
    b = W @ F + np.transpose(de1_log_like_list)
    a = b - np.linalg.solve(np.sqrt(W) @ np.transpose(L), (np.linalg.solve(L, np.sqrt(W) @ K @ b)))
    loglike = 0
    for i in range(N):
        loglike = loglike + LogLogistic(trainy[i],F[i])

    return 0.5* np.transpose(a)@F - loglike + sum(np.log(np.diag(L)))
    '''
    return output

def PDe_Neg_LogMar_Likelihood(Para,args):
    (trainy, trainx, F) = args
    N = len(trainy)
    NumofPara = len(Para)
    de2_log_like_list = []
    for i in range(N):
        de2_log_like_list.append(De2_LogLogistic(F[i]))
    W = - np.diag(de2_log_like_list)
    K = K_CovMatrix(N, trainx, Para)
    B = np.identity(N) + np.sqrt(W) @ K @ np.sqrt(W)
    DrK_sf = PDeK_CovMatrix_sf(N, trainx, Para)
    output = []
    output.append( -0.5*np.transpose(F)@(np.linalg.inv(K)@DrK_sf@np.linalg.inv(K))@F + 0.5*np.trace(np.linalg.inv(B)@W@DrK_sf) )
    for i in range(1,NumofPara):
        Drk_alpha = PDeK_CovMatrix_alpha(N,trainx,Para,i)
        output.append( -0.5*np.transpose(F)@(np.linalg.inv(K)@Drk_alpha@np.linalg.inv(K))@F + 0.5*np.trace(np.linalg.inv(B)@W@Drk_alpha) )


    '''
    de2_log_like_list = []
    de3_log_like_list = []
    N = len(trainy)
    for i in range(N):
        de2_log_like_list.append(De2_LogLogistic(F[i]))
        de3_log_like_list.append(De3_LogLogistic(F[i]))
    de3_log_like_list = np.array(de3_log_like_list)
    W = - np.diag(de2_log_like_list)
    K = K_CovMatrix(N, trainx, Para)
    B = np.identity(N) + np.sqrt(W) @ K @ np.sqrt(W)
    L = np.linalg.cholesky(B)
    de1_log_like_list = []
    for i in range(N):
        de1_log_like_list.append(De1_LogLogistic(trainy[i], F[i]))
    de1_log_like_list = np.array(de1_log_like_list)
    b = W @ F + np.transpose(de1_log_like_list)
    a = b - np.linalg.solve(np.sqrt(W) @ np.transpose(L), (np.linalg.solve(L, np.sqrt(W) @ K @ b)))

    R = np.linalg.solve(np.sqrt(W)@np.transpose(L), np.linalg.solve(L,np.sqrt(W)) )
    C = np.linalg.solve(L,np.sqrt(W)@K)
    s2 = -0.5*np.diag(np.diag(K)-np.diag(np.transpose(C)@C))@np.transpose(de3_log_like_list)

    NumofPara = len(Para)
    output = []
    for i in range(NumofPara):
        if i == 0:
            C = PDeK_CovMatrix_sf(N,trainx,Para)
        else:
            C = PDeK_CovMatrix_alpha(N,trainx,Para,i)
        s1 = 0.5*np.transpose(a)@C@a - 0.5*np.trace(R*C)
        bb = C @ np.transpose(de1_log_like_list)
        s3 = bb - K @ R @ bb
        output.append(-s1 - np.transpose(s2)@s3)
    '''
    return np.array(output)


def OPT_GradientDescent(Fun,J,x0,MaxIter,Termination,Step,args):
    fvals = []
    iter = 1
    x = x0
    fvals.append(Fun(x,args))
    flag = 1
    while (iter <= MaxIter) and (flag == 1):
        iter = iter + 1
        delta = Step*J(x,args)
        x = x - delta
        fvals.append(Fun(x, args))
        #print("The negative log marginal likelihood is {}.".format(fvals[-1]))
        if np.linalg.norm(delta)<Termination:
            flag = 0

    return x,fvals[-1]

def OPT_Newton(Fun,J,H,x0,MaxIter,Termination,Step,args):
    fvals = []
    iter = 1
    x = x0
    dim = len(x)
    fvals.append(Fun(x,args))
    flag = 1
    while (iter <= MaxIter) and (flag == 1):
        iter = iter + 1
        e,v = np.linalg.eig(H(x,args))
        if any(e<0):
            MU = -min(e)+0.001
        else:
            MU = 0
        delta = Step* np.linalg.inv(H(x,args)+MU*np.identity(dim)) @ J(x,args)
        x = x - delta
        fvals.append(Fun(x, args))
        #print("The negative log posterior is {}.".format(fvals[-1]))
        if np.linalg.norm(delta)<Termination:
            flag = 0

    return x, fvals[-1]

# generate data

NumofClass = 2
NumofTrain_each = 25
NumofTrain = NumofTrain_each * NumofClass
NumofTest_each = 5
NumofTest = NumofTest_each * NumofClass

Mu_c1 = [2,5]    # mean of classes
Mu_c2 = [5,-1]

Cov_c1 = [[6,0],[0,10]]   # covariance of classes
Cov_c2 = [[10,5],[5,10]]

x_c1, y_c1 = np.random.multivariate_normal(Mu_c1, Cov_c1, NumofTrain_each).T
x_c2, y_c2 = np.random.multivariate_normal(Mu_c2, Cov_c2, NumofTrain_each).T

TrainX = np.array([x_c1.tolist()+x_c2.tolist(),y_c1.tolist()+y_c2.tolist()])
TrainY = [1]*NumofTrain_each + [-1]*NumofTrain_each

x_c1_te, y_c1_te = np.random.multivariate_normal(Mu_c1, Cov_c1, NumofTest_each).T
x_c2_te, y_c2_te = np.random.multivariate_normal(Mu_c2, Cov_c2, NumofTest_each).T

TestX = np.array([x_c1_te.tolist()+x_c2_te.tolist(),y_c1_te.tolist()+y_c2_te.tolist()])
TestY = [1]*NumofTest_each + [-1]*NumofTest_each

fig1 = plt.figure(figsize=(6,6))
plt.plot(x_c1, y_c1, 'oc')
plt.plot(x_c2, y_c2, 'oy')
plt.plot(TestX[0], TestX[1], '*r',Markersize=10)



#############################################GP Binary Classification
Dim = TrainX.shape[0]   # dimension of inputs
NumofIteration = 5
InitialPara = [1] + [1]*Dim  # initial hyperparameters
InitialF = [0]*NumofTrain    # initial latent function

Para = InitialPara
F = InitialF

while NumofIteration>0:
    # part1: optimize posterior
    F, fun = OPT_Newton(Neg_Log_Posterior,PDe1_Neg_Log_Posterior,PDe2_Neg_Log_Posterior,F,1000,1e-6,0.2,(TrainY,TrainX,Para))
    print("The negative log posterior is {}.".format(fun))
    print("----------")
    '''
    flag = 1
    # part1: optimize posterior 
    while flag==1:
        de2_log_like_list = []
        for i in range(NumofTrain):
            de2_log_like_list.append(De2_LogLogistic(F[i]))
        W = - np.diag(de2_log_like_list)
        K = K_CovMatrix(NumofTrain,TrainX,Para)
        B = np.identity(NumofTrain) + np.sqrt(W)@K@np.sqrt(W)
        L = np.linalg.cholesky(B)
        de1_log_like_list = []
        for i in range(NumofTrain):
            de1_log_like_list.append(De1_LogLogistic(TrainY[i],F[i]))
        de1_log_like_list = np.array(de1_log_like_list)
        b = W@F + np.transpose(de1_log_like_list)
        a = b - np.linalg.solve( np.sqrt(W)@np.transpose(L), (np.linalg.solve(L,np.sqrt(W)@K@b)) )

        Fnew = K@a

        delta = np.linalg.norm(Fnew-F)
        if delta<1e-2:
            flag = 0
        F = Fnew
    '''



    # part2: optimize marginal likelihood
    '''
    opts = {'maxiter': 1,
            'disp': True,  # non-default value.
            'gtol': 1e-1,
            'norm': np.inf,  # default value.
            'eps': 1.4901161193847656e-08}  # default value.
    Para,fun = optimize.minimize(Neg_LogMar_Likelihood, Para, jac=PDe_Neg_LogMar_Likelihood, args=(TrainY,TrainX,F), method="Newton-CG", options=opts)
    '''
    #Para, fun = sp.optimize.fmin_cg(Neg_LogMar_Likelihood, Para, fprime=PDe_Neg_LogMar_Likelihood, args=(TrainY,TrainX,F),gtol=1e-1,maxiter=1)
    #Para, fun = sp.optimize.fmin_bfgs(Neg_LogMar_Likelihood, Para, fprime=PDe_Neg_LogMar_Likelihood,args=(TrainY, TrainX, F), gtol=1e-1, maxiter=1)
    Para, fun = OPT_GradientDescent(Neg_LogMar_Likelihood, PDe_Neg_LogMar_Likelihood, Para, 1000, 1e-6, 0.005, (TrainY,TrainX,F))
    #print("The negative log marginal likelihood is {}.".format(fun))
    print("The negative log marginal likelihood is {}.".format(fun))
    print("----------")

    NumofIteration = NumofIteration - 1


# prediction
de1_log_like_list = []
for i in range(NumofTrain):
    de1_log_like_list.append(De1_LogLogistic(TrainY[i],F[i]))
de1_log_like_list = np.array(de1_log_like_list)
de2_log_like_list = []
for i in range(NumofTrain):
    de2_log_like_list.append(De2_LogLogistic(F[i]))
W = - np.diag(de2_log_like_list)
K = K_CovMatrix(NumofTrain,TrainX,Para)
Kstar = Kstar_CovMatrix(NumofTrain,NumofTest,TrainX,TestX,Para)
Kstar2 = Kstar2_CovMatrix(NumofTest,TestX,Para)
B = np.identity(NumofTrain) + np.sqrt(W) @ K @ np.sqrt(W)
L = np.linalg.cholesky(B)
Fstar_mean = np.transpose(Kstar) @ np.transpose(de1_log_like_list)
v = np.linalg.solve(L, np.sqrt(W)@Kstar)
Fstar_cov = Kstar2 - np.transpose(v) @ v
PI = 1/(1 + np.exp(-Fstar_mean))

print(PI)

for i in range(NumofTest):
    plt.text(TestX[0][i], TestX[1][i], PI[i], fontsize=12)

plt.show()

