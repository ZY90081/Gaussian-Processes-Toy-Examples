import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
import math
import csv

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

def K_CovMatrix(TrainX, Para, NumofClass, NumofTrain):

    Dim = TrainX.shape[0]
    # reformat hyperparameters
    Para = np.array(Para)
    PPara = Para.reshape((NumofClass, Dim+1))

    K = []
    for c in range(NumofClass):
        Kc = np.zeros((NumofTrain, NumofTrain))
        for i in range(NumofTrain):
            for j in range(NumofTrain):
                Kc[i][j] = ARDkernel(TrainX[:, [i]],TrainX[:, [j]], PPara[c])
        K = sp.linalg.block_diag(K,Kc)
    K = np.delete(K, 0, 0)  # delete the first row

    return K

def Kstar_CovMatrix(Xstar,TrainX, Para, NumofClass, NumofTrain):
    Dim = TrainX.shape[0]
    # reformat hyperparameters
    Para = np.array(Para)
    PPara = Para.reshape((NumofClass, Dim+1))

    K = []
    for c in range(NumofClass):
        Kc = np.zeros((NumofTrain, 1))
        for i in range(NumofTrain):
            for j in range(1):
                Kc[i][j] = ARDkernel(TrainX[:, [i]],Xstar, PPara[c])
        K = sp.linalg.block_diag(K,Kc)
    K = np.delete(K, 0, 0)  # delete the first row

    return K


def Kstar2_CovMatrix(Xstar, Para, NumofClass):
    Dim = TrainX.shape[0]
    # reformat hyperparameters
    Para = np.array(Para)
    PPara = Para.reshape((NumofClass, Dim+1))

    Kc = []
    for c in range(NumofClass):
        Kc.append(ARDkernel(Xstar,Xstar, PPara[c]))
    Kc = np.array(Kc)
    return Kc


def PDeK_CovMatrix_sf(TrainX, Para, NumofClass, NumofTrain,cidx):
    Dim = TrainX.shape[0]
    # reformat hyperparameters
    Para = np.array(Para)
    PPara = Para.reshape((NumofClass, Dim+1))

    K = []
    for c in range(NumofClass):
        if c == cidx:
            Kc = np.zeros((NumofTrain, NumofTrain))
            for i in range(NumofTrain):
                for j in range(NumofTrain):
                    Kc[i][j] = PDe_ARDkernel_sf(TrainX[:, [i]],TrainX[:, [j]], PPara[c])
        else:
            Kc = np.zeros((NumofTrain, NumofTrain))

        K = sp.linalg.block_diag(K,Kc)
    K = np.delete(K, 0, 0)  # delete the first row

    return K

def PDeK_CovMatrix_alpha(TrainX, Para, NumofClass, NumofTrain,cidx,q):
    Dim = TrainX.shape[0]
    # reformat hyperparameters
    Para = np.array(Para)
    PPara = Para.reshape((NumofClass, Dim + 1))

    K = []
    for c in range(NumofClass):
        if c == cidx:
            Kc = np.zeros((NumofTrain, NumofTrain))
            for i in range(NumofTrain):
                for j in range(NumofTrain):
                    Kc[i][j] = PDe_ARDkernel_alpha(TrainX[:, [i]], TrainX[:, [j]], PPara[c], q)
        else:
            Kc = np.zeros((NumofTrain, NumofTrain))

        K = sp.linalg.block_diag(K, Kc)
    K = np.delete(K, 0, 0)  # delete the first row
    return K

def PI(F,NumofClass, NumofTrain):
    exp_f = np.exp(F)
    # reformat exp_f
    re_exp_f = exp_f.reshape((NumofClass,NumofTrain))
    tr_re_exp_f = np.transpose(re_exp_f)
    pi = []
    for c in range(NumofClass):
        for i in range(NumofTrain):
            pi.append(re_exp_f[c][i]/sum(tr_re_exp_f[i]))

    return np.array(pi)

def Neg_Log_Posterior(F,args):

    (TrainYY, TrainX, Para, NumofClass, NumofTrain) = args

    # reformat F
    FF = F.reshape((NumofClass,NumofTrain))

    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)

    s_log = 0
    for i in range(NumofTrain):
        s_exp_f = 0
        for c in range(NumofClass):
            s_exp_f = s_exp_f + math.exp(FF[c][i])
        s_log = s_log + math.log(s_exp_f)

    output = 0.5*np.transpose(F)@np.linalg.inv(K)@F - np.transpose(TrainYY)@F + s_log + 0.5*math.log(np.linalg.det(K)+1e-6) + 0.5*NumofClass*NumofTrain*math.log(2*np.pi)
    return output

def PDe1_Neg_Log_Posterior(F,args):
    (TrainYY, TrainX, Para, NumofClass, NumofTrain) = args
    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)
    pi = PI(F, NumofClass, NumofTrain)

    output = np.linalg.inv(K)@F - TrainYY + pi
    return output

def PDe2_Neg_Log_Posterior(F,args):
    (TrainYY, TrainX, Para, NumofClass, NumofTrain) = args
    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)
    pi = PI(F, NumofClass, NumofTrain)
    PPi = pi.reshape((NumofClass,NumofTrain))
    G = np.zeros((1,NumofTrain))
    for c in range(NumofClass):
        G = np.vstack((G,np.diag(PPi[c])))
    G = np.delete(G, 0, 0)  # delete the first row
    W = np.diag(pi) - G@np.transpose(G)
    output = np.linalg.inv(K) + W
    return output

def Neg_LogMar_Likelihood(Para,args):
    (TrainYY, TrainX, F, NumofClass, NumofTrain) = args
    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)
    FF = F.reshape((NumofClass, NumofTrain))
    s_log = 0
    for i in range(NumofTrain):
        s_exp_f = 0
        for c in range(NumofClass):
            s_exp_f = s_exp_f + math.exp(FF[c][i])
        s_log = s_log + math.log(s_exp_f)

    pi = PI(F, NumofClass, NumofTrain)
    PPi = pi.reshape((NumofClass,NumofTrain))
    G = np.zeros((1,NumofTrain))
    for c in range(NumofClass):
        G = np.vstack((G,np.diag(PPi[c])))
    G = np.delete(G, 0, 0)  # delete the first row
    W = np.diag(pi) - G@np.transpose(G)
    W[W < 0] = 0

    output = 0.5*np.transpose(F)@np.linalg.inv(K)@F - np.transpose(TrainYY)@F + s_log + 0.5*math.log(np.linalg.det(np.eye(NumofClass*NumofTrain)+np.sqrt(W)@K@np.sqrt(W)))
    return output

def PDe_Neg_LogMar_Likelihood(Para,args):
    (TrainYY, TrainX, F, NumofClass, NumofTrain) = args
    Dim = TrainX.shape[0]
    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)
    pi = PI(F, NumofClass, NumofTrain)
    PPi = pi.reshape((NumofClass,NumofTrain))
    G = np.zeros((1,NumofTrain))
    for c in range(NumofClass):
        G = np.vstack((G,np.diag(PPi[c])))
    G = np.delete(G, 0, 0)  # delete the first row
    W = np.diag(pi) - G@np.transpose(G)
    W[W < 0] = 0
    B = np.eye(NumofClass*NumofTrain) + np.sqrt(W)@K@np.sqrt(W)

    output = []
    for c in range(NumofClass):
        DrK_sf = PDeK_CovMatrix_sf(TrainX, Para, NumofClass, NumofTrain,c)
        output.append( -0.5*np.transpose(F)@(np.linalg.inv(K)@DrK_sf@np.linalg.inv(K))@F + 0.5*np.trace(np.linalg.inv(B)@W@DrK_sf) )
        for i in range(Dim):
            Drk_alpha = PDeK_CovMatrix_alpha(TrainX, Para, NumofClass, NumofTrain,c,i+1)
            output.append(-0.5 * np.transpose(F) @ (np.linalg.inv(K) @ Drk_alpha @ np.linalg.inv(K)) @ F + 0.5 * np.trace(
                np.linalg.inv(B) @ W @ Drk_alpha))

    return np.array(output)


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
        print("The negative log posterior is {}.".format(fvals[-1]))
        if np.linalg.norm(delta)<Termination:
            flag = 0

    return x, fvals[-1]

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
        print("The negative log marginal likelihood is {}.".format(fvals[-1]))
        if np.linalg.norm(delta)<Termination:
            flag = 0

    return x,fvals[-1]

def Pred_mean(Xstar,F,Para,TrainYY,TrainX,NumofClass,NumofTrain):
    pi = PI(F, NumofClass, NumofTrain)
    Qstar = Kstar_CovMatrix(Xstar,TrainX, Para, NumofClass, NumofTrain)
    return np.transpose(Qstar)@(TrainYY-pi)

def Pred_cov(Xstar,F,Para,TrainYY,TrainX,NumofClass,NumofTrain):
    K = K_CovMatrix(TrainX, Para, NumofClass, NumofTrain)
    Qstar = Kstar_CovMatrix(Xstar, TrainX, Para, NumofClass, NumofTrain)
    Kstar = Kstar2_CovMatrix(Xstar, Para, NumofClass)
    pi = PI(F, NumofClass, NumofTrain)
    PPi = pi.reshape((NumofClass,NumofTrain))
    G = np.zeros((1,NumofTrain))
    for c in range(NumofClass):
        G = np.vstack((G,np.diag(PPi[c])))
    G = np.delete(G, 0, 0)  # delete the first row
    W = np.diag(pi) - G@np.transpose(G)
    W[W < 0] = 0
    return np.diag(Kstar) - np.transpose(Qstar)@np.linalg.inv(K+np.linalg.inv(W))@Qstar

# generate data

NumofClass = 4
NumofTrain_each = 50
NumofTrain = NumofTrain_each * NumofClass
NumofTest_each = 5
NumofTest = NumofTest_each * NumofClass

Mu_c1 = [2,5]    # mean of classes
Mu_c2 = [5,-1]
Mu_c3 = [9,9]
Mu_c4 = [10,-5]

Cov_c1 = [[6,0],[0,10]]   # covariance of classes
Cov_c2 = [[10,5],[5,10]]
Cov_c3 = [[2,1],[1,4]]
Cov_c4 = [[4,0],[0,4]]

x_c1, y_c1 = np.random.multivariate_normal(Mu_c1, Cov_c1, NumofTrain_each).T
x_c2, y_c2 = np.random.multivariate_normal(Mu_c2, Cov_c2, NumofTrain_each).T
x_c3, y_c3 = np.random.multivariate_normal(Mu_c3, Cov_c3, NumofTrain_each).T
x_c4, y_c4 = np.random.multivariate_normal(Mu_c4, Cov_c4, NumofTrain_each).T
TrainX = np.array([x_c1.tolist()+x_c2.tolist()+x_c3.tolist()+x_c4.tolist(),y_c1.tolist()+y_c2.tolist()+y_c3.tolist()+y_c4.tolist()])
TrainY = [1]*NumofTrain_each + [2]*NumofTrain_each + [3]*NumofTrain_each + [4]*NumofTrain_each

x_c1_te, y_c1_te = np.random.multivariate_normal(Mu_c1, Cov_c1, NumofTest_each).T
x_c2_te, y_c2_te = np.random.multivariate_normal(Mu_c2, Cov_c2, NumofTest_each).T
x_c3_te, y_c3_te = np.random.multivariate_normal(Mu_c3, Cov_c3, NumofTest_each).T
x_c4_te, y_c4_te = np.random.multivariate_normal(Mu_c4, Cov_c4, NumofTest_each).T
TestX = np.array([x_c1_te.tolist()+x_c2_te.tolist()+x_c3_te.tolist()+x_c4_te.tolist(),y_c1_te.tolist()+y_c2_te.tolist()+y_c3_te.tolist()+y_c4_te.tolist()])
TestY = [1]*NumofTest_each + [2]*NumofTest_each + [3]*NumofTest_each + [4]*NumofTest_each

fig1 = plt.figure(figsize=(6,6))
plt.plot(x_c1, y_c1, 'oc')
plt.plot(x_c2, y_c2, 'oy')
plt.plot(x_c3, y_c3, 'ob')
plt.plot(x_c4, y_c4, 'og')
plt.plot(TestX[0], TestX[1], '*r',Markersize=10)



#############################################GP multi-class Classification
Dim = TrainX.shape[0]   # dimension of inputs
NumofIteration = 1

# reformat the training labels
TrainYY = []
for c in range(1,NumofClass+1):
    for i in range(NumofTrain):
        if TrainY[i] == c:
            TrainYY.append(1)
        else:
            TrainYY.append(0)
TrainYY = np.array(TrainYY)

InitialPara = ([1] + [1]*Dim)*NumofClass   # initial hyperparameters
InitialF = [0]*NumofTrain*NumofClass       # initial latent function

#InitialPara = np.array(InitialPara)
#print(InitialPara.reshape((NumofClass, Dim+1)))

Para = np.array(InitialPara)
F = np.array(InitialF)

while NumofIteration>0:
    # part1: optimize posterior
    F, fun = OPT_Newton(Neg_Log_Posterior, PDe1_Neg_Log_Posterior, PDe2_Neg_Log_Posterior, F, 10, 1e-6, 0.2,
                        (TrainYY, TrainX, Para, NumofClass, NumofTrain))
    #print("The negative log posterior is {}.".format(fun))
    print("----------")

    # part2: optimize marginal likelihood
    Para, fun = OPT_GradientDescent(Neg_LogMar_Likelihood, PDe_Neg_LogMar_Likelihood, Para, 10, 1e-6, 0.005,
                                    (TrainYY, TrainX, F, NumofClass, NumofTrain))
    #print("The negative log marginal likelihood is {}.".format(fun))
    print("----------")

    NumofIteration = NumofIteration - 1



with open('Training.csv', mode='w') as Trainingfile:
    writer = csv.writer(Trainingfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(F)
    writer.writerow(Para)

# prediction
NumofSample = 100
pistar_forall = []
decision = []
for i in range(NumofTest):
    Fstar_mean = Pred_mean(TestX[:, [i]],F,Para,TrainYY,TrainX,NumofClass,NumofTrain)
    Fstar_cov = Pred_cov(TestX[:, [i]],F,Para,TrainYY,TrainX,NumofClass,NumofTrain)
    fstar_samples = np.random.multivariate_normal(Fstar_mean, Fstar_cov, NumofSample).T
    pistar = np.zeros((NumofClass,1))
    for j in range(NumofSample):
        pistar = pistar + np.exp(fstar_samples[:,[j]])/sum(np.exp(fstar_samples[:,[j]]))
    pistar = np.transpose(pistar/NumofSample)
    #print(pistar)
    maxidx = np.argmax(pistar)
    testlabel = np.zeros((1,NumofClass))
    testlabel[0][maxidx] = 1
    decision.append(testlabel.tolist())
    pistar_forall.append(pistar.tolist())

pistar_forall = np.array(pistar_forall)
decision = np.array(decision)

print(pistar_forall)
print(decision)


for i in range(NumofTest):
    maxidx = np.argmax(pistar_forall[i])
    if maxidx==0:
        plt.plot(TestX[0][i], TestX[1][i], '*c', Markersize=10)
    elif maxidx==1:
        plt.plot(TestX[0][i], TestX[1][i], '*y', Markersize=10)
    elif maxidx==2:
        plt.plot(TestX[0][i], TestX[1][i], '*b', Markersize=10)
    else:
        plt.plot(TestX[0][i], TestX[1][i], '*g', Markersize=10)

plt.show()
