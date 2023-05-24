import numpy as np
import matplotlib.pyplot as plt
import timeit

#exercise 3

#a


#get the data
data = np.loadtxt("https://home.strw.leidenuniv.nl/~belcheva/galaxy_data.txt")
features = data[:,:4]
classif = data[:,4]

xlabels = np.array([r"$\kappa_{CO}$", "redness", "extendedness", "flux emission line (SFR)"])

#scale them to a uniform distribution of mean 0 and standard dev 1
for i in range(4):
	mean = np.mean(features[:,i])
	std = np.std(features[:,i])
	features[:,i] = (features[:,i] - mean)/std

	plt.hist(features[:,i], bins=25)
	if i == 2 or i == 3:
		plt.yscale("log")
	plt.title("distribution of rescaled feature "+str(i+1))
	plt.ylabel("nr of objects")
	plt.xlabel(xlabels[i])
	plt.savefig("NUR4Q3distr"+str(i+1)+".pdf")
	plt.close()


np.savetxt("RescaleFeatures.txt", np.transpose([features[:,0], features[:,1], features[:,2], features[:,3]]))


#b
#getting the cost function J and gradient from the tutorial

def J(h, y=classif):
    """Returns the cost function for data classification y, predicted classification h (sigma(z)), and the number of data points m."""
    m = len(y)
    return -(1/m)*np.sum(y*np.log(h) - (1-y)*np.log(1-h))

def Jgrad(h, x, y=classif):
    """Returns the gradient cost function for data classification y, predicted classification h (sigma(z)), and the number of data points m."""
    m = len(y)
    return (1/m) *np.sum(x.T *(h-y), axis=1)


#golden section search to do line minimization
def Golden(func,a,b,c, maxiter, accur):
    """Golden section search for finding the minimum of func with the initial bracket [a,b,c]. It iterates either over maxiter iterations or stops when the bracket becomes smaller than accur."""
    #golden ratio and weight
    phi = (1+np.sqrt(5))*0.5
    w = 2-phi
    
    
    for k in range(maxiter):
#see which bracket is smaller
        if np.abs(c-b) > np.abs(a-b):
            x = c
        else:
            x = a
#get the new guess for the minimum
        d = b + (x-b)*w

#if d is a worse guess than b, keep b as best guess & make bracket smaller
        if func(b) < func(d):
            if np.abs(c-a) < accur:
                print("accur return b", np.abs(c-a))
                return b
            if np.abs(c-b) > np.abs(a-b):
                c = d
            else:
                a = d
#otherwise, shift bracket to center d
        if func(d) < func(b):
            if np.abs(c-a) < accur:
                print("accur return d", np.abs(c-a))
                return d
        
            if np.abs(c-b) < np.abs(a-b):
                c,b = b,d
            else:
                a,b = b,d

    return b

#Quasinewton adapted to fit logistic regression
def QuasiNewt(func, x, theta, maxiter, accur, grad, H_0= np.array([[1,0,0],[0,1,0],[0,0,1]]), i=0, xall=None):
    """Quasi Newton method adapted to logistic regression to find the best theta for a multidimensional function func using the gradient grad and either iterating maxiter times or until the size of the gradient is smaller than accur. Returns the minimum and all the steps it took to get there."""
#intitial z and h
    z = np.sum(theta*x,axis=1)
    h = 1/(1+np.exp(-z)) 


    #matrix vector multiplication
    n_init = -np.sum(H_0 * grad(h,x), axis=1)
       

    #function to find the best lambda
    def minim(lambd):
        theta_new = theta +lambd*n_init
        z = np.sum(theta_new*x,axis=1)
        h = 1/(1+np.exp(-z))
        return func(h)

#if the function does not exist or returns NaNs for small numbers, search for a lambda in a small range, if it does exist somewhere, search in a bigger range
    if np.isnan(minim(-1)) or np.isnan(minim(1)):
        #print("1 nan")
        lam_i = Golden(minim, -0.1,0.001,0.1,50, 10**(-20))
    if np.isnan(minim(-0.1)) or np.isnan(minim(0.1)):
        #print("0.1 nan")
        lam_i = Golden(minim, -0.01,0.00001,0.01,50, 10**(-20))
    else:
        #print("take 15")
        lam_i = Golden(minim, -15,0.1,15,50, 10**(-20))

    
    #we dont want to take a stepsize of zero
    if lam_i == 0:
        lam_i = 10**(-3)
    
    delta = lam_i *n_init
    
    #we dont want to take a too small step
    while np.abs(delta)[0] < 10**(-10) and np.abs(delta)[1] < 10**(-10):
        delta *= 10
    

    #calculate the new x by adding delta
    theta_new = theta + delta
    z_new = np.sum(theta_new*x,axis=1)
    h_new = 1/(1+np.exp(-z_new))
    
    #save all the steps we take
    if i == 0:
        xall = theta_new.copy()
    else:
        xall = np.vstack((xall, theta_new)).copy()
    


    #get the function values of the old and new x
    f0, f_new = func(h), func(h_new)

    #if the difference is smaller than our accuracy, return, and stop when the difference is 0
    if np.abs(f_new - f0)/(0.5*np.abs(f_new - f0)) < accur or (f_new - f0) == 0:

        return theta_new, xall
    
    #Calculate D
    D_i = grad(h_new, x) - grad(h, x)

    
    #if the gradient converges, return
    if np.abs(np.amax(grad(h_new, x), axis=0)) < accur:
        print("grad conv")
        return theta_new, xall
    
    #H times D
    HD =np.sum(H_0*D_i, axis=1)
    
    u = delta/np.sum(delta*D_i) - HD/np.sum(D_i * HD)

    #calculate the new H
    H_i = H_0 + np.outer(delta, delta)/np.sum(delta*D_i) - np.outer(HD, HD)/np.sum(D_i * HD) + np.sum(D_i * HD)*np.outer(u,u)

    
    #iteration goes up 1, so either return or do another iteration
    i+= 1
    
    if i >= maxiter:
        print("iter return")
        return theta_new, xall
    else:
        return QuasiNewt(func, x, theta_new, maxiter, accur, grad, H_i, i, xall)


#logistic regression
def logregr(x, theta, y, alpha=0.1):
	n = x[0,:].size

	#minimization
	theta_new, theta_all = QuasiNewt(J, x, theta, maxiter=100, accur=10**(-6), grad=Jgrad, H_0=np.identity(n))

	z_new = np.sum(theta_new*x,axis=1)
	h_new = 1/(1+np.exp(-z_new))

	J_all = np.empty(len(theta_all))
	#calculate cost function
	for k in range(len(theta_all)):
		z_all = np.sum(theta_all[k]*x,axis=1)
		h_all = 1/(1+np.exp(-z_all))
		J_all[k] = J(h_all, y)

	return theta_new, theta_all, h_new, J_all

#I choose the last two features, extendedness and SFR, because I feel like those two features are not necessarily correlated so they both say something independent about whether the galaxy is a spiral or an elliptical

feats = features[:,2:]

#first set all thetas to 1
thetas = np.ones(feats[0,:].size)

theta_end, thet_all, h_end, J_alls = logregr(feats, thetas, classif)
print("theta end,  nr of steps", theta_end, len(thet_all))
steps = np.arange(0,len(thet_all),1)

print(theta_end)
plt.plot(steps, J_alls)
plt.xlabel("steps")
plt.ylabel("cost function value")
plt.title("model 1: extendedness & SFR")
plt.savefig("NUR4Q3costfunc1.pdf")
plt.close()


#now do for all features
thetas2 = np.ones(features[0,:].size)

theta_end2, thet_all2, h_end2, J_alls2 = logregr(features, thetas2, classif)
print("theta end,  nr of steps", theta_end2, len(thet_all2))
steps2 = np.arange(0,len(thet_all2),1)
print(thet_all2)

plt.plot(steps2, J_alls2)
plt.xlabel("steps")
plt.ylabel("cost function value")
plt.title("model 2: all features")
plt.savefig("NUR4Q3costfunc2.pdf")
plt.close()


#do two other features, two I think are correlated, aka color and sfr (1 and 3)
feats2 = features[:,1::2]
thetas3 = np.ones(feats2[0,:].size)
print(thetas3.size)

theta_end3, thet_all3, h_end3, J_alls3 = logregr(feats2, thetas3, classif)
print("theta end,  nr of steps", theta_end3, len(thet_all3))
steps3 = np.arange(0,len(thet_all3),1)
print(thet_all3)

plt.plot(steps3, J_alls3)
plt.xlabel("steps")
plt.ylabel("cost function value")
plt.title("model 3: color & SFR")
plt.savefig("NUR4Q3costfunc3.pdf")
plt.close()


#do two other non correlated features (first two)
feats3 = features[:,:2]
thetas4 = np.ones(feats3[0,:].size)
#print(thetas3.size)

theta_end4, thet_all4, h_end4, J_alls4 = logregr(feats3, thetas4, classif)
print("theta end,  nr of steps", theta_end4, len(thet_all4))
steps4 = np.arange(0,len(thet_all4),1)
print(thet_all4)

plt.plot(steps4, J_alls4)
plt.xlabel("steps")
plt.ylabel("cost function value")
plt.title(r"model 4: $\kappa_{CO}$ & redness")
plt.savefig("NUR4Q3costfunc4.pdf")
plt.close()


#seems like for 'correlated' features it performs wacky? but it still does well

#c

#get the final classifications
final_class = np.ones(len(h_end))
final_class2 = np.ones(len(h_end2))
final_class3 = np.ones(len(h_end3))
final_class4 = np.ones(len(h_end4))

#if h > 0.5, class is 1, otherwise 0
for j in range(len(h_end)):
	if h_end[j] < 0.5:
		final_class[j] = 0

	if h_end2[j] < 0.5:
		final_class2[j] = 0

	if h_end3[j] < 0.5:
		final_class3[j] = 0

	if h_end4[j] < 0.5:
		final_class4[j] = 0

#get the masks for which ones are rightly classified and which ones wrongly
right_class = final_class == classif
right_class2 = final_class2 == classif
right_class3 = final_class3 == classif
right_class4 = final_class4 == classif
wrong_class = final_class != classif
wrong_class2 = final_class2 != classif
wrong_class3 = final_class3 != classif
wrong_class4 = final_class4 != classif

#get the number of true/false positives/negatives by summing over the number of Trues of a mask
TP = sum(final_class[right_class] == 1)
TP2 = sum(final_class2[right_class2] == 1)
TP3 = sum(final_class3[right_class3] == 1)
TP4 = sum(final_class4[right_class4] == 1)

TN = sum(final_class[right_class] == 0)
TN2 = sum(final_class2[right_class2] == 0)
TN3 = sum(final_class3[right_class3] == 0)
TN4 = sum(final_class4[right_class4] == 0)

FP = sum(final_class[wrong_class] == 1)
FP2 = sum(final_class2[wrong_class2] == 1)
FP3 = sum(final_class3[wrong_class3] == 1)
FP4 = sum(final_class4[wrong_class4] == 1)

FN = sum(final_class[wrong_class] == 0)
FN2 = sum(final_class2[wrong_class2] == 0)
FN3 = sum(final_class3[wrong_class3] == 0)
FN4 = sum(final_class4[wrong_class4] == 0)

print("model 1", TP, TN, FP, FN)
print("model 2", TP2, TN2, FP2, FN2)
print("model 3", TP3, TN3, FP3, FN3)
print("model 4", TP4, TN4, FP4, FN4)


np.savetxt("TPvalues1.txt", np.transpose([TP, TN, FP, FN]), fmt='%i')
np.savetxt("TPvalues2.txt", np.transpose([TP2, TN2, FP2, FN2]), fmt='%i')
np.savetxt("TPvalues3.txt", np.transpose([TP3, TN3, FP3, FN3]), fmt='%i')
np.savetxt("TPvalues4.txt", np.transpose([TP4, TN4, FP4, FN4]), fmt='%i')

prec = TP / (TP + TN)
recall = TP / (TP + FP)
prec2 = TP2 / (TP2 + TN2)
recall2 = TP2 / (TP2 + FP2)
prec3 = TP3 / (TP3 + TN3)
recall3 = TP3 / (TP3 + FP3)
prec4 = TP4 / (TP4 + TN4)
recall4 = TP4 / (TP4 + FP4)


F11 = 2 * prec * recall / (prec + recall)
F12 = 2 * prec2 * recall2 / (prec2 + recall2)
F13 = 2 * prec3 * recall3 / (prec3 + recall3)
F14 = 2 * prec4 * recall4 / (prec4 + recall4)

print("Fs,", F11, F12, F13, F14)

np.savetxt("F1values.txt", np.transpose([F11, F12, F13, F14]))

#doesnt necessarily agree with the J (model 1 has lower J than 4 for example)

#plotting decision boundaries for model 2, the model with all parameters
#we have that theta_0 * x_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 = 0 at the decision boundary, so per two features we have x_i = -theta_j/theta_i * x_j


for p in range(6):
	if p < 3:
		x = features[:,0]
		y = features[:,p+1]
		slope = -theta_end2[0]/theta_end2[p+1]
		labs = [xlabels[0], xlabels[p+1]]
	if p >= 3 and p < 5: 
		x = features[:,1]
		y = features[:,p-1]
		slope = -theta_end2[1]/theta_end2[p-1]
		labs = [xlabels[1], xlabels[p-1]]
	if p == 5:
		x = features[:,2]
		y = features[:,3]
		slope = -theta_end2[2]/theta_end2[3]
		labs = [xlabels[2], xlabels[3]]

	decbound = slope * x
	plt.scatter(x,y, marker='.')
	plt.plot(x,decbound, color='crimson', label='decision boundary')
	plt.xlabel('rescaled '+labs[0])
	plt.ylabel('rescaled '+labs[1])
	plt.legend()
	#plt.show()
	plt.savefig("NUR4Q3decbound"+str(p)+".pdf")
	plt.close()


