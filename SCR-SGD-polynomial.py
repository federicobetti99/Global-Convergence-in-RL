#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:30:04 2022

@author: masiha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:52:04 2022

@author: salehkal
"""
import math
import sympy as sym
import torch 
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable
#torch.manual_seed(2)


eps = 0.0001
rho = 210#280#240#210#24#60#0.02
l = 42#20#0.1
c_ = 1#0.1
T_eps = 15#int(l/(rho*eps)**0.5+1)*10
w_opt = 0
batch_size = 20
#np.random.uniform(-1,1).
max_Instance = 50
Total_oracle_calls = 2000
sigma1=1
sigma2=1

#p=4 , q=1
# SCRN: rho = 25, l = 25, c_ = 2, T_eps =10, batch_size = 0, w_init = 3, eps = 0.001
# SGD: eta = 0.01, 1/sqrt(t)


# def F(w):
#     return torch.mean(1+torch.pow(w,2))

# def f(w):
#     return torch.mean(torch.pow(torch.normal(0, 2, size=(1,w.size(dim=1)))-w*torch.normal(0, 2, size=(1,w.size(dim=1))),2))
p=14
q=2

def boundaries(w):
    #w = Variable(w_init*torch.ones(1,1), requires_grad=True)
    #if -0.1 < w[0][0] and w[0][0] < 0.1:
    #    return True
    #else:
    return False

def F(w):
    if boundaries(w) == True:
        return torch.mean(torch.pow(w, 2))
    else:
        return torch.mean(torch.pow(torch.pow(w, p),1/q))
    
def f(w):
    if boundaries(w) == True:
        return torch.mean(torch.pow(w, 2)*(1 + torch.normal(0, sigma1, size=(1,w.size(dim=1)))))
    else:
        return torch.mean((torch.pow(torch.pow(w, p),1/q)) + torch.normal(0, sigma1, size=(1,w.size(dim=1))))

#ef F(w):
 #   return torch.mean(torch.pow(torch.pow(w/10, p),1/q))


#def f(w):
#    return torch.mean((torch.pow(torch.pow(w/10, p),1/q))*(1 + torch.normal(0, 2, size=(1,w.size(dim=1)))))

def batch(w, batch_size):
    return w*torch.ones(batch_size)

def SGD(w_init,eps=None):
    eta = 10**-1
    w = Variable(w_init*torch.ones(1,1), requires_grad=True)
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls)):
        #if i % 10 == 0: print(w)
        w_batch = batch(w, 1)#batch(w, batch_size)
        F(w_batch).backward()
        grad = w.grad.clone()
        grad = grad + torch.normal(0, sigma1, size=(1,w.size(dim=1)))
        #print("w:", w)
        #print("grad:",grad)
        #print("w:", w)
        #print("grad:",grad)
        delta = (1/(2*l))*float(1/np.sqrt(i+1))*grad#eta * grad#float(1/(i+1))*grad#eta * grad#float(1/(i+1))*grad#eta * grad#
        w_new = w.detach() - delta
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        f_history.append(F(w).detach().numpy())
        grad_history.append(grad.detach().numpy().tolist()[0])
        num_oracle = num_oracle + batch_size
        num_oracle_history.append(num_oracle)
        w.requires_grad = True
    return w, w_norm_history, num_oracle_history, f_history



def cubic_regularization(eps, batch_size, w_init):
    w = Variable(w_init*torch.ones(1,1), requires_grad=True)
    c = -(eps ** 3 / rho) ** (0.5) / 100
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls/batch_size)):
        #if i % 10 == 0: print(w)
        #print(w)
        w_batch = batch(w,batch_size)
        F(w_batch).backward()
        grad = w.grad.clone()[0]
        grad= grad + torch.normal(0, sigma1/(batch_size)**0.5, size=(1,w.size(dim=1)))
        hessian = torch.autograd.functional.hessian(f, inputs=w_batch)#[0][0][0]
        hessian=torch.reshape(torch.sum(torch.reshape(hessian,(-1,))),(1,1))
        hessian= hessian + torch.normal(0, sigma2/(batch_size)**0.5, size=(1,w.size(dim=1)))
       #print("w:",w)
       #print("grad:", grad) 
       #print("hessian:",hessian)
        num_oracle = num_oracle + batch_size*2
        delta, delta_m = cubic_subsolver(grad, hessian, eps, l)#*(1/math.pow(i+1,1)),l* math.pow(i+1,0))
        w_new = w.detach() + delta #change stepsize
        #if delta_m.item() >= c:
        #    delta = cubic_finalsolver(grad, hessian, eps)
        #    return w + delta, w_norm_history
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        num_oracle_history.append(num_oracle)
        f_history.append(F(w).detach().numpy().tolist())
        grad_history.append(grad.detach().numpy().tolist()[0])
        w.requires_grad = True
    #print(w_norm_history)
    return w_norm_history, f_history, grad_history#, num_oracle_history

def cubic_subsolver(grad, hessian, eps,l):
  
    g_norm = torch.norm(grad)
    #print("g_norm:",g_norm)
    if g_norm > l**2/rho:
        temp = grad @ hessian @ grad.T / rho / g_norm.pow(2) 
        R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / rho)
        delta = -R_c * grad / g_norm
    else:
        delta = torch.zeros(grad.size())
        sigma = c_ * (eps * rho)**0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = -torch.rand(grad.size())*2 + torch.ones(grad.size())
        vec /= torch.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu *(g_ + delta @ hessian + rho / 2 * torch.norm(delta) * delta)
#            if np.isnan(delta.numpy()):
#                print(delta)
        
    delta_m = grad @ delta.T + delta @ hessian @ delta.T / 2 + rho / 6 * torch.norm(delta).pow(3)
    return delta, delta_m


def cubic_finalsolver(grad, hessian, eps):
    delta = torch.zeros(grad.size())
    g_m = grad
    #print(torch.norm(g_m))
    mu = 1 / (20 * l)
    while torch.norm(g_m) > eps/2:
        delta -= mu * g_m
        g_m = grad + delta @ hessian + rho / 2 * torch.norm(delta) * delta
    return delta


# # # # ####cubic_regularization#######

total_hist_f = []
total_hist_grad = []
for i in range(max_Instance):
     w_init = 0.95#*np.random.uniform(-1,1)
     hist_w, hist_f, hist_grad = cubic_regularization(eps,batch_size,w_init)
     total_hist_f.append(hist_f)
     total_hist_grad.append(np.abs(hist_grad))
     #print(hist_w)
     #print(hist_f)

mean_hist_f = np.mean(np.array(total_hist_f),axis = 0)
mean_hist_grad = np.mean(np.array(total_hist_grad),axis = 0)


# # ########SGD#########

left = 0
right = 500


total_hist_f_SGD = []
 #total_hist_grad = []
for i in range(max_Instance):
       w_init = 0.95#*np.random.uniform(-1,1)
       w0, hist_w, hist_oracle, f_history = SGD(w_init,eps)
       total_hist_f_SGD.append(f_history)

mean_hist_f_SGD = np.mean(np.array(total_hist_f_SGD),axis = 0)


plt.plot([batch_size*(i+1) for i in range(len(mean_hist_f))], mean_hist_f, color='green', label="SCRN")
plt.plot([(i+1) for i in range(len(mean_hist_f_SGD))], mean_hist_f_SGD, color='red', label="SGD")
plt.xlabel('Number of calls')
plt.ylabel('E[F(x)]-F*')
plt.grid(True)
plt.legend(loc="upper right")
ax = plt.gca()
#ax.set_xlim([left, right]) 
#ax.set_ylim([ymin,ymax])

#plt.xscale('log')
plt.yscale('log')
plt.savefig('SCRN-vs-SGD-alpha_less_2.pdf')






