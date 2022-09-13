#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 10:58:42 2022

@author: masiha
"""

import math
import torch 
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import Variable
#torch.manual_seed(2)



b=0#1.603
rho =(4.603-b)*4*2
l_sgd =2+(4.603-b)*2
l=2+(4.603-b)*2
c_ = 1#1e-7

w_opt = 0
batch_size = 10
eps = 1e-6#1/(batch_size)
T_eps = 20#int(l/(rho*eps)**0.5+1)
max_Instance = 50
num_iteration=250
Total_oracle_calls = 2000
init_batch=1
num_init_batch=100
sigma1=1
sigma2=1

# def F(w):
#     return torch.mean(1+torch.pow(w,2))

# def f(w):
#     return torch.mean(torch.pow(torch.normal(0, 2, size=(1,w.size(dim=1)))-w*torch.normal(0, 2, size=(1,w.size(dim=1))),2))

def F(w):
    return torch.mean(torch.pow(w, 2) + (4.603-b)*torch.pow(torch.sin(w), 2))


def f(w):
    return torch.mean((torch.pow(w, 2) + (4.603-b)*torch.pow(torch.sin(w), 2)) + torch.normal(0, 2, size=(1,w.size(dim=1))))

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
        delta = (4/(l_sgd))*float(1/(i+1))*grad#eta * grad#float(1/(i+1))*grad#eta * grad#float(1/(i+1))*grad#eta * grad#
        w_new = w.detach() - delta
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        f_history.append(F(w).detach().numpy())
        grad_history.append(grad.detach().numpy().tolist()[0])
        if i<=100:
            batch_size1=5
        else:
            batch_size1=batch_size
        num_oracle = num_oracle + batch_size1
        num_oracle_history.append(num_oracle)
        w.requires_grad = True
    return w, w_norm_history, num_oracle_history, f_history



def cubic_regularization(eps, batch_size,w_init):
    w = Variable(w_init*torch.ones(1,1), requires_grad=True)
    c = -(eps ** 3 / rho) ** (0.5) / 100
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int((Total_oracle_calls-num_init_batch*init_batch)/batch_size+num_init_batch)):
        #if i % 10 == 0: print(w)
        if i<num_init_batch:
            batch_size1=init_batch
        else:
            batch_size1=batch_size
        w_batch = batch(w,batch_size1)
        F(w_batch).backward()
        grad = w.grad.clone()[0]
        grad= grad + torch.normal(0, sigma1/(batch_size1)**0.5, size=(1,w.size(dim=1)))
        hessian = torch.autograd.functional.hessian(f, inputs=w_batch)#[0][0][0]
        hessian=torch.reshape(torch.sum(torch.reshape(hessian,(-1,))),(1,1))
        hessian= hessian + torch.normal(0, sigma2/(batch_size1)**0.5, size=(1,w.size(dim=1)))
        #print("w:",w)
        #print("grad:", grad) 
        #print("hessian:",hessian)
        num_oracle = num_oracle + batch_size1*2
        delta, delta_m = cubic_subsolver(grad, hessian, eps)
        w_new = w.detach() + delta
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

def cubic_subsolver(grad, hessian, eps):
    g_norm = torch.norm(grad)
    # print(g_norm)
    if g_norm > l**2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm.pow(2) 
        R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / rho)
        delta = -R_c * grad / g_norm
    else:
        delta = torch.zeros(grad.size())
        sigma = c_ * (eps * rho)**0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = torch.randn(grad.size())#*2 + torch.ones(grad.size())
        vec /= torch.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu *(g_ + delta @ hessian + rho / 2 * torch.norm(delta) * delta)
        
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
total_hist_f1 = []
total_hist_grad = []
for i in range(max_Instance):
     #SCRN
     w_init = 5*np.random.uniform(-1,1)
     hist_w, hist_f, hist_grad = cubic_regularization(eps,batch_size,w_init)
     total_hist_f.append(hist_f)
     total_hist_grad.append(np.abs(hist_grad))
     #SGD
     w0, hist_w1, hist_oracle, f_history = SGD(w_init,eps)
     total_hist_f1.append(f_history)
     #print(hist_w)
     #print(hist_f)

mean_hist_f = np.mean(np.array(total_hist_f),axis = 0)
mean_hist_grad = np.mean(np.array(total_hist_grad),axis = 0)
mean_hist_f_SGD = np.mean(np.array(total_hist_f1),axis = 0)

batch_size11=[None]*int((Total_oracle_calls-num_init_batch*init_batch)/batch_size+num_init_batch)
for i in range(int((Total_oracle_calls-num_init_batch*init_batch)/batch_size+num_init_batch)):
    if i<num_init_batch:
        batch_size11[i]=init_batch
    else:
        batch_size11[i]=batch_size


plt.plot([int(batch_size11[i])*(i+1) for i in range(len(mean_hist_f))], mean_hist_f, color='green', label="SCRN")
plt.plot([(i+1) for i in range(len(mean_hist_f_SGD))], mean_hist_f_SGD, color='red', label="SGD")
plt.xlabel('Number of calls')
plt.ylabel('E[F(x)]-F*')
plt.grid(True)
plt.legend(loc="upper right")
ax = plt.gca()
#ax.set_xlim([left, right]) 
#ax.set_ylim([ymin,ymax])
plt.savefig('SCRN-vs-SGD-tuavarying.pdf')
#plt.xscale('log')
plt.yscale('log')
plt.savefig('SCRN-vs-SGD-tuavarying-log.pdf')


