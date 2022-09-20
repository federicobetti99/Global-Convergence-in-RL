#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:52:38 2022

Comparing SCRn (Mt), SCRN, and SGD with multiplicative noise model on a function satisfying PL alpha=2

@author: salehkal
"""

import json
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
# torch.manual_seed(2)


param = {}

b = 1.603  # 1.603
rho = (4.603 - b) * 4 * 2
l_sgd = 2 + (4.603 - b) * 2
l = 2 + (4.603 - b) * 2
c_ = 1  # 1e-7

w_opt = 0
batch_size = 10
eps = 1e-6  # 1/(batch_size)
T_eps = 20  # int(l/(rho*eps)**0.5+1)
max_Instance = 10
Total_oracle_calls = 10000  # 2500
sigma = 0.01

param['b'] = b
param['rho'] = rho
param['l_sgd'] = l_sgd
param['c_'] = c_
param['batch_size'] = batch_size
param['eps'] = eps
param['T_eps'] = T_eps
param['max_Instance'] = max_Instance
param['Total_oracle_calls'] = Total_oracle_calls
param['sigma'] = sigma
param['w_init'] = 30
param['M_coeff'] = 10


def F(w):
    return torch.mean(torch.pow(w, 2) + (4.6033333333 - b) * torch.pow(torch.sin(w), 2))


def f(w):
    return torch.mean((torch.pow(w, 2) + (4.6033333333 - b) * torch.pow(torch.sin(w), 2)) * (
                1 + torch.normal(0, sigma, size=(1, w.size(dim=1)))))


def batch(w, batch_size):
    return w * torch.ones(batch_size)


def SGD(w_init, eps=None):
    eta = 10 ** -1
    w = Variable(w_init * torch.ones(1, 1), requires_grad=True)
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls)):
        # if i % 10 == 0: print(w)
        w_batch = batch(w, 1)  # batch(w, batch_size)
        f(w_batch).backward()
        grad = w.grad.clone()
        # print("w:", w)
        # print("grad:",grad)
        delta = (4 / (l_sgd)) * float(
            1 / (i + 1)) * grad  # eta * grad#float(1/(i+1))*grad#eta * grad#float(1/(i+1))*grad#eta * grad#
        w_new = w.detach() - delta
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        f_history.append(F(w).detach().numpy())
        grad_history.append(grad.detach().numpy().tolist()[0])
        num_oracle = num_oracle + batch_size
        num_oracle_history.append(num_oracle)
        w.requires_grad = True
    return w, w_norm_history, num_oracle_history, f_history


def cubic_regularization_adaptive(eps, w_init):
    w = Variable(w_init * torch.ones(1, 1), requires_grad=True)
    c = -(eps ** 3 / rho) ** (0.5) / 100
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls)):
        # if i % 10 == 0: print(w)
        w_batch = batch(w, 1)
        f(w_batch).backward()
        grad = w.grad.detach()
        hessian = torch.autograd.functional.hessian(f, inputs=w_batch).squeeze().squeeze()
        hessian = torch.reshape(hessian, (1, 1))
        rho_t = rho + param['M_coeff'] * (i + 1)  # np.sqrt(i+1)
        # print("w:",w)
        # print("grad:", grad)
        # print("hessian:",hessian)
        num_oracle = num_oracle + batch_size * 2
        delta, delta_m = cubic_subsolver_adaptive(grad, hessian, eps, rho_t)
        w_new = w.detach() + delta
        # if delta_m.item() >= c:
        #    delta = cubic_finalsolver(grad, hessian, eps)
        #    return w + delta, w_norm_history
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        num_oracle_history.append(num_oracle)
        f_history.append(F(w).detach().numpy().tolist())
        grad_history.append(grad.detach().numpy().tolist()[0])
        w.requires_grad = True
    # print(w_norm_history)
    return w_norm_history, f_history, grad_history  # , num_oracle_history


def cubic_regularization(eps, batch_size, w_init):
    w = Variable(w_init * torch.ones(1, 1), requires_grad=True)
    c = -(eps ** 3 / rho) ** (0.5) / 100
    w_norm_history = []
    num_oracle_history = []
    f_history = []
    grad_history = []
    num_oracle = 0
    for i in range(int(Total_oracle_calls / batch_size)):
        # if i % 10 == 0: print(w)
        w_batch = batch(w, batch_size)
        f(w_batch).backward()
        grad = w.grad.detach()[0]
        hessian = torch.autograd.functional.hessian(f, inputs=w_batch).squeeze().squeeze()
        hessian = torch.reshape(torch.sum(torch.diagonal(hessian, 0)), (1, 1))
        # print("cubic")
        # print("w:",w)
        # print("grad:", grad)
        # print("hessian:",hessian)

        # print("w:",w)
        # print("grad:", grad)
        # print("hessian:",hessian)
        num_oracle = num_oracle + batch_size * 2
        delta, delta_m = cubic_subsolver(grad, hessian, eps)
        w_new = w.detach() + delta
        # if delta_m.item() >= c:
        #    delta = cubic_finalsolver(grad, hessian, eps)
        #    return w + delta, w_norm_history
        w = w_new
        w_norm_history.append(torch.norm(w - w_opt).item())
        num_oracle_history.append(num_oracle)
        f_history.append(F(w).detach().numpy().tolist())
        grad_history.append(grad.detach().numpy().tolist()[0])
        w.requires_grad = True
    # print(w_norm_history)
    return w_norm_history, f_history, grad_history  # , num_oracle_history


def cubic_subsolver(grad, hessian, eps):
    g_norm = torch.norm(grad)
    # print(g_norm)
    if g_norm > l ** 2 / rho:
        temp = grad @ hessian @ grad.T / rho / g_norm.pow(2)
        R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / rho)
        delta = -R_c * grad / g_norm
    else:
        delta = torch.zeros(grad.size())
        sigma = c_ * (eps * rho) ** 0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = torch.randn(grad.size())  # *2 + torch.ones(grad.size())
        vec /= torch.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu * (g_ + delta @ hessian + rho / 2 * torch.norm(delta) * delta)

    delta_m = grad @ delta.T + delta @ hessian @ delta.T / 2 + rho / 6 * torch.norm(delta).pow(3)
    return delta, delta_m


def cubic_subsolver_adaptive(grad, hessian, eps, rho_t):
    g_norm = torch.norm(grad)
    # print(g_norm)
    if g_norm > l ** 2 / rho_t:
        temp = grad @ hessian @ grad.T / rho_t / g_norm.pow(2)
        R_c = -temp + torch.sqrt(temp.pow(2) + 2 * g_norm / rho_t)
        delta = -R_c * grad / g_norm
    else:
        delta = torch.zeros(grad.size())
        sigma = c_ * (eps * rho_t) ** 0.5 / l
        mu = 1.0 / (20.0 * l)
        vec = torch.randn(grad.size())  # *2 + torch.ones(grad.size())
        vec /= torch.norm(vec)
        g_ = grad + sigma * vec
        # g_ = grad
        for _ in range(T_eps):
            delta -= mu * (g_ + delta @ hessian + rho_t / 2 * torch.norm(delta) * delta)

    delta_m = grad @ delta.T + delta @ hessian @ delta.T / 2 + rho_t / 6 * torch.norm(delta).pow(3)
    return delta, delta_m


def cubic_finalsolver(grad, hessian, eps):
    delta = torch.zeros(grad.size())
    g_m = grad
    # print(torch.norm(g_m))
    mu = 1 / (20 * l)
    while torch.norm(g_m) > eps / 2:
        delta -= mu * g_m
        g_m = grad + delta @ hessian + rho / 2 * torch.norm(delta) * delta
    return delta


# # # # ####cubic_regularization#######

total_hist_f = []
total_hist_f_adapt = []
total_hist_f1 = []
total_hist_grad = []
for i in range(max_Instance):
    # SCRN
    w_init = 30  # 10*np.random.uniform(-1,1)
    hist_w, hist_f, hist_grad = cubic_regularization(eps, batch_size, w_init)
    total_hist_f.append(hist_f)
    total_hist_grad.append(np.abs(hist_grad))
    # SCRN_adaptive
    hist_w, hist_f, hist_grad = cubic_regularization_adaptive(eps, w_init)
    total_hist_f_adapt.append(hist_f)
    # SGD
    w0, hist_w1, hist_oracle, f_history = SGD(w_init, eps)
    total_hist_f1.append(f_history)
    # print(hist_w)
    # print(hist_f)
    print("Instance:", i)

mean_hist_f = np.mean(np.array(total_hist_f), axis=0)
mean_hist_f_adapt = np.mean(np.array(total_hist_f_adapt), axis=0)
mean_hist_grad = np.mean(np.array(total_hist_grad), axis=0)
mean_hist_f_SGD = np.mean(np.array(total_hist_f1), axis=0)

plt.plot([batch_size * (i + 1) for i in range(len(mean_hist_f))], mean_hist_f, color='green', label="SCRN")
plt.plot([(i + 1) for i in range(len(mean_hist_f_adapt))], mean_hist_f_adapt, color='blue', label="SCRN_adapt")
plt.plot([(i + 1) for i in range(len(mean_hist_f_SGD))], mean_hist_f_SGD, color='red', label="SGD")
plt.xlabel('Number of calls')
plt.ylabel('E[F(x)]-F*')
plt.grid(True)
plt.legend(loc="upper right")
ax = plt.gca()
# ax.set_xlim([left, right])
# ax.set_ylim([ymin,ymax])
# plt.xscale('log')
plt.yscale('log')

plt.savefig('Figures/Batch_free_PL_2_multiplicative_noise_' + dt_string + '.pdf')
# Store data (serialize)
with open('Params/Batch_free_PL_2_multiplicative_noise_' + dt_string + '.json', 'w') as f:
    json.dump(param, f)

# with open('Params/Batch_free_PL_2_multiplicative_noise_'+dt_string+'.pickle', 'wb') as handle:
#     pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
