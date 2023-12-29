import argparse
import importlib
import os
import sys

import time

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import numpy as np
import torch as th
import sympy as sp
import math

from scipy import optimize
from scipy.integrate import nquad, quad
import yaml

import torch.nn as nn
import torch.nn.functional as F

from docplex.mp.model import Model

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import auto_LiRPA.operators.nonlinear as NonLinear

import warnings
warnings.filterwarnings("ignore")

train_diameter = 3.0

UB_CONST_GLOBAL_D_E = train_diameter + 0.1
UB_CONST_GLOBAL_THETA_E = train_diameter + 0.1
err_origin = min(train_diameter * 0.05, 0.1)
N_train_steps = 500

theta_verify_list1 = list(np.arange(-train_diameter, 0.0, 0.5)) + [-err_origin]
theta_verify_list2 = [err_origin] + list(np.arange(0.5, train_diameter + 0.5, 0.5))

env_params = {
    "speed": 2.0,
    "length": 1.0,
    "radius": 10.0,
    "dt": 0.05,  # seconds between state updates
    "max_force": 0.84 #40 degree steering angle, tan(40 degree)
}

SAMPLE_SIZE = 512
STATE_SPACE_SIZE = 2

#LB_CONST_GLOBAL = 0.1
BoundSin_fun = NonLinear.BoundSin(None, None, None, None)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet,self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.policy(x)

class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet,self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.lyapunov(x)

def dynamics(state, action):
    speed = env_params['speed']
    length = env_params['length']
    radius = env_params['radius']
    dt = env_params['dt']
    max_force = env_params['max_force']

    force = th.clamp(action, min = -max_force, max = max_force)
    d_e = state[:, 0:1]
    theta_e = state[:, 1:2]
    sintheta_e = th.sin(theta_e)
    costheta_e = th.cos(theta_e)

    d_e_acc = speed * sintheta_e
    coef = radius / speed
    theta_e_acc = (speed * force / length) - (costheta_e / (coef - sintheta_e))
    d_e_next = d_e + dt * d_e_acc
    theta_e_next = theta_e + dt * theta_e_acc
    return th.cat((d_e_next, theta_e_next), dim = 1)

def lyap_diff(policy_model, lyap_model, state):
    lyap_value = lyap_model(state)
    action = policy_model(state)
    state_next = dynamics(state, action)
    lyap_value_next = lyap_model(state_next)
    lyap_value_diff = lyap_value_next - lyap_value
    reg = th.sqrt(state[:,0:1]**2 + state[:,1:2]**2)
    return lyap_value_diff, th.clamp(reg, max = 0.0005)

def gradient_lyap_diff(policy_model, lyap_model, state): #to find counterexample of energy increase
    lyap_diff_value, _ = lyap_diff(policy_model, lyap_model, state)
    target_func = th.sum(lyap_diff_value)
    policy_model.zero_grad()
    lyap_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return th.sign(grad)

def gradient_lyap_value(lyap_model, state): #to find counterexample of negative energy
    lyap_value = lyap_model(state)
    target_func = th.sum(-lyap_value)
    lyap_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return th.sign(grad)

def get_lower_upper_bound(LB, UB):
    # for function cos(theta)/(a - sin(theta))
    speed = env_params['speed']
    radius = env_params['radius']
    coef = radius / speed
    theta_e_sp = sp.symbols('theta_e_sp')
    func = sp.cos(theta_e_sp) / (coef - sp.sin(theta_e_sp))

    theta_e_opt_sp = sp.symbols('theta_e_opt_sp')
    func_diff_sp = func.diff(theta_e_sp)

    bound_func_diff = func.subs(theta_e_sp, theta_e_opt_sp) + func_diff_sp.subs(theta_e_sp, theta_e_opt_sp) * (theta_e_sp - theta_e_opt_sp) - func
    bound_func_diff2 = sp.lambdify((theta_e_sp, theta_e_opt_sp), bound_func_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda theta_e_opt_sp: nquad(bound_func_diff2, [[LB, UB]], args=(theta_e_opt_sp,))[0]

    result = optimize.minimize_scalar(bound_func_int, bounds=(LB, UB), method='bounded')
    if result.success:
        theta_e_opt = result.x
    ub_slope = func_diff_sp.subs(theta_e_sp, theta_e_opt)
    ub_bias = func.subs(theta_e_sp, theta_e_opt) - theta_e_opt * ub_slope

    lb_slope = (func.subs(theta_e_sp, LB) - func.subs(theta_e_sp, UB))/(LB - UB)
    lb_bias = func.subs(theta_e_sp, LB) - lb_slope * LB
    lb_slope = lb_slope.evalf()
    lb_bias = lb_bias.evalf()
    return float(lb_slope), float(lb_bias), float(ub_slope), float(ub_bias)


def FindCounterExamples(policy_model, lyap_model, device, steps):
    delta = th.zeros(SAMPLE_SIZE, STATE_SPACE_SIZE).uniform_(-1, 1)
    min_state = delta * th.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E])
    min_state = min_state.to(device)
    relative_step_size = 1/steps
    for _ in range(steps):
        min_state.requires_grad = True
        min_state = min_state + relative_step_size * gradient_lyap_diff(policy_model, lyap_model, min_state) 
        min_state = min_state.detach()
        min_state[:, 0:1] = th.clamp(min_state[:, 0:1], min = -UB_CONST_GLOBAL_D_E, max = UB_CONST_GLOBAL_D_E)
        min_state[:, 1:2] = th.clamp(min_state[:, 1:2], min = -UB_CONST_GLOBAL_THETA_E, max = UB_CONST_GLOBAL_THETA_E)

    lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, min_state)
    min_state_return1 = min_state[lyap_value_diff.flatten() >= 0].clone().detach()
    
    delta = th.zeros(SAMPLE_SIZE, STATE_SPACE_SIZE).uniform_(-1, 1)
    min_state = delta * th.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E])
    min_state = min_state.to(device)
    #steps = 30
    x_0 = th.zeros(1, 2).to(device)
    relative_step_size = 1/steps
    for _ in range(steps):
        min_state.requires_grad = True
        min_state = min_state + relative_step_size * gradient_lyap_value(lyap_model, min_state) 
        min_state = min_state.detach()
        min_state[:, 0:1] = th.clamp(min_state[:, 0:1], min = -UB_CONST_GLOBAL_D_E, max = UB_CONST_GLOBAL_D_E)
        min_state[:, 1:2] = th.clamp(min_state[:, 1:2], min = -UB_CONST_GLOBAL_THETA_E, max = UB_CONST_GLOBAL_THETA_E)

    lyap_value_pos = lyap_model(min_state) - lyap_model(x_0)
    min_state_return2 = min_state[lyap_value_pos.flatten() <= 0].clone().detach()

    min_state_return = th.cat((min_state_return1, min_state_return2), dim = 0)
    return min_state_return

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, th.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, th.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, upper, lower):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    upper/lower: list
    '''
    bigM = 0
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
            bigM = np.max((bigM, th.max(th.abs(upper)).item(), th.max(th.abs(lower)).item()))
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
            bigM = np.max((bigM, th.max(th.abs(upper)).item(), th.max(th.abs(lower)).item()))
        else:
            print('Unsupported layer:', type(layer))
    return int(bigM + 1)


def network_bounds_orig(model, upper, lower):
    '''
    get interval bound progation upper and lower bounds for the activation of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            upper, lower = weighted_bound(layer, upper, lower)
        else:
            print('Unsupported layer:', type(layer))
    return upper, lower

def certify(policy_model, lyap_model, LB_d_e, UB_d_e, LB_theta_e, UB_theta_e): #this applies to [-0.05, 0.05]
    x_orig = th.zeros([1, STATE_SPACE_SIZE])
    origin_val = lyap_model(x_orig).item()
    speed = env_params['speed']
    length = env_params['length']
    dt = env_params['dt']
    max_force = env_params['max_force']
    radius = env_params['radius']
    satisfy_cond_flag = True

    print("LB_d_e:", round(LB_d_e, 8), "UB_d_e:", round(UB_d_e, 8), "LB_theta_e:", round(LB_theta_e, 8), "UB_theta_e:", round(UB_theta_e, 8))

    # load weights to numpy for CPLEX
    policy_w1 = policy_model.state_dict()['policy.0.weight'].data.cpu().numpy()
    policy_b1 = policy_model.state_dict()['policy.0.bias'].data.cpu().numpy()
    policy_w2 = policy_model.state_dict()['policy.2.weight'].data.cpu().numpy()
    policy_b2 = policy_model.state_dict()['policy.2.bias'].data.cpu().numpy()
    policy_w3 = policy_model.state_dict()['policy.4.weight'].data.cpu().numpy()
    policy_b3 = policy_model.state_dict()['policy.4.bias'].data.cpu().numpy()

    lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
    lyapunov_b1 = lyap_model.state_dict()['lyapunov.0.bias'].data.cpu().numpy()
    lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
    lyapunov_b2 = lyap_model.state_dict()['lyapunov.2.bias'].data.cpu().numpy()
    lyapunov_w3 = lyap_model.state_dict()['lyapunov.4.weight'].data.cpu().numpy()
    lyapunov_b3 = lyap_model.state_dict()['lyapunov.4.bias'].data.cpu().numpy()

    lyap_cplex_model = Model(name='Lyapunov Verification')
    
    lower_bound_state = th.FloatTensor([[LB_d_e, LB_theta_e]])
    upper_bound_state = th.FloatTensor([[UB_d_e, UB_theta_e]])
    u_upper, u_lower = network_bounds_orig(policy_model.policy, upper_bound_state, lower_bound_state)
    u_abs_max = np.max((np.abs(u_upper.item()), np.abs(u_lower.item())))

    theta_dot_max = (speed * u_abs_max / length) + (speed / (radius - speed))
    upper_bound_ibp = th.FloatTensor([UB_d_e + speed * dt, UB_theta_e + theta_dot_max * dt])
    lower_bound_ibp = th.FloatTensor([LB_d_e - speed * dt, LB_theta_e - theta_dot_max * dt])

    M = network_bounds(lyap_model.lyapunov, upper_bound_ibp, lower_bound_ibp)
    M_policy = network_bounds(policy_model.policy, upper_bound_state, lower_bound_state)

    x_0 = {(i): lyap_cplex_model.continuous_var(name='x_0_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(2)} #lb = 0 by default
    #bound the input variables (x, x_dot, theta, theta_dot)

    lyap_cplex_model.add_constraint((x_0[0]) <= UB_d_e)
    lyap_cplex_model.add_constraint((x_0[1]) <= UB_theta_e)
    lyap_cplex_model.add_constraint((x_0[0]) >= LB_d_e)
    lyap_cplex_model.add_constraint((x_0[1]) >= LB_theta_e)
    
    # get the final policy (force) is the action
    x_1 = {(i): lyap_cplex_model.continuous_var(name='x_1_{}'.format(i)) for i in range(8)}
    s_1 = {(i): lyap_cplex_model.continuous_var(name='s_1_{}'.format(i)) for i in range(8)}
    zx_1 = {(i): lyap_cplex_model.binary_var(name='zx_1_{}'.format(i)) for i in range(8)}
    x_2 = {(i): lyap_cplex_model.continuous_var(name='x_2_{}'.format(i)) for i in range(8)}
    s_2 = {(i): lyap_cplex_model.continuous_var(name='s_2_{}'.format(i)) for i in range(8)}
    zx_2 = {(i): lyap_cplex_model.binary_var(name='zx_2_{}'.format(i)) for i in range(8)}
    force_mid = lyap_cplex_model.continuous_var(name='force_mid', lb = -lyap_cplex_model.infinity)
    force = lyap_cplex_model.continuous_var(name='force', lb = -lyap_cplex_model.infinity)

    for i in range(8):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(policy_w1[i][j] * x_0[j] for j in range(2)) + policy_b1[i] == x_1[i] - s_1[i])
        lyap_cplex_model.add_constraint(x_1[i] <= M_policy * zx_1[i])
        lyap_cplex_model.add_constraint(s_1[i] <= M_policy * (1 - zx_1[i]))
    
    for i in range(8):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(policy_w2[i][j] * x_1[j] for j in range(8)) + policy_b2[i] == x_2[i] - s_2[i])
        lyap_cplex_model.add_constraint(x_2[i] <= M_policy * zx_2[i])
        lyap_cplex_model.add_constraint(s_2[i] <= M_policy * (1 - zx_2[i]))

    lyap_cplex_model.add_constraint(force_mid == lyap_cplex_model.sum(x_2[j] * policy_w3[0][j] for j in range(8)) + policy_b3[0])
    lyap_cplex_model.add_constraint(force == lyap_cplex_model.min(lyap_cplex_model.max(force_mid, -max_force), max_force))

    d_e_acc = lyap_cplex_model.continuous_var(name='d_e_acc', lb = -lyap_cplex_model.infinity)
    theta_e_acc = lyap_cplex_model.continuous_var(name='theta_e_acc', lb = -lyap_cplex_model.infinity)

    lower_slope, lower_bias, upper_slope, upper_bias = BoundSin_fun.bound_relax_impl(th.FloatTensor([LB_theta_e]),th.FloatTensor([UB_theta_e]))
    lower_slope = lower_slope.item()
    lower_bias = lower_bias.item()
    upper_slope = upper_slope.item()
    upper_bias = upper_bias.item()
    lyap_cplex_model.add_constraint(d_e_acc >= speed * (lower_slope * x_0[1] + lower_bias))
    lyap_cplex_model.add_constraint(d_e_acc <= speed * (upper_slope * x_0[1] + upper_bias))

    lower_slope_func, lower_bias_func, upper_slope_func, upper_bias_func = get_lower_upper_bound(LB_theta_e, UB_theta_e)

    lyap_cplex_model.add_constraint(theta_e_acc <= speed * force / length - (lower_slope_func * x_0[1] + lower_bias_func))
    lyap_cplex_model.add_constraint(theta_e_acc >= speed * force / length - (upper_slope_func * x_0[1] + upper_bias_func))

    x_0_next = {(i): lyap_cplex_model.continuous_var(name = 'x_0_next_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(2)} #lb = 0 by default

    lyap_cplex_model.add_constraint(x_0_next[0] == x_0[0] + d_e_acc * dt)
    lyap_cplex_model.add_constraint(x_0_next[1] == x_0[1] + theta_e_acc * dt)

    # get the lyapunov_function_value for (x, x_dot, theta, theta_dot)
    lyx_1 = {(i): lyap_cplex_model.continuous_var(name='lyx_1_{}'.format(i), lb = 0) for i in range(16)}
    lys_1 = {(i): lyap_cplex_model.continuous_var(name='lys_1_{}'.format(i), lb = 0) for i in range(16)}
    lyzx_1 = {(i): lyap_cplex_model.binary_var(name='lyzx_1_{}'.format(i)) for i in range(16)}
    lyx_2 = {(i): lyap_cplex_model.continuous_var(name='lyx_2_{}'.format(i), lb = 0) for i in range(16)}
    lys_2 = {(i): lyap_cplex_model.continuous_var(name='lys_2_{}'.format(i), lb = 0) for i in range(16)}
    lyzx_2 = {(i): lyap_cplex_model.binary_var(name='lyzx_2_{}'.format(i)) for i in range(16)}
    lyp_val_x0 = lyap_cplex_model.continuous_var(name='lyp_val_x0', lb = -lyap_cplex_model.infinity)

    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0[j] for j in range(2)) + lyapunov_b1[i] == lyx_1[i] - lys_1[i])#+ lyapunov_b1[i] == lyx_1[i] - lys_1[i])
        lyap_cplex_model.add_constraint(lyx_1[i] <= M * lyzx_1[i])
        lyap_cplex_model.add_constraint(lys_1[i] <= M * (1 - lyzx_1[i]))
    
    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1[j] for j in range(16)) + lyapunov_b2[i] == lyx_2[i] - lys_2[i])#+ lyapunov_b2[i] == lyx_2[i] - lys_2[i])
        lyap_cplex_model.add_constraint(lyx_2[i] <= M * lyzx_2[i])
        lyap_cplex_model.add_constraint(lys_2[i] <= M * (1 - lyzx_2[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0 == lyap_cplex_model.sum(lyx_2[j] * lyapunov_w3[0][j] for j in range(16)) + lyapunov_b3[0])# + lyapunov_b3[0])
    
    # get the lyapunov_function_value for (x_next, x_dot_next, theta_next, theta_dot_next)
    lyx_1_next = {(i): lyap_cplex_model.continuous_var(name='lyx_1_next_{}'.format(i)) for i in range(16)}
    lys_1_next = {(i): lyap_cplex_model.continuous_var(name='lys_1_next_{}'.format(i)) for i in range(16)}
    lyzx_1_next = {(i): lyap_cplex_model.binary_var(name='lyzx_1_next_{}'.format(i)) for i in range(16)}
    lyx_2_next = {(i): lyap_cplex_model.continuous_var(name='lyx_2_next_{}'.format(i)) for i in range(16)}
    lys_2_next = {(i): lyap_cplex_model.continuous_var(name='lys_2_next_{}'.format(i)) for i in range(16)}
    lyzx_2_next = {(i): lyap_cplex_model.binary_var(name='lyzx_2_next_{}'.format(i)) for i in range(16)}
    lyp_val_x0_next = lyap_cplex_model.continuous_var(name='lyp_val_x0_next', lb = -lyap_cplex_model.infinity)

    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0_next[j] for j in range(2)) + lyapunov_b1[i] == lyx_1_next[i] - lys_1_next[i])
        lyap_cplex_model.add_constraint(lyx_1_next[i] <= M * lyzx_1_next[i])
        lyap_cplex_model.add_constraint(lys_1_next[i] <= M * (1 - lyzx_1_next[i]))
    
    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1_next[j] for j in range(16)) + lyapunov_b2[i] == lyx_2_next[i] - lys_2_next[i])
        lyap_cplex_model.add_constraint(lyx_2_next[i] <= M * lyzx_2_next[i])
        lyap_cplex_model.add_constraint(lys_2_next[i] <= M * (1 - lyzx_2_next[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0_next == lyap_cplex_model.sum(lyx_2_next[j] * lyapunov_w3[0][j] for j in range(16)) + lyapunov_b3[0])
    
    lyap_cplex_model.minimize(lyp_val_x0 - origin_val)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()
    sol_print = round(solution_.objective_value, 8)
    if sol_print < 0:
        satisfy_cond_flag = False

    counter_example1 = solution_.get_values([x_0[0], x_0[1]])

    lyap_cplex_model.maximize(lyp_val_x0_next - lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()

    counter_example2 = solution_.get_values([x_0[0], x_0[1]])

    sol_print = round(solution_.objective_value, 8)
    if sol_print > -0.00001:
        satisfy_cond_flag = False 
    print(satisfy_cond_flag)
    return satisfy_cond_flag, counter_example1, counter_example2

def calc_cut_value_min_outside(lyap_model):
    lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
    lyapunov_b1 = lyap_model.state_dict()['lyapunov.0.bias'].data.cpu().numpy()
    lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
    lyapunov_b2 = lyap_model.state_dict()['lyapunov.2.bias'].data.cpu().numpy()
    lyapunov_w3 = lyap_model.state_dict()['lyapunov.4.weight'].data.cpu().numpy()
    lyapunov_b3 = lyap_model.state_dict()['lyapunov.4.bias'].data.cpu().numpy()

    lyap_cplex_model = Model(name='Lyapunov Verification')

    speed = env_params['speed']
    length = env_params['length']
    dt = env_params['dt']
    radius = env_params['radius']
    max_force = env_params['max_force']

    outer_val = np.max((speed, (speed * max_force / length) + (speed / (radius - speed)))) * dt
    scale = (train_diameter + outer_val)/train_diameter
    lower_bound_state = th.FloatTensor([[-train_diameter * scale, -train_diameter * scale]])
    upper_bound_state = th.FloatTensor([[train_diameter * scale, train_diameter * scale]])
    M = network_bounds(lyap_model.lyapunov, upper_bound_state, lower_bound_state)

    x_0 = {(i): lyap_cplex_model.continuous_var(name='x_0_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(2)} #lb = 0 by default

    lyap_cplex_model.add_constraint(x_0[0] <= train_diameter * scale)
    lyap_cplex_model.add_constraint(x_0[1] <= train_diameter * scale)
    lyap_cplex_model.add_constraint(x_0[0] >= -train_diameter * scale)
    lyap_cplex_model.add_constraint(x_0[1] >= -train_diameter * scale)

    lyap_cplex_model.add_constraint(lyap_cplex_model.max(lyap_cplex_model.abs(x_0[0]), lyap_cplex_model.abs(x_0[1])) >= train_diameter)

    lyx_1 = {(i): lyap_cplex_model.continuous_var(name='lyx_1_{}'.format(i), lb = 0) for i in range(16)}
    lys_1 = {(i): lyap_cplex_model.continuous_var(name='lys_1_{}'.format(i), lb = 0) for i in range(16)}
    lyzx_1 = {(i): lyap_cplex_model.binary_var(name='lyzx_1_{}'.format(i)) for i in range(16)}
    lyx_2 = {(i): lyap_cplex_model.continuous_var(name='lyx_2_{}'.format(i), lb = 0) for i in range(16)}
    lys_2 = {(i): lyap_cplex_model.continuous_var(name='lys_2_{}'.format(i), lb = 0) for i in range(16)}
    lyzx_2 = {(i): lyap_cplex_model.binary_var(name='lyzx_2_{}'.format(i)) for i in range(16)}
    lyp_val_x0 = lyap_cplex_model.continuous_var(name='lyp_val_x0', lb = -lyap_cplex_model.infinity)

    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0[j] for j in range(2)) + lyapunov_b1[i] == lyx_1[i] - lys_1[i])
        lyap_cplex_model.add_constraint(lyx_1[i] <= M * lyzx_1[i])
        lyap_cplex_model.add_constraint(lys_1[i] <= M * (1 - lyzx_1[i]))
    
    for i in range(16):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1[j] for j in range(16)) + lyapunov_b2[i] == lyx_2[i] - lys_2[i])
        lyap_cplex_model.add_constraint(lyx_2[i] <= M * lyzx_2[i])
        lyap_cplex_model.add_constraint(lys_2[i] <= M * (1 - lyzx_2[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0 == lyap_cplex_model.sum(lyx_2[j] * lyapunov_w3[0][j] for j in range(16)) + lyapunov_b3[0])
    
    lyap_cplex_model.minimize(lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()

    return solution_.objective_value


def calc_cut_value(lyap_model):
    min_outside = calc_cut_value_min_outside(lyap_model)
    return min_outside - 0.000001

def certify_list_sub(policy_model, lyap_model, certify_counter_example, certify_list):
    success_flag = True
    is_success = False
    for element in certify_list:
        result, counter_example1, counter_example2 = certify(policy_model, lyap_model, element[0], element[1], element[2], element[3])
        if not result:
            success_flag = False
            certify_counter_example.append(th.FloatTensor([counter_example1]))
            certify_counter_example.append(th.FloatTensor([counter_example2]))

    if success_flag:
        certify_counter_example, not_certified_list = certify_list_all(policy_model, lyap_model, certify_counter_example)
        if len(not_certified_list) == 0:
            is_success = True

    return certify_counter_example, is_success

def certify_list_all(policy_model, lyap_model, certify_counter_example):
    not_certified_list = []

    for idx in range(len(theta_verify_list1) - 1):
        lb = theta_verify_list1[idx]
        ub = theta_verify_list1[idx + 1]
        result, counter_example1, counter_example2 = certify(policy_model, lyap_model, -train_diameter, train_diameter, lb, ub)
        if not result:
            certify_counter_example.append(th.FloatTensor([counter_example1]))
            certify_counter_example.append(th.FloatTensor([counter_example2]))
            not_certified_list.append([-train_diameter, train_diameter, lb, ub])

    for idx in range(len(theta_verify_list2) - 1):
        lb = theta_verify_list2[idx]
        ub = theta_verify_list2[idx + 1]
        result, counter_example1, counter_example2 = certify(policy_model, lyap_model, -train_diameter, train_diameter, lb, ub)
        if not result:
            certify_counter_example.append(th.FloatTensor([counter_example1]))
            certify_counter_example.append(th.FloatTensor([counter_example2]))
            not_certified_list.append([-train_diameter, train_diameter, lb, ub])

    result, counter_example1, counter_example2 = certify(policy_model, lyap_model, err_origin, train_diameter, -err_origin, err_origin)
    if not result:
        certify_counter_example.append(th.FloatTensor([counter_example1]))
        certify_counter_example.append(th.FloatTensor([counter_example2]))
        not_certified_list.append([err_origin, train_diameter, -err_origin, err_origin])

    result, counter_example1, counter_example2 = certify(policy_model, lyap_model, -train_diameter, -err_origin, -err_origin, err_origin)
    if not result:
        certify_counter_example.append(th.FloatTensor([counter_example1]))
        certify_counter_example.append(th.FloatTensor([counter_example2]))
        not_certified_list.append([-train_diameter, -err_origin, -err_origin, err_origin])

    return certify_counter_example, not_certified_list

def train_lyapunov():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)

    args = parser.parse_args()


    policy_model = PolicyNet()
    saved_state = th.load(
        "./ppo/path_tracking_pretrain_policy.pt",
        map_location=lambda storage, loc: storage)
    if "model_state_dict" in saved_state.keys():
        saved_state = saved_state['model_state_dict']
    policy_model.load_state_dict(saved_state)

    lyap_model = LyapunovNet()
    
    total_iter_all = 0
    training_total_time = 0
    training_start_time = time.time()
    for _ in range(1):
        N = 500             # sample size
        iter_num = 0 
        max_iters = N_train_steps
        learning_rate = 0.01
        optimizer = th.optim.Adam(list(lyap_model.parameters()), lr=learning_rate)
        valid_num = 0

        x = th.zeros(N, STATE_SPACE_SIZE).uniform_(-1, 1) * th.FloatTensor(th.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E]))
        x_all = x.clone()
        x_0 = th.zeros([1, STATE_SPACE_SIZE])
        
        while (iter_num < max_iters):
            total_iter_all += 1
            lyap_value_diff, reg = lyap_diff(policy_model, lyap_model, x)
            V0 = lyap_model(x_0)
            lyap_pos = lyap_model(x) - V0
            Lyapunov_risk = (F.relu(-lyap_pos + 0.0001 * reg) + 1.2*F.relu(lyap_value_diff + 0.12)).mean()+ 1.2*(V0).pow(2)

            #print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 

            optimizer.zero_grad()
            Lyapunov_risk.backward()
            optimizer.step() 

            if iter_num % 10 == 0:
                result = FindCounterExamples(policy_model, lyap_model, 'cpu', 30)
                if (len(result) != 0): 
                    pass
                    #print("Not a Lyapunov function. Found counterexample: ")
                    #print(result)
                    #print(result.size())
                else:
                    valid_num += 1
                    #print("Satisfy conditions!! Iter:", valid_num)
                #print('==============================') 

                indices = np.random.choice(x_all.size()[0], size=min(x_all.size()[0], 512))
                x = th.cat((result, x_all[indices, :]), dim = 0)
                x_all = th.cat((result, x_all), dim = 0)
            iter_num += 1

    training_total_time += time.time() - training_start_time

    
    certify_counter_example = []
    is_success_all = False
    loop_iter_num = 0
    while not is_success_all:
        training_start_time = time.time()
        N = 500             # sample size
        iter_num = 0 
        max_iters = N_train_steps
        learning_rate = 0.01
        optimizer = th.optim.Adam(list(policy_model.parameters()) + list(lyap_model.parameters()), lr=learning_rate)
        valid_num = 0

        x = th.zeros(N, STATE_SPACE_SIZE).uniform_(-1, 1) * th.FloatTensor(th.FloatTensor([UB_CONST_GLOBAL_D_E, UB_CONST_GLOBAL_THETA_E]))
        if len(certify_counter_example) != 0:
            certify_counter_example = th.cat(certify_counter_example, dim = 0)
            x = th.cat((x, certify_counter_example), dim = 0)
            certify_counter_example= []
        x_all = x.clone()
        x_0 = th.zeros([1, STATE_SPACE_SIZE])

        while (iter_num < max_iters):
            total_iter_all += 1

            lyap_value_diff, reg = lyap_diff(policy_model, lyap_model, x)
            V0 = lyap_model(x_0)
            lyap_pos = lyap_model(x) - V0
            Lyapunov_risk = (F.relu(-lyap_pos + 0.0001 * reg) + 1.2*F.relu(lyap_value_diff + 0.12)).mean()+ 1.2*(V0).pow(2)

            optimizer.zero_grad()
            Lyapunov_risk.backward()
            optimizer.step() 

            if iter_num % 10 == 0:
                result = FindCounterExamples(policy_model, lyap_model, 'cpu', 30)
                if (len(result) != 0): 
                    pass
                    #print("Not a Lyapunov function. Found counterexample: ")
                    #print(result.size())
                else:
                    valid_num += 1
                    #print("Satisfy conditions!! Iter:", valid_num)
                #print('==============================') 

                indices = np.random.choice(x_all.size()[0], size=min(x_all.size()[0], 512))
                x = th.cat((result, x_all[indices, :]), dim = 0)
                x_all = th.cat((result, x_all), dim = 0)
            iter_num += 1

            if (iter_num % 10 == 0) and (Lyapunov_risk.item() == 0):
                break

        training_total_time += time.time() - training_start_time

        certify_start_time = time.time()
        if loop_iter_num % 10 == 0:
            certify_counter_example, not_certified_list = certify_list_all(policy_model, lyap_model, certify_counter_example)
            if len(not_certified_list) == 0:
                is_success_all = True
        else:
            certify_counter_example, is_success_all = certify_list_sub(policy_model, lyap_model, certify_counter_example, not_certified_list)

        print("counterexamples added:", len(certify_counter_example))

        certify_total_time += time.time() - certify_start_time
        loop_iter_num = loop_iter_num + 1

    X_expand = np.linspace(-train_diameter, train_diameter, 2000)
    Y_expand = np.linspace(-train_diameter, train_diameter, 2000)
    x1_expand, x2_expand = np.meshgrid(X_expand,Y_expand)

    lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
    lyapunov_b1 = lyap_model.state_dict()['lyapunov.0.bias'].data.cpu().numpy()
    lyapunov_b1 = np.expand_dims(lyapunov_b1, axis=(1,2))

    lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
    lyapunov_b2 = lyap_model.state_dict()['lyapunov.2.bias'].data.cpu().numpy()
    lyapunov_b2 = np.expand_dims(lyapunov_b2, axis=(1,2))

    lyapunov_w3 = lyap_model.state_dict()['lyapunov.4.weight'].data.cpu().numpy()
    lyapunov_b3 = lyap_model.state_dict()['lyapunov.4.bias'].data.cpu().numpy()
    lyapunov_b3 = np.expand_dims(lyapunov_b3, axis=(1,2))

    layer1_result =  np.maximum(np.tensordot(lyapunov_w1, np.array([x1_expand, x2_expand]), axes = 1) + lyapunov_b1, 0)
    layer2_result =  np.maximum(np.tensordot(lyapunov_w2, layer1_result, axes = 1) + lyapunov_b2, 0)
    V_NN = (np.tensordot(lyapunov_w3, layer2_result, axes = 1) + lyapunov_b3)[0]

    cut_value = calc_cut_value(lyap_model)
    
    region_area = np.sum(V_NN <= cut_value)/(2000 * 2000)
    roa_area = region_area * train_diameter * train_diameter * 4

    f = open("path_tracking_result.txt", "a")
    f.write("seed:"+ str(args.seed) + ":training_time:" + str(training_total_time)+ ":certify_time:"+ str(certify_total_time)+ ":total_iter_all:" + str(total_iter_all) + ":ROA_area:"+ str(roa_area)+ ":certify_success:"+ str(is_success_all)+"\n")
    f.close()

    state_to_save_lyap = {'model_state_dict':lyap_model.state_dict()}
    state_to_save_policy = {'model_state_dict':policy_model.state_dict()}
    th.save(state_to_save_lyap, "path_tracking_state_to_save_lyap_" + str(int(args.seed)) + ".pt")
    th.save(state_to_save_policy, "path_tracking_state_to_save_policy_" + str(int(args.seed)) + ".pt")
    
if __name__ == "__main__":
    train_lyapunov()
