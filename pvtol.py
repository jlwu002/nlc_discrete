import argparse
import sys

import numpy as np
import torch as th
import sympy as sp

from scipy import optimize
from scipy.integrate import nquad, quad

import time
import random
import torch.nn as nn
import torch.nn.functional as F

from docplex.mp.model import Model

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import auto_LiRPA.operators.nonlinear as NonLinear

import math
import warnings
warnings.filterwarnings("ignore")

# adjusted dynamics
env_params = {
    "mass": 4.0,
    "inertia": 0.0475,
    "dist": 0.25,
    "gravity": 9.8,
    "dt": 0.05,  # seconds between state updates
    "max_force_ub": 39.2,
    "max_force_lb": 0,
}

SAMPLE_SIZE = 1024
STATE_SPACE_SIZE = 6

MID_LAYER_SIZE = 32

train_diameter = 1.0

UB_CONST_GLOBAL_x = train_diameter
UB_CONST_GLOBAL_y = train_diameter
UB_CONST_GLOBAL_theta = train_diameter
UB_CONST_GLOBAL_x_d = train_diameter
UB_CONST_GLOBAL_y_d = train_diameter
UB_CONST_GLOBAL_theta_d = train_diameter

err_origin = 0.1
CONST_C = 0.12

BoundSin_fun = NonLinear.BoundSin(None, None, None, None)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet,self).__init__()
        self.policy = nn.Linear(STATE_SPACE_SIZE, 1, bias = False)
        self.policy.weight = th.nn.Parameter(th.FloatTensor([[0.70710678, -0.70710678, -5.03954871,  1.10781077, -1.82439774, -1.20727555],
       [-0.70710678, -0.70710678,  5.03954871, -1.10781077, -1.82439774, 1.20727555]])) #LQR solution

    def forward(self, x):
        return self.policy(x)

class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet,self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, MID_LAYER_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(MID_LAYER_SIZE, MID_LAYER_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(MID_LAYER_SIZE, 1, bias=False)
        )

    def forward(self,x):
        return self.lyapunov(x)

def dynamics(state, action):
    mass = env_params['mass']
    inertia = env_params['inertia']
    dist = env_params['dist']
    gravity = env_params['gravity']
    dt = env_params['dt']
    max_force_ub_const = env_params['max_force_ub']
    max_force_lb_const = env_params['max_force_lb']

    u_1 = th.clamp(action[:, 0:1] + mass * gravity / 2.0, min = max_force_lb_const, max = max_force_ub_const) 
    u_2 = th.clamp(action[:, 1:2] + mass * gravity / 2.0, min = max_force_lb_const, max = max_force_ub_const)

    x_pos = state[:, 0:1]
    y_pos = state[:, 1:2]
    theta_pos = state[:, 2:3]
    x_pos_d = state[:, 3:4]
    y_pos_d = state[:, 4:5]
    theta_pos_d = state[:, 5:6]

    sintheta_pos = th.sin(theta_pos)
    costheta_pos = th.cos(theta_pos)

    x_change = x_pos_d * costheta_pos - y_pos_d * sintheta_pos
    y_change = x_pos_d * sintheta_pos + y_pos_d * costheta_pos
    theta_change = theta_pos_d
    x_dot_change = y_pos_d * theta_pos_d - gravity * sintheta_pos
    y_dot_change = -x_pos_d * theta_pos_d - gravity * costheta_pos + (u_1 + u_2)/mass
    theta_dot_change = (u_1 - u_2) * dist / inertia

    x_pos_next = x_pos + dt * x_change
    y_pos_next = y_pos + dt * y_change
    theta_pos_next = theta_pos + dt * theta_change
    x_pos_d_next = x_pos_d + dt * x_dot_change
    y_pos_d_next = y_pos_d + dt * y_dot_change
    theta_pos_d_next = theta_pos_d + dt * theta_dot_change

    return th.cat((x_pos_next, y_pos_next, theta_pos_next, x_pos_d_next, y_pos_d_next, theta_pos_d_next), dim = 1)

def lyap_diff(policy_model, lyap_model, state):
    lyap_value = lyap_model(state)
    action = policy_model(state)
    state_next = dynamics(state, action)
    lyap_value_next = lyap_model(state_next)
    lyap_value_diff = lyap_value_next - lyap_value
    return lyap_value_diff, state

def gradient_lyap_diff(policy_model, lyap_model, state): #to find counterexample of energy increase
    lyap_diff_value, _ = lyap_diff(policy_model, lyap_model, state)
    target_func = th.sum(lyap_diff_value)
    policy_model.zero_grad()
    lyap_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return th.sign(grad)

def gradient_lyap_value(lyap_model, state): #to find counterexample of negative energy
    #V0 = lyap_model(x_0)
    lyap_value = lyap_model(state)
    target_func = th.sum(-lyap_value)
    lyap_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return th.sign(grad)

def FindCounterExamples(policy_model, lyap_model, device):
    delta = th.zeros(SAMPLE_SIZE, STATE_SPACE_SIZE).uniform_(-1, 1)
    min_state = delta * th.FloatTensor([UB_CONST_GLOBAL_x, UB_CONST_GLOBAL_y, UB_CONST_GLOBAL_theta, UB_CONST_GLOBAL_x_d, UB_CONST_GLOBAL_y_d, UB_CONST_GLOBAL_theta_d])
    min_state = min_state.to(device)
    steps = 30
    relative_step_size = 1/steps
    for _ in range(steps):
        min_state.requires_grad = True
        min_state = min_state + relative_step_size * gradient_lyap_diff(policy_model, lyap_model, min_state) 
        min_state = min_state.detach()
        min_state[:, 0:1] = th.clamp(min_state[:, 0:1], min = -UB_CONST_GLOBAL_x, max = UB_CONST_GLOBAL_x)
        min_state[:, 1:2] = th.clamp(min_state[:, 1:2], min = -UB_CONST_GLOBAL_y, max = UB_CONST_GLOBAL_y)
        min_state[:, 2:3] = th.clamp(min_state[:, 2:3], min = -UB_CONST_GLOBAL_theta, max = UB_CONST_GLOBAL_theta)
        min_state[:, 3:4] = th.clamp(min_state[:, 3:4], min = -UB_CONST_GLOBAL_x_d, max = UB_CONST_GLOBAL_x_d)
        min_state[:, 4:5] = th.clamp(min_state[:, 4:5], min = -UB_CONST_GLOBAL_y_d, max = UB_CONST_GLOBAL_y_d)
        min_state[:, 5:6] = th.clamp(min_state[:, 5:6], min = -UB_CONST_GLOBAL_theta_d, max = UB_CONST_GLOBAL_theta_d)

    lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, min_state)
    min_state_return1 = min_state[lyap_value_diff.flatten() >= -0.0001].clone().detach()
    
    return min_state_return1

def BoundCos_fun(LB, UB):
    lower_slope, lower_bias, upper_slope, upper_bias = BoundSin_fun.bound_relax_impl(LB + 0.5 * th.pi, UB + 0.5 * th.pi)
    return lower_slope, lower_slope * (0.5 * th.pi) + lower_bias, upper_slope, upper_slope * (0.5 * th.pi) + upper_bias

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

def bound_func_cos(lb_x, ub_x, lb_theta, ub_theta): # only applies to [-pi/2, pi/2]
    # bound function x * cos(theta)
    if (lb_x >= ub_x) or (lb_theta >= ub_theta):
        raise Exception("bound is messed up: LB >= UB.")

    if (lb_theta * ub_theta < 0) or (lb_x * ub_x < 0):
        raise Exception("not monotone w.r.t theta in the region")

    if ub_x <= 0: #make sure x is positive
        lb_x_func = -ub_x
        ub_x_func = -lb_x
        alter_sign = True
    else:
        lb_x_func = lb_x
        ub_x_func = ub_x
        alter_sign = False

    x_sp = sp.symbols('x_sp')   
    theta_sp = sp.symbols('theta_sp')

    func1_sp = x_sp * sp.cos(theta_sp)
    x_min = lb_x_func
    x_max = ub_x_func

    if ub_theta > 0:
        costheta_min = np.cos(ub_theta)
        costheta_max = np.cos(lb_theta)
    else:
        costheta_min = np.cos(lb_theta)
        costheta_max = np.cos(ub_theta)
    
    lower_slope_cos, lower_bias_cos, upper_slope_cos, upper_bias_cos = BoundCos_fun(th.FloatTensor([lb_theta]),th.FloatTensor([ub_theta]))

    coef = sp.symbols('coef')
    func1_sp_lb = coef * (x_min * (lower_slope_cos * theta_sp + lower_bias_cos))  + (1 - coef) * (x_sp * costheta_min)
    func1_sp_ub = coef * (x_max * (upper_slope_cos * theta_sp + upper_bias_cos))  + (1 - coef) * (x_sp * costheta_max)

    #get upper bound
    func_ub_diff = func1_sp_ub - func1_sp
    func_ub_diff2 = sp.lambdify((x_sp, theta_sp, coef), func_ub_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_ub_diff2, [[lb_x_func, ub_x_func], [lb_theta, ub_theta]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    ub_slope_x = (1 - coef_opt) * costheta_max
    ub_slope_theta = coef_opt * x_max * upper_slope_cos
    ub_bias = coef_opt * x_max * upper_bias_cos

    #get lower bound
    func_lb_diff = func1_sp - func1_sp_lb
    func_lb_diff2 = sp.lambdify((x_sp, theta_sp, coef), func_lb_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_lb_diff2, [[lb_x_func, ub_x_func], [lb_theta, ub_theta]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    lb_slope_x = (1 - coef_opt) * costheta_min
    lb_slope_theta = coef_opt * x_min * lower_slope_cos
    lb_bias = coef_opt * x_min * lower_bias_cos
    
    ub_slope_x = float(ub_slope_x)
    ub_slope_theta = float(ub_slope_theta)
    ub_bias = float(ub_bias)
    lb_slope_x = float(lb_slope_x)
    lb_slope_theta = float(lb_slope_theta)
    lb_bias = float(lb_bias)

    if alter_sign:
        lb_slope_x_final = ub_slope_x
        lb_slope_theta_final = ub_slope_theta
        lb_bias_final = ub_bias

        ub_slope_x_final = lb_slope_x
        ub_slope_theta_final = lb_slope_theta
        ub_bias_final = lb_bias
    else:
        lb_slope_x_final = lb_slope_x
        lb_slope_theta_final = lb_slope_theta
        lb_bias_final = lb_bias

        ub_slope_x_final = ub_slope_x
        ub_slope_theta_final = ub_slope_theta
        ub_bias_final = ub_bias       

    return lb_slope_x_final, lb_slope_theta_final, lb_bias_final, ub_slope_x_final, ub_slope_theta_final, ub_bias_final

def bound_func_sin(lb_x, ub_x, lb_theta, ub_theta): # only applies to [-pi/2, pi/2]
    if (lb_x >= ub_x) or (lb_theta >= ub_theta):
        raise Exception("bound is messed up: LB >= UB.")

    if (lb_theta * ub_theta < 0) or (lb_x * ub_x < 0):
        raise Exception("not monotone w.r.t theta in the region")

    lb_theta_func = lb_theta
    ub_theta_func = ub_theta

    if (ub_x <= 0): #make sure x is positive
        lb_theta_func = ub_theta
        ub_theta_func = lb_theta
    else:
        lb_theta_func = lb_theta
        ub_theta_func = ub_theta

    if ub_theta <= 0: #make sure x is positive
        lb_x_func = ub_x
        ub_x_func = lb_x
    else:
        lb_x_func = lb_x
        ub_x_func = ub_x

    x_sp = sp.symbols('x_sp')   
    theta_sp = sp.symbols('theta_sp')

    func1_sp = x_sp * sp.sin(theta_sp)
    
    sintheta_min = np.sin(lb_theta_func)
    sintheta_max = np.sin(ub_theta_func)
    
    lower_slope_sin, lower_bias_sin, upper_slope_sin, upper_bias_sin = BoundSin_fun.bound_relax_impl(th.FloatTensor([lb_theta_func]),th.FloatTensor([ub_theta_func]))

    coef = sp.symbols('coef')
    func1_sp_lb = coef * (lb_x_func * (lower_slope_sin * theta_sp + lower_bias_sin))  + (1 - coef) * (x_sp * sintheta_min)
    func1_sp_ub = coef * (ub_x_func * (upper_slope_sin * theta_sp + upper_bias_sin))  + (1 - coef) * (x_sp * sintheta_max)

    #get upper bound
    func_ub_diff = func1_sp_ub - func1_sp
    func_ub_diff2 = sp.lambdify((x_sp, theta_sp, coef), func_ub_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_ub_diff2, [[lb_x_func, ub_x_func], [lb_theta_func, ub_theta_func]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    ub_slope_x = (1 - coef_opt) * sintheta_max
    ub_slope_theta = coef_opt * ub_x_func * upper_slope_sin
    ub_bias = coef_opt * ub_x_func * upper_bias_sin

    #get lower bound
    func_lb_diff = func1_sp - func1_sp_lb
    func_lb_diff2 = sp.lambdify((x_sp, theta_sp, coef), func_lb_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_lb_diff2, [[lb_x_func, ub_x_func], [lb_theta_func, ub_theta_func]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    lb_slope_x = (1 - coef_opt) * sintheta_min
    lb_slope_theta = coef_opt * lb_x_func * lower_slope_sin
    lb_bias = coef_opt * lb_x_func * lower_bias_sin
    
    ub_slope_x = float(ub_slope_x)
    ub_slope_theta = float(ub_slope_theta)
    ub_bias = float(ub_bias)
    lb_slope_x = float(lb_slope_x)
    lb_slope_theta = float(lb_slope_theta)
    lb_bias = float(lb_bias)    

    return lb_slope_x, lb_slope_theta, lb_bias, ub_slope_x, ub_slope_theta, ub_bias

def bound_func_xy(lb_x, ub_x, lb_y, ub_y): # only applies to [-pi/2, pi/2]
    if (lb_x >= ub_x) or (lb_y >= ub_y):
        raise Exception("bound is messed up: LB >= UB.")

    if (lb_x * ub_x < 0) or (lb_x * ub_x < 0):
        raise Exception("please do not cross zero when splitting")

    if (ub_x <= 0): #make sure x is positive
        y_min = ub_y
        y_max = lb_y
    else:
        y_min = lb_y
        y_max = ub_y

    if (ub_y <= 0): #make sure x is positive
        x_min = ub_x
        x_max = lb_x
    else:
        x_min = lb_x
        x_max = ub_x

    x_sp = sp.symbols('x_sp')   
    y_sp = sp.symbols('y_sp')

    func1_sp = x_sp * y_sp
    
    coef = sp.symbols('coef')
    func1_sp_lb = coef * x_min * y_sp + (1 - coef) * y_min * x_sp
    func1_sp_ub = coef * x_max * y_sp + (1 - coef) * y_max * x_sp

    #get upper bound
    func_ub_diff = func1_sp_ub - func1_sp
    func_ub_diff2 = sp.lambdify((x_sp, y_sp, coef), func_ub_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_ub_diff2, [[lb_x, ub_x], [lb_y, ub_y]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    ub_slope_x = (1 - coef_opt) * y_max
    ub_slope_y = coef_opt * x_max

    #get lower bound
    func_lb_diff = func1_sp - func1_sp_lb
    func_lb_diff2 = sp.lambdify((x_sp, y_sp, coef), func_lb_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(func_lb_diff2, [[lb_x, ub_x], [lb_y, ub_y]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x

    lb_slope_x = (1 - coef_opt) * y_min
    lb_slope_y = coef_opt * x_min
    
    ub_slope_x = float(ub_slope_x)
    ub_slope_y = float(ub_slope_y)
    lb_slope_x = float(lb_slope_x)
    lb_slope_y = float(lb_slope_y)

    return lb_slope_x, lb_slope_y, ub_slope_x, ub_slope_y

def certify(policy_model, lyap_model, element):
    lb_x = element[0]
    ub_x = element[1]
    lb_y = element[2]
    ub_y = element[3]
    lb_theta = element[4]
    ub_theta = element[5]
    lb_x_d = element[6]
    ub_x_d = element[7]
    lb_y_d = element[8]
    ub_y_d = element[9]
    lb_theta_d = element[10]
    ub_theta_d = element[11]

    mass = env_params['mass']
    inertia = env_params['inertia']
    dist = env_params['dist']
    gravity = env_params['gravity']
    dt = env_params['dt']
    max_force_ub_const = env_params['max_force_ub']
    max_force_lb_const = env_params['max_force_lb']
    
    satisfy_cond_flag = True

    print("bound: ", element) 

    # load weights to numpy for CPLEX
    policy_w1 = policy_model.state_dict()['policy.weight'].data.cpu().numpy()

    lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
    lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
    lyapunov_w3 = lyap_model.state_dict()['lyapunov.4.weight'].data.cpu().numpy()

    lyap_cplex_model = Model(name='Lyapunov Verification')

    sin_theta_1 = np.sin(lb_theta)
    sin_theta_2 = np.sin(ub_theta)
    cos_theta_1 = np.cos(lb_theta)
    cos_theta_2 = np.cos(ub_theta)

    max_sin = np.max((sin_theta_1, sin_theta_2))
    max_cos = np.max((cos_theta_1, cos_theta_2))
    min_sin = np.min((sin_theta_1, sin_theta_2))
    min_cos = np.min((cos_theta_1, cos_theta_2))

    vx_cos = np.array([lb_x_d * cos_theta_1, ub_x_d * cos_theta_1, lb_x_d * cos_theta_2, ub_x_d * cos_theta_2])
    vx_sin = np.array([lb_x_d * sin_theta_1, ub_x_d * sin_theta_1, lb_x_d * sin_theta_2, ub_x_d * sin_theta_2])

    vy_cos = np.array([lb_y_d * cos_theta_1, ub_y_d * cos_theta_1, lb_y_d * cos_theta_2, ub_y_d * cos_theta_2])
    vy_sin = np.array([lb_y_d * sin_theta_1, ub_y_d * sin_theta_1, lb_y_d * sin_theta_2, ub_y_d * sin_theta_2])

    vx_theta_d = np.array([lb_x_d * lb_theta_d, lb_x_d * ub_theta_d, ub_x_d * lb_theta_d, ub_x_d * ub_theta_d])
    vy_theta_d = np.array([lb_y_d * lb_theta_d, lb_y_d * ub_theta_d, ub_y_d * lb_theta_d, ub_y_d * ub_theta_d])

    upper_bound_input = th.FloatTensor([ub_x, ub_y, ub_theta, ub_x_d, ub_y_d, ub_theta_d])
    lower_bound_input = th.FloatTensor([lb_x, lb_y, lb_theta, lb_x_d, lb_y_d, lb_theta_d])

    max_force_ub, max_force_lb = weighted_bound(policy_model.policy, upper_bound_input, lower_bound_input)
    max_force_ub = th.clamp(max_force_ub + mass * gravity / 2.0, min = max_force_lb_const, max = max_force_ub_const)
    max_force_lb = th.clamp(max_force_lb + mass * gravity / 2.0, min = max_force_lb_const, max = max_force_ub_const)

    x_change_ub = np.max(vx_cos) - np.min(vy_sin)
    x_change_lb = np.min(vx_cos) - np.max(vy_sin)

    y_change_ub = np.max(vx_sin) + np.max(vy_cos)
    y_change_lb = np.min(vx_sin) + np.min(vy_cos)

    theta_change_ub = ub_theta_d
    theta_change_lb = lb_theta_d

    xd_change_ub = np.max(vy_theta_d) - gravity * min_sin
    xd_change_lb = np.min(vy_theta_d) - gravity * max_sin

    yd_change_ub = -np.min(vx_theta_d) - gravity * min_cos + th.sum(max_force_ub).item() / mass
    yd_change_lb = -np.max(vx_theta_d) - gravity * max_cos + th.sum(max_force_lb).item() / mass

    u1_m_u2 = nn.Linear(2, 1, bias = False)
    u1_m_u2.weight = th.nn.Parameter(th.FloatTensor([[1, -1]]))
    u1_m_u2_ub, u1_m_u2_lb = weighted_bound(u1_m_u2, max_force_ub, max_force_lb)

    thetad_change_ub = u1_m_u2_ub.item() * dist/inertia
    thetad_change_lb = u1_m_u2_lb.item() * dist/inertia

    upper_bound_ibp = th.FloatTensor([ub_x + x_change_ub * dt, ub_y + y_change_ub * dt, ub_theta + theta_change_ub * dt, \
        ub_x_d + xd_change_ub * dt, ub_y_d + yd_change_ub * dt, ub_theta_d + thetad_change_ub * dt])
    lower_bound_ibp = th.FloatTensor([lb_x + x_change_lb * dt, lb_y + y_change_lb * dt, lb_theta + theta_change_lb * dt, \
        lb_x_d + xd_change_lb * dt, lb_y_d + yd_change_lb * dt, lb_theta_d + thetad_change_lb * dt])

    M = network_bounds(lyap_model.lyapunov, upper_bound_ibp, lower_bound_ibp)
    x_0 = {(i): lyap_cplex_model.continuous_var(name='x_0_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(6)} #lb = 0 by default

    lyap_cplex_model.add_constraint(x_0[0] >= lb_x)
    lyap_cplex_model.add_constraint(x_0[0] <= ub_x)

    lyap_cplex_model.add_constraint(x_0[1] >= lb_y)
    lyap_cplex_model.add_constraint(x_0[1] <= ub_y)

    lyap_cplex_model.add_constraint(x_0[2] >= lb_theta)
    lyap_cplex_model.add_constraint(x_0[2] <= ub_theta)

    lyap_cplex_model.add_constraint(x_0[3] >= lb_x_d)
    lyap_cplex_model.add_constraint(x_0[3] <= ub_x_d)

    lyap_cplex_model.add_constraint(x_0[4] >= lb_y_d)
    lyap_cplex_model.add_constraint(x_0[4] <= ub_y_d)

    lyap_cplex_model.add_constraint(x_0[5] >= lb_theta_d)
    lyap_cplex_model.add_constraint(x_0[5] <= ub_theta_d)

    lyap_cplex_model.add_constraint(lyap_cplex_model.max(lyap_cplex_model.abs(x_0[i]) for i in range(6)) >= 0.1)

    # get the final policy (force) is the action
    force_mid = {(i): lyap_cplex_model.continuous_var(name='force_mid_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(2)} #lb = 0 by default
    force = {(i): lyap_cplex_model.continuous_var(name='force_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(2)} #lb = 0 by default

    for i in range(2):
        lyap_cplex_model.add_constraint(force_mid[i] == lyap_cplex_model.sum(x_0[j] * policy_w1[i][j] for j in range(6)))
        lyap_cplex_model.add_constraint(force[i] == lyap_cplex_model.min(lyap_cplex_model.max(force_mid[i] + mass * gravity / 2.0, max_force_lb_const), max_force_ub_const))

    x_0_next = {(i): lyap_cplex_model.continuous_var(name = 'x_0_next_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(6)} #lb = 0 by default
    x_change = {(i): lyap_cplex_model.continuous_var(name = 'x_change_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(6)} #lb = 0 by default

    vx_cos_lb_slope_vx, vx_cos_lb_slope_theta, vx_cos_lb_bias, vx_cos_ub_slope_vx, vx_cos_ub_slope_theta, vx_cos_ub_bias = bound_func_cos(lb_x_d, ub_x_d, lb_theta, ub_theta)
    vy_sin_lb_slope_vy, vy_sin_lb_slope_theta, vy_sin_lb_bias, vy_sin_ub_slope_vy, vy_sin_ub_slope_theta, vy_sin_ub_bias = bound_func_sin(lb_y_d, ub_y_d, lb_theta, ub_theta)

    vx_sin_lb_slope_vx, vx_sin_lb_slope_theta, vx_sin_lb_bias, vx_sin_ub_slope_vx, vx_sin_ub_slope_theta, vx_sin_ub_bias = bound_func_sin(lb_x_d, ub_x_d, lb_theta, ub_theta)
    vy_cos_lb_slope_vy, vy_cos_lb_slope_theta, vy_cos_lb_bias, vy_cos_ub_slope_vy, vy_cos_ub_slope_theta, vy_cos_ub_bias = bound_func_cos(lb_y_d, ub_y_d, lb_theta, ub_theta)

    vx_thd_lb_slope_vx, vx_thd_lb_slope_thetad, vx_thd_ub_slope_vx, vx_thd_ub_slope_thetad = bound_func_xy(lb_x_d, ub_x_d, lb_theta_d, ub_theta_d)
    vy_thd_lb_slope_vy, vy_thd_lb_slope_thetad, vy_thd_ub_slope_vy, vy_thd_ub_slope_thetad = bound_func_xy(lb_y_d, ub_y_d, lb_theta_d, ub_theta_d)

    lower_slope_sin, lower_bias_sin, upper_slope_sin, upper_bias_sin = BoundSin_fun.bound_relax_impl(th.FloatTensor([lb_theta]),th.FloatTensor([ub_theta]))
    lower_slope_cos, lower_bias_cos, upper_slope_cos, upper_bias_cos = BoundCos_fun(th.FloatTensor([lb_theta]),th.FloatTensor([ub_theta]))

    lower_slope_sin = lower_slope_sin.item()
    lower_bias_sin = lower_bias_sin.item()
    upper_slope_sin = upper_slope_sin.item()
    upper_bias_sin = upper_bias_sin.item()

    lower_slope_cos = lower_slope_cos.item()
    lower_bias_cos = lower_bias_cos.item()
    upper_slope_cos = upper_slope_cos.item()
    upper_bias_cos = upper_bias_cos.item()

    #x_d
    lyap_cplex_model.add_constraint(x_change[0] <= (x_0[3] * vx_cos_ub_slope_vx + x_0[2] * vx_cos_ub_slope_theta + vx_cos_ub_bias) - \
        (x_0[4] * vy_sin_lb_slope_vy + x_0[2] * vy_sin_lb_slope_theta + vy_sin_lb_bias))
    lyap_cplex_model.add_constraint(x_change[0] >= (x_0[3] * vx_cos_lb_slope_vx + x_0[2] * vx_cos_lb_slope_theta + vx_cos_lb_bias) - \
        (x_0[4] * vy_sin_ub_slope_vy + x_0[2] * vy_sin_ub_slope_theta + vy_sin_ub_bias))

    #y_d
    lyap_cplex_model.add_constraint(x_change[1] <= (x_0[3] * vx_sin_ub_slope_vx + x_0[2] * vx_sin_ub_slope_theta + vx_sin_ub_bias) + \
        (x_0[4] * vy_cos_ub_slope_vy + x_0[2] * vy_cos_ub_slope_theta + vy_cos_ub_bias))
    lyap_cplex_model.add_constraint(x_change[1] >= (x_0[3] * vx_sin_lb_slope_vx + x_0[2] * vx_sin_lb_slope_theta + vx_sin_lb_bias) + \
        (x_0[4] * vy_cos_lb_slope_vy + x_0[2] * vy_cos_lb_slope_theta + vy_cos_lb_bias))    

    #theta_d
    lyap_cplex_model.add_constraint(x_change[2] == x_0[5])

    #x_d_d
    lyap_cplex_model.add_constraint(x_change[3] <= vy_thd_ub_slope_vy * x_0[4] + vy_thd_ub_slope_thetad * x_0[5] - gravity * (lower_slope_sin * x_0[2] + lower_bias_sin))
    lyap_cplex_model.add_constraint(x_change[3] >= vy_thd_lb_slope_vy * x_0[4] + vy_thd_lb_slope_thetad * x_0[5] - gravity * (upper_slope_sin * x_0[2] + upper_bias_sin))

    #y_d_d
    lyap_cplex_model.add_constraint(x_change[4] <= -(vx_thd_lb_slope_vx * x_0[3] + vx_thd_lb_slope_thetad * x_0[5]) - gravity * (lower_slope_cos * x_0[2] + lower_bias_cos) + (force[0] + force[1])/mass)
    lyap_cplex_model.add_constraint(x_change[4] >= -(vx_thd_ub_slope_vx * x_0[3] + vx_thd_ub_slope_thetad * x_0[5]) - gravity * (upper_slope_cos * x_0[2] + upper_bias_cos) + (force[0] + force[1])/mass)

    #theta_d_d
    lyap_cplex_model.add_constraint(x_change[5] == (force[0] - force[1]) * dist/inertia)

    lyap_cplex_model.add_constraint(x_0_next[0] == x_0[0] + x_change[0] * dt)
    lyap_cplex_model.add_constraint(x_0_next[1] == x_0[1] + x_change[1] * dt)
    lyap_cplex_model.add_constraint(x_0_next[2] == x_0[2] + x_change[2] * dt)
    lyap_cplex_model.add_constraint(x_0_next[3] == x_0[3] + x_change[3] * dt)
    lyap_cplex_model.add_constraint(x_0_next[4] == x_0[4] + x_change[4] * dt)
    lyap_cplex_model.add_constraint(x_0_next[5] == x_0[5] + x_change[5] * dt)

    # get the lyapunov_function_value for (x, x_dot, theta, theta_dot)
    lyx_1 = {(i): lyap_cplex_model.continuous_var(name='lyx_1_{}'.format(i), lb = 0) for i in range(MID_LAYER_SIZE)}
    lys_1 = {(i): lyap_cplex_model.continuous_var(name='lys_1_{}'.format(i), lb = 0) for i in range(MID_LAYER_SIZE)}
    lyzx_1 = {(i): lyap_cplex_model.binary_var(name='lyzx_1_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyx_2 = {(i): lyap_cplex_model.continuous_var(name='lyx_2_{}'.format(i), lb = 0) for i in range(MID_LAYER_SIZE)}
    lys_2 = {(i): lyap_cplex_model.continuous_var(name='lys_2_{}'.format(i), lb = 0) for i in range(MID_LAYER_SIZE)}
    lyzx_2 = {(i): lyap_cplex_model.binary_var(name='lyzx_2_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyp_val_x0 = lyap_cplex_model.continuous_var(name='lyp_val_x0', lb = -lyap_cplex_model.infinity)

    for i in range(MID_LAYER_SIZE):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0[j] for j in range(6)) == lyx_1[i] - lys_1[i])
        lyap_cplex_model.add_constraint(lyx_1[i] <= M * lyzx_1[i])
        lyap_cplex_model.add_constraint(lys_1[i] <= M * (1 - lyzx_1[i]))
    
    for i in range(MID_LAYER_SIZE):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1[j] for j in range(MID_LAYER_SIZE)) == lyx_2[i] - lys_2[i])
        lyap_cplex_model.add_constraint(lyx_2[i] <= M * lyzx_2[i])
        lyap_cplex_model.add_constraint(lys_2[i] <= M * (1 - lyzx_2[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0 == lyap_cplex_model.sum(lyx_2[j] * lyapunov_w3[0][j] for j in range(MID_LAYER_SIZE)))
    
    # get the lyapunov_function_value for (x_next, x_dot_next, theta_next, theta_dot_next)
    lyx_1_next = {(i): lyap_cplex_model.continuous_var(name='lyx_1_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lys_1_next = {(i): lyap_cplex_model.continuous_var(name='lys_1_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyzx_1_next = {(i): lyap_cplex_model.binary_var(name='lyzx_1_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyx_2_next = {(i): lyap_cplex_model.continuous_var(name='lyx_2_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lys_2_next = {(i): lyap_cplex_model.continuous_var(name='lys_2_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyzx_2_next = {(i): lyap_cplex_model.binary_var(name='lyzx_2_next_{}'.format(i)) for i in range(MID_LAYER_SIZE)}
    lyp_val_x0_next = lyap_cplex_model.continuous_var(name='lyp_val_x0_next', lb = -lyap_cplex_model.infinity)

    for i in range(MID_LAYER_SIZE):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0_next[j] for j in range(6)) == lyx_1_next[i] - lys_1_next[i])
        lyap_cplex_model.add_constraint(lyx_1_next[i] <= M * lyzx_1_next[i])
        lyap_cplex_model.add_constraint(lys_1_next[i] <= M * (1 - lyzx_1_next[i]))
    
    for i in range(MID_LAYER_SIZE):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1_next[j] for j in range(MID_LAYER_SIZE)) == lyx_2_next[i] - lys_2_next[i])
        lyap_cplex_model.add_constraint(lyx_2_next[i] <= M * lyzx_2_next[i])
        lyap_cplex_model.add_constraint(lys_2_next[i] <= M * (1 - lyzx_2_next[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0_next == lyap_cplex_model.sum(lyx_2_next[j] * lyapunov_w3[0][j] for j in range(MID_LAYER_SIZE)))
    
    lyap_cplex_model.minimize(lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()
    sol_print = round(solution_.objective_value, 8)
    counter_example1 = solution_.get_values([x_0[0], x_0[1], x_0[2], x_0[3], x_0[4], x_0[5]])
    if sol_print < 0:
        satisfy_cond_flag = False
        #print("sign not satisfied")
        x = th.FloatTensor([counter_example1])
        #print("lyap value:", lyap_model(x))

    lyap_cplex_model.add_constraint(lyp_val_x0_next - lyp_val_x0 >= -0.00001)
    lyap_cplex_model.maximize(lyp_val_x0_next - lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()

    lyap_value_diff = 0
    counter_example2 = None
    if solution_ is not None:
        counter_example2 = []
        satisfy_cond_flag = False 
        #print("energy decrease not satisfied")
        sol_print = round(solution_.objective_value, 8)
        counter_example2 = solution_.get_values([x_0[0], x_0[1], x_0[2], x_0[3], x_0[4], x_0[5]])
        #print("counter_example2", counter_example2)
        counter_example2 = th.FloatTensor([counter_example2])
        lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, counter_example2)
        #print("lyap_value_diff:", lyap_value_diff.mean())
        #print("sol_print", sol_print)

    #print(satisfy_cond_flag)
    return satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff


def certify_split(policy_model, lyap_model, counter_example2, element):
    print("splitting the region")
    LB_theta_d = element[10]
    UB_theta_d = element[11]
    
    theta_d_split = (LB_theta_d + UB_theta_d)/2.0
    
    element_split_lower = element.copy()
    element_split_lower[11] = theta_d_split
    
    element_split_upper = element.copy()
    element_split_upper[10] = theta_d_split    
    
    satisfy_cond_flag_r1, _, counter_example2_r1, _ = certify(policy_model, lyap_model, element_split_lower)
    if not satisfy_cond_flag_r1:
        return True
    satisfy_cond_flag_r2, _, counter_example2_r2, _ = certify(policy_model, lyap_model, element_split_upper)
    if not satisfy_cond_flag_r2:
        return True
    
    return False

def main_certify_all(policy_model, lyap_model, certify_list, certify_counter_example):
    success_flag_main = True
    for element in certify_list:
        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, element)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if (lyap_value_diff.item() < 0):
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, element)

            if store_counterexample:  
                certify_counter_example.append(counter_example2)
                success_flag_main = False

    return success_flag_main, certify_counter_example


def lyap_train_main(policy_model, lyap_model, certify_counter_example, certify_list, learning_rate, args, const_c = CONST_C):
    print("lyap_train_main")
    N = 500             # sample size
    iter_num = 0 
    max_iters = 1000
    optimizer = th.optim.Adam(list(policy_model.parameters()) + list(lyap_model.parameters()), lr=learning_rate)
    valid_num = 0

    bound = th.FloatTensor([[UB_CONST_GLOBAL_x, UB_CONST_GLOBAL_y, UB_CONST_GLOBAL_theta, UB_CONST_GLOBAL_x_d, UB_CONST_GLOBAL_y_d, UB_CONST_GLOBAL_theta_d]]).to(args.device)

    x = th.zeros(N, STATE_SPACE_SIZE).uniform_(-1, 1).to(args.device) * bound.squeeze(0)
    if len(certify_counter_example) != 0:
        certify_counter_example_th = th.cat(certify_counter_example, dim = 0)
        x = th.cat((x, certify_counter_example_th), dim = 0)

    x_all = x.clone()
    x_0 = th.zeros([1, STATE_SPACE_SIZE]).to(args.device)
    
    while (iter_num < max_iters):

        model_bound = BoundedModule(lyap_model, x_0)
        ptb_val = PerturbationLpNorm(norm=np.inf, x_L = -bound, x_U = bound)
        state_range = BoundedTensor(x_0, ptb_val)
        lyap_value, _ = model_bound.compute_bounds(x=(state_range,), method="backward")

        lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, x)
        Lyapunov_risk = F.relu(-lyap_value).mean() + 1.2*F.relu(lyap_value_diff + const_c).mean()

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        for param in policy_model.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step() 

        if iter_num % 10 == 0:
            result = FindCounterExamples(policy_model, lyap_model, args.device)
            if (len(result) != 0): 
                #print("Not a Lyapunov function. Found counterexample: ")
                #print(result)
                print(result.size())
                
            else:
                valid_num += 1
                #print("Satisfy conditions!! Iter:", valid_num)
            #print('==============================') 

            indices = np.random.choice(x_all.size()[0], size=min(x_all.size()[0], 512))
            x = th.cat((result, x_all[indices, :]), dim = 0)
            if len(certify_counter_example) != 0:
                x = th.cat((x, certify_counter_example_th), dim = 0)
            x_all = th.cat((result, x_all), dim = 0)
            
        iter_num += 1

        if (iter_num % 10 == 0) and (Lyapunov_risk.item() == 0):
            break

    print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 

    success_flag_main = True

    counter_example_interval_list = []
    counter_example_diff_list = []
    for element in certify_list:
        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, element)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if (lyap_value_diff.item() < 0):
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, element)

            if store_counterexample: 
                success_flag_main = False
                counter_example_interval_list.append(element)
                certify_counter_example.append(th.FloatTensor(counter_example2))
                counter_example_diff_list.append(lyap_value_diff.item())

    return certify_counter_example, counter_example_interval_list, success_flag_main, counter_example_diff_list

def train_main(policy_model, lyap_model, certify_list, certify_counter_example, args, const_c = CONST_C):
    numer_of_trains_count = 0
    global CONST_C
    success_flag_main = False
    while not success_flag_main:
        numer_of_trains_count = numer_of_trains_count + 1
        learning_rate_main = 0.00625
        print("CONST_C: ", CONST_C)
        certify_counter_example, counter_example_interval_list, success_flag_main, counter_example_diff_list = lyap_train_main(policy_model, lyap_model, certify_counter_example, certify_list, learning_rate_main, args, const_c = CONST_C)
        if len(counter_example_diff_list) > 0:
            CONST_C = max(0.12, -np.min(counter_example_diff_list) + 0.01)
        else:
            CONST_C = 0.12
        if success_flag_main:
            return numer_of_trains_count

        for _ in range(10):
            print("CONST_C: ", CONST_C)
            learning_rate_main = learning_rate_main * 0.9
            certify_counter_example, _, success_flag_main, counter_example_diff_list = lyap_train_main(policy_model, lyap_model, certify_counter_example, counter_example_interval_list, learning_rate_main, args, const_c = CONST_C)
            if len(counter_example_diff_list) > 0:
                CONST_C = max(0.12, -np.min(counter_example_diff_list) + 0.01)
            else:
                CONST_C = 0.12
            if success_flag_main:
                break
        
        if success_flag_main:
            success_flag_main, certify_counter_example = main_certify_all(policy_model, lyap_model, certify_list, certify_counter_example)
    
    return numer_of_trains_count


def pre_train(lyap_model, policy_model, args):
    print("pre_train")
    N = 500             # sample size
    iter_num = 0 
    max_iters = 1000
    learning_rate = 0.00625
    optimizer = th.optim.Adam(list(lyap_model.parameters()), lr=learning_rate)
    valid_num = 0

    bound = th.FloatTensor([[UB_CONST_GLOBAL_x, UB_CONST_GLOBAL_y, UB_CONST_GLOBAL_theta, UB_CONST_GLOBAL_x_d, UB_CONST_GLOBAL_y_d, UB_CONST_GLOBAL_theta_d]]).to(args.device)

    x = th.zeros(N, STATE_SPACE_SIZE).uniform_(-1, 1).to(args.device) * bound.squeeze(0)
    x_all = x.clone()
    x_0 = th.zeros([1, STATE_SPACE_SIZE]).to(args.device)

    while (iter_num < max_iters):

        model_bound = BoundedModule(lyap_model, x_0)
        ptb_val = PerturbationLpNorm(norm=np.inf, x_L = -bound, x_U = bound)
        state_range = BoundedTensor(x_0, ptb_val)
        lyap_value, _ = model_bound.compute_bounds(x=(state_range,), method="backward")

        lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, x)

        Lyapunov_risk = (F.relu(-lyap_value) + 1.2*F.relu(lyap_value_diff + CONST_C)).mean()#+ 1.2*(V0).pow(2)

        #print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step()

        if iter_num % 10 == 0:
            result = FindCounterExamples(policy_model, lyap_model, args.device)
            if (len(result) != 0): 
                print(result.size())
                
            else:
                valid_num += 1
                #print("Satisfy conditions!! Iter:", valid_num)
            #print('==============================') 

            indices = np.random.choice(x_all.size()[0], size=min(x_all.size()[0], 512))
            x = th.cat((result, x_all[indices, :]), dim = 0)
            x_all = th.cat((result, x_all), dim = 0)
            
        iter_num += 1
    print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 
    return

def train_lyapunov(args):  # noqa: C901

    seed_num = args.seed
    random.seed(seed_num)
    th.manual_seed(seed_num)  
    np.random.seed(seed_num)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    # set_random_seed(args.seed)
    print("seed:", args.seed)
    policy_model = PolicyNet().to(args.device)
    lyap_model = LyapunovNet().to(args.device)


    training_start_time = time.time()
    pre_train(lyap_model, policy_model, args)

    certify_counter_example = []

    certify_list = []
    split_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
    split_list_theta = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for ii in range(len(split_list_theta) - 1):
        for jj in range(len(split_list) - 1):
            for kk in range(len(split_list) - 1):
                for ll in range(len(split_list) - 1):
                    certify_list.append([-train_diameter, train_diameter, -train_diameter, train_diameter, split_list_theta[ii], split_list_theta[ii+1], split_list[jj], split_list[jj + 1], split_list[kk], split_list[kk + 1], split_list[ll], split_list[ll + 1]])

    numer_of_trains_count = train_main(policy_model, lyap_model, certify_list, certify_counter_example, args, const_c = CONST_C)
    
    training_total_time = time.time() - training_start_time
    f = open("pvtol_result.txt", "a")
    f.write("learning rate:" + str(args.lr_linear)+ "\n")
    f.write("seed:"+ str(args.seed) + ":training_time:" + str(training_total_time)+ ":numer_of_trains_count:" + str(numer_of_trains_count) + ":mac_force_ub:" + str(env_params["max_force_ub"]) + "\n\n")
    f.close()

    state_to_save_lyap = {'model_state_dict':lyap_model.state_dict()}
    state_to_save_policy = {'model_state_dict':policy_model.state_dict()}
    th.save(state_to_save_lyap, "pvtol_state_to_save_lyap_seed" + str(int(args.seed)) + str(args.lr_linear) + "_v1.pt")
    th.save(state_to_save_policy, "pvtol_state_to_save_policy_seed" + str(int(args.seed)) + str(args.lr_linear)+ "_v1.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-linear", help="learning rate for linear", type=float, default=None) 
    parser.add_argument("--seed", help="Random generator seed", type=int, default=None) 
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default=None, type=str)

    args = parser.parse_args()

    #pass
    train_lyapunov(args)
