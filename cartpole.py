import argparse

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
import warnings
warnings.filterwarnings("ignore")

env_params = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "total_mass": 1.1,
    "length": 1.0,  
    "tau": 0.05,  # seconds between state updates
    "max_force": 30.0,
}

BOUND_CART = 1.0
BOUND_CART_SPEED = 1.0
BOUND_POLE = 1.0
BOUND_POLE_SPEED = 1.0

certify_diameter = 1.0
SAMPLE_SIZE = 512
STATE_SPACE_SIZE = 4

certify_list_points = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet,self).__init__()
        self.policy = nn.Linear(STATE_SPACE_SIZE, 1, bias = False)
        self.policy.weight = th.nn.Parameter(th.FloatTensor([[ 1.0, 2.4109461, 34.36203947, 10.70094483]]))

    def forward(self, x):
        return self.policy(x)

class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet,self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(STATE_SPACE_SIZE, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )

    def forward(self,x):
        return self.lyapunov(x)

def dynamics(state, action):
    gravity = env_params['gravity']
    masscart = env_params['masscart']
    masspole = env_params['masspole']
    total_mass = env_params['total_mass']
    length = env_params['length']
    tau = env_params['tau']
    max_force = env_params['max_force']

    force = action
    force = th.clamp(force, min = -max_force, max = max_force)

    x = state[:, 0:1]
    x_dot = state[:, 1:2]
    theta = state[:, 2:3]
    theta_dot = state[:, 3:4]

    costheta = th.cos(theta)
    sintheta = th.sin(theta)

    temp = masscart + masspole * sintheta**2
    thetaacc = (-force * costheta - masspole * length * theta_dot**2 * costheta * sintheta \
        + total_mass * gravity * sintheta) / (length * temp)
    xacc = (force + masspole * sintheta * (length * theta_dot**2 - gravity * costheta)) / temp

    x_next = x + tau * x_dot
    x_dot_next = x_dot + tau * xacc
    theta_next = theta + tau * theta_dot
    theta_dot_next = theta_dot + tau * thetaacc

    return th.cat((x_next, x_dot_next, theta_next, theta_dot_next), dim = 1)

def lyap_diff(policy_model, lyap_model, state):
    lyap_value = lyap_model(state)
    action = policy_model(state)
    state_next = th.zeros_like(state)

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
    lyap_value = lyap_model(state)
    target_func = th.sum(-lyap_value)
    lyap_model.zero_grad()
    target_func.backward()
    grad = state.grad
    return th.sign(grad)

def FindCounterExamples(policy_model, lyap_model, device):
    delta = th.zeros(SAMPLE_SIZE, STATE_SPACE_SIZE).uniform_(-1, 1)
    min_state = delta * th.FloatTensor([BOUND_CART, BOUND_CART_SPEED, BOUND_POLE, BOUND_POLE_SPEED])
    min_state = min_state.to(device)
    steps = 30
    relative_step_size = 1/steps
    for _ in range(steps):
        min_state.requires_grad = True
        min_state = min_state + relative_step_size * gradient_lyap_diff(policy_model, lyap_model, min_state) 
        min_state = min_state.detach()
        min_state[:, 0:1] = th.clamp(min_state[:, 0:1], min = -BOUND_CART, max = BOUND_CART)
        min_state[:, 1:2] = th.clamp(min_state[:, 1:2], min = -BOUND_CART_SPEED, max = BOUND_CART_SPEED)
        min_state[:, 2:3] = th.clamp(min_state[:, 2:3], min = -BOUND_POLE, max = BOUND_POLE)
        min_state[:, 3:4] = th.clamp(min_state[:, 3:4], min = -BOUND_POLE_SPEED, max = BOUND_POLE_SPEED)

    lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, min_state)
    min_state_return1 = min_state[lyap_value_diff.flatten() >= -0.0001].clone().detach()

    return min_state_return1

def func1(theta):
    return 1/(env_params['masscart'] + env_params['masspole'] * np.sin(theta)**2)

def func2(theta):
    return -np.cos(theta) * func1(theta)/env_params['length']

# for func1 and func2, LB and UB does not matter
def func1_bound(LB_theta, UB_theta):
    return func1(LB_theta), func1(UB_theta)

def func2_bound(LB_theta, UB_theta):
    return func2(LB_theta), func2(UB_theta)

def get_upper_bound(func, theta_sp, LB, UB, is_concave):
    if type(func) == int or type(func) == float:
        return 0, func

    if is_concave:
        theta_opt_sp = sp.symbols('theta_opt_sp')
        func_diff_sp = func.diff(theta_sp)
        #assume concave, find upper bound
        bound_func_diff = func.subs(theta_sp, theta_opt_sp) + func_diff_sp.subs(theta_sp, theta_opt_sp) * (theta_sp - theta_opt_sp) - func
        bound_func_diff2 = sp.lambdify((theta_sp, theta_opt_sp), bound_func_diff, modules = ["scipy", "numpy"])
        bound_func_int = lambda theta_opt_sp: nquad(bound_func_diff2, [[LB, UB]], args=(theta_opt_sp,))[0]

        result = optimize.minimize_scalar(bound_func_int, bounds=(LB, UB), method='bounded')
        if result.success:
            theta_opt = result.x
        ub_slope = func_diff_sp.subs(theta_sp, theta_opt)
        ub_bias = func.subs(theta_sp, theta_opt) - theta_opt * ub_slope
    else:
        ub_slope = (func.subs(theta_sp, LB) - func.subs(theta_sp, UB))/(LB - UB)
        ub_bias = func.subs(theta_sp, LB) - ub_slope * LB
    
    ub_slope = ub_slope.evalf()
    ub_bias = ub_bias.evalf()
    return float(ub_slope), float(ub_bias)


def get_lower_bound(func, theta_sp, LB, UB, is_concave):
    if type(func) == int or type(func) == float:
        return 0, func
    if is_concave:
        lb_slope = (func.subs(theta_sp, LB) - func.subs(theta_sp, UB))/(LB - UB)
        lb_bias = func.subs(theta_sp, LB) - lb_slope * LB
    else:
        theta_opt_sp = sp.symbols('theta_opt_sp')
        func_diff_sp = func.diff(theta_sp)
        #assume concave, find upper bound
        bound_func_diff = func.subs(theta_sp, theta_opt_sp) + func_diff_sp.subs(theta_sp, theta_opt_sp) * (theta_sp - theta_opt_sp) - func
        bound_func_diff = -bound_func_diff
        bound_func_diff2 = sp.lambdify((theta_sp, theta_opt_sp), bound_func_diff, modules = ["scipy", "numpy"])
        bound_func_int = lambda theta_opt_sp: nquad(bound_func_diff2, [[LB, UB]], args=(theta_opt_sp,))[0]
        result = optimize.minimize_scalar(bound_func_int, bounds=(LB, UB), method='bounded')
        if result.success:
            theta_opt = result.x
        lb_slope = func_diff_sp.subs(theta_sp, theta_opt)
        lb_bias = func.subs(theta_sp, theta_opt) - theta_opt * lb_slope
    lb_slope = lb_slope.evalf()
    lb_bias = lb_bias.evalf()
    return float(lb_slope), float(lb_bias)

# bound function for one variable
def bound_func(func, theta_sp, LB, UB, is_concave):
    lb_slope, lb_bias = get_lower_bound(func, theta_sp, LB, UB, is_concave)
    ub_slope, ub_bias = get_upper_bound(func, theta_sp, LB, UB, is_concave)

    return lb_slope, lb_bias, ub_slope, ub_bias


def func3_bound(LB, UB): #func3 is decreasing in region [-0.05, 0.05]
    if LB >= UB:
        raise Exception("bound is messed up: LB >= UB.")

    theta_sp = sp.symbols('theta_sp')   
    func3_sp = -env_params['gravity'] * env_params['masspole'] * sp.cos(theta_sp) * sp.sin(theta_sp) / (env_params['masscart'] + env_params['masspole'] * sp.sin(theta_sp)**2)
    
    if LB < 0:
        is_concave = True
    elif UB > 0:
        is_concave = False
    else:
        raise Exception("bound is messed up")

    return bound_func(func3_sp, theta_sp, LB, UB, is_concave)

def func4_bound(LB, UB): 
    if LB >= UB:
        raise Exception("bound is messed up: LB >= UB.")

    theta_sp = sp.symbols('theta_sp')   
    func4_sp =  env_params['total_mass'] * env_params['gravity'] * sp.sin(theta_sp) / ((env_params['masscart'] + env_params['masspole'] * sp.sin(theta_sp)**2) * env_params['length'])
    
    #is_concave = check_convexity(func4_sp, theta_sp, LB, UB)
    if LB < 0:
        is_concave = False
    elif UB > 0:
        is_concave = True
    else:
        raise Exception("bound is messed up")

    return bound_func(func4_sp, theta_sp, LB, UB, is_concave)

def func5_bound(LB_theta, UB_theta, LB_theta_d, UB_theta_d):
    if (LB_theta >= UB_theta) or (LB_theta_d >= UB_theta_d):
        raise Exception("bound is messed up: LB >= UB.")

    theta_sp = sp.symbols('theta_sp')   
    theta_d_sp = sp.symbols('theta_d_sp')   
    func5_sp = env_params['masspole'] * sp.sin(theta_sp) * env_params['length'] * theta_d_sp**2 /(env_params['masscart'] + env_params['masspole'] * sp.sin(theta_sp)**2)

    # th_min: theta value that will result in the minimum f(theta)
    if (LB_theta + UB_theta) * (LB_theta_d + UB_theta_d) > 0:
        th_min = LB_theta
        th_max = UB_theta
        th_d_min = LB_theta_d
        th_d_max = UB_theta_d
    else: # if theta * theta_d < 0
        th_min = LB_theta
        th_max = UB_theta
        th_d_min = UB_theta_d
        th_d_max = LB_theta_d

    if (LB_theta + UB_theta) > 0:
        th_is_concave = True
        th_d_is_concave = False
    elif (LB_theta + UB_theta) < 0:
        th_is_concave = False
        th_d_is_concave = True
    else:
        raise Exception("bound is messed up")

    # get the lower bound for the function
    coef = sp.symbols('coef')
    lb_func_th = func5_sp.subs(theta_d_sp, th_d_min)
    lb_func_th_d = func5_sp.subs(theta_sp, th_min)
    
    lb_func_th_slope, lb_func_th_bias = get_lower_bound(lb_func_th, theta_sp, LB_theta, UB_theta, th_is_concave)
    lb_func_th_d_slope, lb_func_th_d_bias = get_lower_bound(lb_func_th_d, theta_d_sp, LB_theta_d, UB_theta_d, th_d_is_concave)

    func5_sp_lb = coef * (lb_func_th_slope * theta_sp + lb_func_th_bias) + (1 - coef) * (lb_func_th_d_slope * theta_d_sp + lb_func_th_d_bias) #linearized function, now optimize w.r.t. coeff
    bound_func_diff = func5_sp - func5_sp_lb
    bound_func_diff2 = sp.lambdify((theta_sp, theta_d_sp, coef), bound_func_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(bound_func_diff2, [[LB_theta, UB_theta], [LB_theta_d, UB_theta_d]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x
    lb_slope_theta = coef_opt * lb_func_th_slope
    lb_slope_theta_d = (1 - coef_opt) * lb_func_th_d_slope
    lb_bias = coef_opt * lb_func_th_bias + (1 - coef_opt) * lb_func_th_d_bias

    # get the upper bound for the function
    ub_func_th = func5_sp.subs(theta_d_sp, th_d_max)
    ub_func_th_d = func5_sp.subs(theta_sp, th_max)
    
    ub_func_th_slope, ub_func_th_bias = get_upper_bound(ub_func_th, theta_sp, LB_theta, UB_theta, th_is_concave)
    ub_func_th_d_slope, ub_func_th_d_bias = get_upper_bound(ub_func_th_d, theta_d_sp, LB_theta_d, UB_theta_d, th_d_is_concave)

    func5_sp_ub = coef * (ub_func_th_slope * theta_sp + ub_func_th_bias) + (1 - coef) * (ub_func_th_d_slope * theta_d_sp + ub_func_th_d_bias) #linearized function, now optimize w.r.t. coeff
    bound_func_diff = func5_sp_ub - func5_sp
    bound_func_diff2 = sp.lambdify((theta_sp, theta_d_sp, coef), bound_func_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(bound_func_diff2, [[LB_theta, UB_theta], [LB_theta_d, UB_theta_d]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x
    ub_slope_theta = coef_opt * ub_func_th_slope
    ub_slope_theta_d = (1 - coef_opt) * ub_func_th_d_slope
    ub_bias = coef_opt * ub_func_th_bias + (1 - coef_opt) * ub_func_th_d_bias

    return float(lb_slope_theta), float(lb_slope_theta_d), float(lb_bias), float(ub_slope_theta), float(ub_slope_theta_d), float(ub_bias)


def func6_bound(LB_theta, UB_theta, LB_theta_d, UB_theta_d):
    if (LB_theta >= UB_theta) or (LB_theta_d >= UB_theta_d):
        raise Exception("bound is messed up: LB >= UB.")

    theta_sp = sp.symbols('theta_sp')   
    theta_d_sp = sp.symbols('theta_d_sp')   
    func6_sp = -env_params['masspole'] * sp.sin(theta_sp) * sp.cos(theta_sp)* theta_d_sp**2 /(env_params['masscart'] + env_params['masspole'] * sp.sin(theta_sp)**2)

    # th_min: theta value that will result in the minimum f(theta)
    if (LB_theta + UB_theta) * (LB_theta_d + UB_theta_d) > 0:
        th_min = UB_theta
        th_max = LB_theta
        th_d_min = UB_theta_d
        th_d_max = LB_theta_d 
    else: # if theta * theta_d < 0
        th_min = UB_theta
        th_max = LB_theta
        th_d_min = LB_theta_d
        th_d_max = UB_theta_d
 
    if (LB_theta + UB_theta) > 0:
        th_is_concave = False
        th_d_is_concave = True
    elif (LB_theta + UB_theta) < 0:
        th_is_concave = True
        th_d_is_concave = False

    else:
        raise Exception("bound is messed up")

    # get the lower bound for the function
    coef = sp.symbols('coef')
    lb_func_th = func6_sp.subs(theta_d_sp, th_d_min)
    lb_func_th_d = func6_sp.subs(theta_sp, th_min)
    
    lb_func_th_slope, lb_func_th_bias = get_lower_bound(lb_func_th, theta_sp, LB_theta, UB_theta, th_is_concave)
    lb_func_th_d_slope, lb_func_th_d_bias = get_lower_bound(lb_func_th_d, theta_d_sp, LB_theta_d, UB_theta_d, th_d_is_concave)

    func6_sp_lb = coef * (lb_func_th_slope * theta_sp + lb_func_th_bias) + (1 - coef) * (lb_func_th_d_slope * theta_d_sp + lb_func_th_d_bias) #linearized function, now optimize w.r.t. coeff
    bound_func_diff = func6_sp - func6_sp_lb
    bound_func_diff2 = sp.lambdify((theta_sp, theta_d_sp, coef), bound_func_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(bound_func_diff2, [[LB_theta, UB_theta], [LB_theta_d, UB_theta_d]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x
    lb_slope_theta = coef_opt * lb_func_th_slope
    lb_slope_theta_d = (1 - coef_opt) * lb_func_th_d_slope
    lb_bias = coef_opt * lb_func_th_bias + (1 - coef_opt) * lb_func_th_d_bias

    # get the upper bound for the function
    ub_func_th = func6_sp.subs(theta_d_sp, th_d_max)
    ub_func_th_d = func6_sp.subs(theta_sp, th_max)
    
    ub_func_th_slope, ub_func_th_bias = get_upper_bound(ub_func_th, theta_sp, LB_theta, UB_theta, th_is_concave)
    ub_func_th_d_slope, ub_func_th_d_bias = get_upper_bound(ub_func_th_d, theta_d_sp, LB_theta_d, UB_theta_d, th_d_is_concave)

    func6_sp_ub = coef * (ub_func_th_slope * theta_sp + ub_func_th_bias) + (1 - coef) * (ub_func_th_d_slope * theta_d_sp + ub_func_th_d_bias) #linearized function, now optimize w.r.t. coeff
    bound_func_diff = func6_sp_ub - func6_sp
    bound_func_diff2 = sp.lambdify((theta_sp, theta_d_sp, coef), bound_func_diff, modules = ["scipy", "numpy"])
    bound_func_int = lambda coef: nquad(bound_func_diff2, [[LB_theta, UB_theta], [LB_theta_d, UB_theta_d]],args=(coef,))[0]
    result = optimize.minimize_scalar(bound_func_int, bounds=(0, 1), method='bounded')
    if result.success:
        coef_opt = result.x
    ub_slope_theta = coef_opt * ub_func_th_slope
    ub_slope_theta_d = (1 - coef_opt) * ub_func_th_d_slope
    ub_bias = coef_opt * ub_func_th_bias + (1 - coef_opt) * ub_func_th_d_bias

    return float(lb_slope_theta), float(lb_slope_theta_d), float(lb_bias), float(ub_slope_theta), float(ub_slope_theta_d), float(ub_bias)

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

def certify(policy_model, lyap_model, LB_theta, UB_theta, LB_theta_d, UB_theta_d, LB_x, UB_x, LB_x_dot, UB_x_dot):

    print("LB_theta:", round(LB_theta, 5), "UB_theta:", round(UB_theta, 5), "LB_theta_d:", round(LB_theta_d, 5), "UB_theta_d:", round(UB_theta_d, 5))


    satisfy_cond_flag = True
    # load weights to numpy for CPLEX
    policy_w1 = policy_model.state_dict()['policy.weight'].data.cpu().numpy()

    lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
    lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
    lyapunov_w3 = lyap_model.state_dict()['lyapunov.4.weight'].data.cpu().numpy()

    if (np.sign(LB_theta * UB_theta) == -1) or (np.sign(LB_theta_d * UB_theta_d) == -1):
        raise Exception("Not monotone in the selected region. (e.g., the region cross zero)")

    lyap_cplex_model = Model(name='Lyapunov Verification')
    
    x_0 = {(i): lyap_cplex_model.continuous_var(name='x_0_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(4)} #lb = 0 by default
    #bound the input variables (x, x_dot, theta, theta_dot)
    lyap_cplex_model.add_constraint(x_0[0] >= LB_x)
    lyap_cplex_model.add_constraint(x_0[0] <= UB_x)

    lyap_cplex_model.add_constraint(x_0[1] >= LB_x_dot)
    lyap_cplex_model.add_constraint(x_0[1] <= UB_x_dot)

    lyap_cplex_model.add_constraint((x_0[2]) <= UB_theta)
    lyap_cplex_model.add_constraint((x_0[3]) <= UB_theta_d)
    lyap_cplex_model.add_constraint((x_0[2]) >= LB_theta)
    lyap_cplex_model.add_constraint((x_0[3]) >= LB_theta_d)
    
    # get the final policy (force) is the action
    force_mid = lyap_cplex_model.continuous_var(name='force_mid', lb = -lyap_cplex_model.infinity)
    force = lyap_cplex_model.continuous_var(name='force', lb = -lyap_cplex_model.infinity)

    lyap_cplex_model.add_constraint(force_mid == lyap_cplex_model.sum(x_0[j] * policy_w1[0][j] for j in range(4)))
    lyap_cplex_model.add_constraint(force == lyap_cplex_model.min(lyap_cplex_model.max(force_mid, -env_params['max_force']), env_params['max_force']))

    func1_b1, func1_b2 = func1_bound(LB_theta, UB_theta)
    func2_b1, func2_b2 = func2_bound(LB_theta, UB_theta)
    func3_lb_theta, func3_lb_bias, func3_ub_theta, func3_ub_bias = func3_bound(LB_theta, UB_theta)
    func4_lb_theta, func4_lb_bias, func4_ub_theta, func4_ub_bias = func4_bound(LB_theta, UB_theta)
    func5_lb_theta, func5_lb_theta_d, func5_lb_bias, func5_ub_theta, func5_ub_theta_d, func5_ub_bias = func5_bound(LB_theta, UB_theta, LB_theta_d, UB_theta_d)
    func6_lb_theta, func6_lb_theta_d, func6_lb_bias, func6_ub_theta, func6_ub_theta_d, func6_ub_bias = func6_bound(LB_theta, UB_theta, LB_theta_d, UB_theta_d)

    #calculate big M
    #first get the upper and lower bound of x_next
    gravity = env_params['gravity']
    masspole = env_params['masspole']
    total_mass = env_params['total_mass']
    length = env_params['length']
    tau = env_params['tau']

    upper_bound_input = th.FloatTensor([UB_x, UB_x_dot, UB_theta, UB_theta_d])
    lower_bound_input = th.FloatTensor([LB_x, LB_x_dot, LB_theta, LB_theta_d])

    max_force_1, max_force_2 = weighted_bound(policy_model.policy, upper_bound_input, lower_bound_input)
    max_force_abs = np.min((np.max((np.abs(max_force_1.item()), np.abs(max_force_2.item()))), env_params['max_force']))

    temp = masspole * length * np.max((UB_theta_d**2, LB_theta_d**2))
    x_dot_dot_nextdiff_abs = max_force_abs + temp + (masspole * gravity)/2.0
    theta_dot_dot_nextdiff_abs = max_force_abs + temp/2.0 + total_mass * gravity

    upper_bound_ibp = th.FloatTensor([UB_x + UB_x_dot * tau, UB_x_dot + x_dot_dot_nextdiff_abs * tau, UB_theta + UB_theta_d * tau, UB_theta_d + theta_dot_dot_nextdiff_abs * tau])
    lower_bound_ibp = th.FloatTensor([LB_x + LB_x_dot * tau, LB_x_dot - x_dot_dot_nextdiff_abs * tau, LB_theta + LB_theta_d * tau, LB_theta_d - theta_dot_dot_nextdiff_abs * tau])

    upper_bound_ibp = th.maximum(upper_bound_ibp, upper_bound_input)
    lower_bound_ibp = th.minimum(lower_bound_ibp, lower_bound_input)

    M = network_bounds(lyap_model.lyapunov, upper_bound_ibp, lower_bound_ibp)
    xdd = lyap_cplex_model.continuous_var(name='xdd', lb = -lyap_cplex_model.infinity)
    thdd = lyap_cplex_model.continuous_var(name='thdd', lb = -lyap_cplex_model.infinity)

    lyap_cplex_model.add_constraint(xdd >= lyap_cplex_model.min(func1_b1 * force, func1_b2 * force) + func3_lb_theta * x_0[2] + func3_lb_bias + func5_lb_theta * x_0[2] + func5_lb_theta_d * x_0[3] + func5_lb_bias)
    lyap_cplex_model.add_constraint(xdd <= lyap_cplex_model.max(func1_b1 * force, func1_b2 * force) + func3_ub_theta * x_0[2] + func3_ub_bias + func5_ub_theta * x_0[2] + func5_ub_theta_d * x_0[3] + func5_ub_bias)
    
    lyap_cplex_model.add_constraint(thdd >= lyap_cplex_model.min(func2_b1 * force, func2_b2 * force) + func4_lb_theta * x_0[2] + func4_lb_bias + func6_lb_theta * x_0[2] + func6_lb_theta_d * x_0[3] + func6_lb_bias)
    lyap_cplex_model.add_constraint(thdd <= lyap_cplex_model.max(func2_b1 * force, func2_b2 * force) + func4_ub_theta * x_0[2] + func4_ub_bias + func6_ub_theta * x_0[2] + func6_ub_theta_d * x_0[3] + func6_ub_bias)

    x_0_next = {(i): lyap_cplex_model.continuous_var(name='x_0_next_{}'.format(i), lb = -lyap_cplex_model.infinity) for i in range(4)} #lb = 0 by default

    lyap_cplex_model.add_constraint(x_0_next[0] == x_0[0] + x_0[1] * env_params['tau'])
    lyap_cplex_model.add_constraint(x_0_next[2] == x_0[2] + x_0[3] * env_params['tau'])
    lyap_cplex_model.add_constraint(x_0_next[1] == x_0[1] + xdd * env_params['tau'])
    lyap_cplex_model.add_constraint(x_0_next[3] == x_0[3] + thdd * env_params['tau'])

    # get the lyapunov_function_value for (x, x_dot, theta, theta_dot)
    lyx_1 = {(i): lyap_cplex_model.continuous_var(name='lyx_1_{}'.format(i), lb = 0) for i in range(32)}
    lys_1 = {(i): lyap_cplex_model.continuous_var(name='lys_1_{}'.format(i), lb = 0) for i in range(32)}
    lyzx_1 = {(i): lyap_cplex_model.binary_var(name='lyzx_1_{}'.format(i)) for i in range(32)}
    lyx_2 = {(i): lyap_cplex_model.continuous_var(name='lyx_2_{}'.format(i), lb = 0) for i in range(32)}
    lys_2 = {(i): lyap_cplex_model.continuous_var(name='lys_2_{}'.format(i), lb = 0) for i in range(32)}
    lyzx_2 = {(i): lyap_cplex_model.binary_var(name='lyzx_2_{}'.format(i)) for i in range(32)}
    lyp_val_x0 = lyap_cplex_model.continuous_var(name='lyp_val_x0', lb = -lyap_cplex_model.infinity)

    for i in range(32):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0[j] for j in range(4)) == lyx_1[i] - lys_1[i])#+ lyapunov_b1[i] == lyx_1[i] - lys_1[i])
        lyap_cplex_model.add_constraint(lyx_1[i] <= M * lyzx_1[i])
        lyap_cplex_model.add_constraint(lys_1[i] <= M * (1 - lyzx_1[i]))
    
    for i in range(32):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1[j] for j in range(32)) == lyx_2[i] - lys_2[i])#+ lyapunov_b2[i] == lyx_2[i] - lys_2[i])
        lyap_cplex_model.add_constraint(lyx_2[i] <= M * lyzx_2[i])
        lyap_cplex_model.add_constraint(lys_2[i] <= M * (1 - lyzx_2[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0 == lyap_cplex_model.sum(lyx_2[j] * lyapunov_w3[0][j] for j in range(32)))# + lyapunov_b3[0])
    
    # get the lyapunov_function_value for (x_next, x_dot_next, theta_next, theta_dot_next)
    lyx_1_next = {(i): lyap_cplex_model.continuous_var(name='lyx_1_next_{}'.format(i)) for i in range(32)}
    lys_1_next = {(i): lyap_cplex_model.continuous_var(name='lys_1_next_{}'.format(i)) for i in range(32)}
    lyzx_1_next = {(i): lyap_cplex_model.binary_var(name='lyzx_1_next_{}'.format(i)) for i in range(32)}
    lyx_2_next = {(i): lyap_cplex_model.continuous_var(name='lyx_2_next_{}'.format(i)) for i in range(32)}
    lys_2_next = {(i): lyap_cplex_model.continuous_var(name='lys_2_next_{}'.format(i)) for i in range(32)}
    lyzx_2_next = {(i): lyap_cplex_model.binary_var(name='lyzx_2_next_{}'.format(i)) for i in range(32)}
    lyp_val_x0_next = lyap_cplex_model.continuous_var(name='lyp_val_x0_next', lb = -lyap_cplex_model.infinity)

    for i in range(32):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w1[i][j] * x_0_next[j] for j in range(4)) == lyx_1_next[i] - lys_1_next[i])
        lyap_cplex_model.add_constraint(lyx_1_next[i] <= M * lyzx_1_next[i])
        lyap_cplex_model.add_constraint(lys_1_next[i] <= M * (1 - lyzx_1_next[i]))
    
    for i in range(32):
        lyap_cplex_model.add_constraint(lyap_cplex_model.sum(lyapunov_w2[i][j] * lyx_1_next[j] for j in range(32)) == lyx_2_next[i] - lys_2_next[i])
        lyap_cplex_model.add_constraint(lyx_2_next[i] <= M * lyzx_2_next[i])
        lyap_cplex_model.add_constraint(lys_2_next[i] <= M * (1 - lyzx_2_next[i]))

    lyap_cplex_model.add_constraint(lyp_val_x0_next == lyap_cplex_model.sum(lyx_2_next[j] * lyapunov_w3[0][j] for j in range(32)))
    
    lyap_cplex_model.minimize(lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()
    sol_print = round(solution_.objective_value, 8)
    counter_example1 = solution_.get_values([x_0[0], x_0[1], x_0[2], x_0[3]])
    if sol_print < 0:
        satisfy_cond_flag = False
        print("sign not satisfied")
        x = th.FloatTensor([counter_example1])
        print("lyap value:", lyap_model(x))
    
    lyap_cplex_model.add_constraint(lyp_val_x0_next - lyp_val_x0 >= -0.00001)
    lyap_cplex_model.maximize(lyp_val_x0_next - lyp_val_x0)
    lyap_cplex_model.solve()
    solution_ = lyap_cplex_model._get_solution()

    lyap_value_diff = 0
    counter_example2 = None
    if solution_ is not None:
        counter_example2 = []
        satisfy_cond_flag = False 
        print("energy decrease not satisfied")
        sol_print = round(solution_.objective_value, 8)
        counter_example2 = solution_.get_values([x_0[0], x_0[1], x_0[2], x_0[3]])
        counter_example2 = th.FloatTensor([counter_example2])
        lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, counter_example2)
        print("lyap_value_diff:", lyap_value_diff.mean())
        print("sol_print", sol_print)

    print(satisfy_cond_flag)
    return satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff


def certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_d, ub_theta_d, lb_x = -BOUND_CART, ub_x = BOUND_CART, lb_x_dot = -BOUND_CART_SPEED, ub_x_dot = BOUND_CART_SPEED):
    print("splitting the region (near origin)")
    theta_counter = counter_example2[0][2].item()
    theta_dot_counter = counter_example2[0][3].item()

    if np.abs(theta_counter -  lb_theta) < 0.001:
        theta_split = lb_theta + 0.001
    elif np.abs(theta_counter -  ub_theta) < 0.001:
        theta_split = ub_theta - 0.001
    else:
        theta_split = (lb_theta + ub_theta)/2.0

    if np.abs(theta_dot_counter -  lb_theta_d) < 0.001:
        theta_dot_split = lb_theta_d + 0.001
    elif np.abs(theta_dot_counter -  ub_theta_d) < 0.001:
        theta_dot_split = ub_theta_d - 0.001
    else:
        theta_dot_split = (lb_theta_d + ub_theta_d)/2.0

    satisfy_cond_flag_r1, _, counter_example2_r1, _ = certify(policy_model, lyap_model, lb_theta, theta_split, theta_dot_split, ub_theta_d, lb_x, ub_x, lb_x_dot, ub_x_dot)
    if not satisfy_cond_flag_r1:
        return True
    satisfy_cond_flag_r2, _, counter_example2_r2, _ = certify(policy_model, lyap_model, theta_split, ub_theta, theta_dot_split, ub_theta_d, lb_x, ub_x, lb_x_dot, ub_x_dot)
    if not satisfy_cond_flag_r2:
        return True
    satisfy_cond_flag_r3, _, counter_example2_r3, _ = certify(policy_model, lyap_model, lb_theta, theta_split, lb_theta_d, theta_dot_split, lb_x, ub_x, lb_x_dot, ub_x_dot)
    if not satisfy_cond_flag_r3:
        return True
    satisfy_cond_flag_r4, _, counter_example2_r4, _ = certify(policy_model, lyap_model, theta_split, ub_theta, lb_theta_d, theta_dot_split, lb_x, ub_x, lb_x_dot, ub_x_dot)
    if not satisfy_cond_flag_r4:
        return True
    
    return False

def main_certify_all(policy_model, lyap_model, certify_list, certify_counter_example):
    success_flag_main = True
    for element in certify_list:
        lb_theta = element[0]
        ub_theta = element[1]

        lb_theta_d = element[2]
        ub_theta_d = element[3]

        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, -certify_diameter, certify_diameter, -certify_diameter, certify_diameter)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_d, ub_theta_d)

            if store_counterexample:  
                certify_counter_example.append(counter_example2)
                success_flag_main = False

    return success_flag_main, certify_counter_example

# both theta and theta_dot are close to the origin
def near_origin_certify_all(policy_model, lyap_model, certify_counter_example):
    success_flag_near_origin = True

    element_list = [[-0.1, 0.0, -0.1, 0.0], [-0.1, 0.0, 0.0, 0.1], [0.0, 0.1, -0.1, 0.0], [0.0, 0.1, 0.0, 0.1]]

    
    for element in element_list:
        lb_theta = element[0]
        ub_theta = element[1]
        lb_theta_dot = element[2]
        ub_theta_dot = element[3]

        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, -0.1, -certify_diameter, certify_diameter)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, -0.1, -certify_diameter, certify_diameter)
            if store_counterexample:
                certify_counter_example.append(counter_example2)
                success_flag_near_origin = False
    
        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, 0.1, certify_diameter, -certify_diameter, certify_diameter)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, 0.1, certify_diameter, -certify_diameter, certify_diameter)
            if store_counterexample:
                certify_counter_example.append(counter_example2)
                success_flag_near_origin = False
    
        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, certify_diameter, -certify_diameter, -0.1)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, certify_diameter, -certify_diameter, -0.1)
            if store_counterexample:
                certify_counter_example.append(counter_example2)
                success_flag_near_origin = False
    
        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, certify_diameter, 0.1, certify_diameter)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_dot, ub_theta_dot, -certify_diameter, certify_diameter, 0.1, certify_diameter)
            if store_counterexample:
                certify_counter_example.append(counter_example2)
                success_flag_near_origin = False

    return success_flag_near_origin, certify_counter_example

def lyap_train_main(policy_model, lyap_model, certify_counter_example, certify_list, learning_rate, args, include_near_origin_cplex, const_c = 0.12):
    print("lyap_train_main")
    N = 500             # sample size
    iter_num = 0 
    max_iters = 500
    optimizer = th.optim.Adam(list(policy_model.parameters()) + list(lyap_model.parameters()), lr=learning_rate)
    valid_num = 0

    bound = th.FloatTensor([[BOUND_CART, BOUND_CART_SPEED, BOUND_POLE, BOUND_POLE_SPEED]]).to(args.device)

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

        #find index where |x|<0.1
        x_abs = th.abs(x)
        filter_const = ((x_abs[:,2:3]<0.1) * (x_abs[:,3:4]<0.1)).flatten()

        x_regular = x[~filter_const]
        x_small = x[filter_const]
        
        lyap_value_diff_regular, _ = lyap_diff(policy_model, lyap_model, x_regular)
        lyap_value_diff_small, _ = lyap_diff(policy_model, lyap_model, x_small)
        Lyapunov_risk = F.relu(-lyap_value).mean() + 1.2*F.relu(lyap_value_diff_regular + const_c).mean() + 1.2*F.relu(lyap_value_diff_small + 0.005).mean()

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

    while np.min(policy_model.state_dict()['policy.weight'].data.cpu().numpy()[0]) < 0:
        print("negative, restart")
        policy_model = PolicyNet().to(args.device)
        lyap_model = LyapunovNet().to(args.device)
        pre_train(lyap_model, policy_model, args)
        _, _, _, _, _ = lyap_train_main(policy_model, lyap_model, certify_counter_example, [], 0.00625, args, False, 0.005)
            
    success_flag_main = True
    success_flag_near_origin = True

    counter_example_interval_list = []
    for element in certify_list:
        lb_theta = element[0]
        ub_theta = element[1]

        lb_theta_d = element[2]
        ub_theta_d = element[3]

        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, -certify_diameter, certify_diameter, -certify_diameter, certify_diameter)
        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < 0:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_d, ub_theta_d)

            if store_counterexample: 
                success_flag_main = False
                counter_example_interval_list = counter_example_interval_list + [[lb_theta, ub_theta, lb_theta_d, ub_theta_d]]
                certify_counter_example.append(th.FloatTensor(counter_example2))

    counter_example_interval_list_near_origin = []
    if include_near_origin_cplex:
        cert_list_near_origin = [[-0.1, 0.0, -0.1, 0.0], [-0.1, 0.0, 0.0, 0.1], [0.0, 0.1, -0.1, 0.0], [0.0, 0.1, 0.0, 0.1]]
        for element in cert_list_near_origin:
            lb_theta = element[0]
            ub_theta = element[1]

            lb_theta_d = element[2]
            ub_theta_d = element[3]

            success_flag_near_origin_curr = True
            satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, -certify_diameter, -0.1, -certify_diameter, certify_diameter)
            if ((not satisfy_cond_flag) and (counter_example2 is not None)):
                certify_counter_example.append(counter_example2)
                success_flag_near_origin_curr = False

            satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, 0.1, certify_diameter, -certify_diameter, certify_diameter)
            if ((not satisfy_cond_flag) and (counter_example2 is not None)):
                certify_counter_example.append(counter_example2)
                success_flag_near_origin_curr = False

            satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, -certify_diameter, certify_diameter, -certify_diameter, -0.1)
            if ((not satisfy_cond_flag) and (counter_example2 is not None)):
                certify_counter_example.append(counter_example2)
                success_flag_near_origin_curr = False

            satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, -certify_diameter, certify_diameter, 0.1, certify_diameter)
            if ((not satisfy_cond_flag) and (counter_example2 is not None)):
                certify_counter_example.append(counter_example2)
                success_flag_near_origin_curr = False

            if not success_flag_near_origin_curr:
                counter_example_interval_list_near_origin = counter_example_interval_list_near_origin + [[lb_theta, ub_theta, lb_theta_d, ub_theta_d]]
                success_flag_near_origin = False

    else:
        success_flag_near_origin = False

    return certify_counter_example, counter_example_interval_list, success_flag_main, success_flag_near_origin, counter_example_interval_list_near_origin


def lyap_train_near_origin(policy_model, lyap_model, certify_counter_example, certify_list, learning_rate, args):

    print("lyap_train_near_origin")
    N = 500             # sample size
    iter_num = 1 
    optimizer = th.optim.Adam(list(policy_model.parameters()) + list(lyap_model.parameters()), lr=learning_rate)
    valid_num = 0
    max_iters = 1000

    bound = th.FloatTensor([[BOUND_CART, BOUND_CART_SPEED, BOUND_POLE, BOUND_POLE_SPEED]]).to(args.device)

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

        Lyapunov_risk = (F.relu(-lyap_value) + 1.2*F.relu(lyap_value_diff + 0.005)).mean()#+ 1.2*(V0).pow(2)

        #print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 

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

        if (iter_num % 20 == 0) and (Lyapunov_risk.item() == 0):
            break

    print(iter_num, "Lyapunov Risk=", Lyapunov_risk.item()) 
    #certify_counter_example = []
    success_flag_near_origin = True

    counter_example_interval_list = []
    counter_example_interval_diff_list = []
    if len(certify_list) == 0:
        raise NotImplementedError
        
    for element in certify_list:
        lb_theta = element[0]
        ub_theta = element[1]

        lb_theta_d = element[2]
        ub_theta_d = element[3]

        lb_x = element[4]
        ub_x = element[5]

        lb_x_dot = element[6]
        ub_x_dot = element[7]

        satisfy_cond_flag, counter_example1, counter_example2, lyap_value_diff = certify(policy_model, lyap_model, lb_theta, ub_theta, lb_theta_d, ub_theta_d, lb_x, ub_x, lb_x_dot, ub_x_dot)

        if ((not satisfy_cond_flag) and (counter_example2 is not None)):
            store_counterexample = True
            if lyap_value_diff.item() < -0.005:
                store_counterexample = certify_split(policy_model, lyap_model, counter_example2, lb_theta, ub_theta, lb_theta_d, ub_theta_d, lb_x, ub_x, lb_x_dot, ub_x_dot)

            if store_counterexample:        
                certify_counter_example.append(counter_example2)
                success_flag_near_origin = False
                counter_example_interval_list = counter_example_interval_list + [element]
                counter_example_interval_diff_list = counter_example_interval_diff_list + [lyap_value_diff.item()]

    numerical_issues = False
    if len(counter_example_interval_diff_list) > 0:
        if (max(counter_example_interval_diff_list) < -0.005):
            numerical_issues = True

    return certify_counter_example, counter_example_interval_list, success_flag_near_origin, numerical_issues


def train_main(policy_model, lyap_model, certify_list, certify_counter_example, args, const_c = 0.12):
    success_flag_main = False
    while not success_flag_main:

        learning_rate_main = 0.00625
        certify_counter_example, counter_example_interval_list, success_flag_main, success_flag_near_origin, _ = lyap_train_main(policy_model, lyap_model, certify_counter_example, certify_list, learning_rate_main, args, include__cplex = args.include_near_origin_cplex, const_c = const_c)
        

        if success_flag_main:
            return certify_counter_example, success_flag_near_origin

        for _ in range(20):
            learning_rate_main = learning_rate_main * 0.9
            certify_counter_example, _, success_flag_main, success_flag_near_origin, _ = lyap_train_main(policy_model, lyap_model, certify_counter_example, counter_example_interval_list, learning_rate_main, args, include_near_origin_cplex = args.include_near_origin_cplex, const_c = const_c)

            if success_flag_main:
                break
        
        if success_flag_main:
            success_flag_main, certify_counter_example = main_certify_all(policy_model, lyap_model, certify_list, certify_counter_example)

    return certify_counter_example, success_flag_near_origin

def train_near_origin(policy_model, lyap_model, cert_list_near_origin_expand, certify_counter_example, args):
    success_flag_near_origin = False

    for _ in range(5):
        learning_rate_near_origin = args.lr_near_origin
        certify_counter_example, counter_example_interval_list, success_flag_near_origin, numerical_issues = lyap_train_near_origin(policy_model, lyap_model, certify_counter_example, cert_list_near_origin_expand, learning_rate_near_origin, args)

        if success_flag_near_origin:
            return certify_counter_example, success_flag_near_origin
        
        if numerical_issues:
            return certify_counter_example, success_flag_near_origin

        for ii in range(6):
            if ii == 4:
                learning_rate_near_origin = 0.0000125
                
            print("learning_rate_near_origin", learning_rate_near_origin)
            certify_counter_example, _, success_flag_near_origin_sub, numerical_issues = lyap_train_near_origin(policy_model, lyap_model, certify_counter_example, counter_example_interval_list, learning_rate_near_origin, args)
            if numerical_issues:
                return certify_counter_example, success_flag_near_origin
            
            if success_flag_near_origin_sub:
                break

        if success_flag_near_origin_sub:
            success_flag_near_origin, certify_counter_example = near_origin_certify_all(policy_model, lyap_model, certify_counter_example)

        if success_flag_near_origin:
            return certify_counter_example, success_flag_near_origin
    
    return certify_counter_example, success_flag_near_origin

def pre_train(lyap_model, policy_model, args):
    print("pre_train")
    N = 500             # sample size
    iter_num = 0 
    max_iters = 1000
    learning_rate = 0.00625
    optimizer = th.optim.Adam(list(lyap_model.parameters()), lr=learning_rate)
    valid_num = 0

    bound = th.FloatTensor([[BOUND_CART, BOUND_CART_SPEED, BOUND_POLE, BOUND_POLE_SPEED]]).to(args.device)

    x = th.zeros(N, STATE_SPACE_SIZE).uniform_(-1, 1).to(args.device) * bound.squeeze(0)
    x_all = x.clone()
    x_0 = th.zeros([1, STATE_SPACE_SIZE]).to(args.device)

    while (iter_num < max_iters):

        model_bound = BoundedModule(lyap_model, x_0)
        ptb_val = PerturbationLpNorm(norm=np.inf, x_L = -bound, x_U = bound)
        state_range = BoundedTensor(x_0, ptb_val)
        lyap_value, _ = model_bound.compute_bounds(x=(state_range,), method="backward")

        lyap_value_diff, _ = lyap_diff(policy_model, lyap_model, x)

        Lyapunov_risk = (F.relu(-lyap_value) + 1.2*F.relu(lyap_value_diff + 0.12)).mean()#+ 1.2*(V0).pow(2)

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

    print("seed:", args.seed)
    policy_model = PolicyNet().to(args.device)
    lyap_model = LyapunovNet().to(args.device)

    training_start_time = time.time()
    pre_train(lyap_model, policy_model, args)
    certify_list = []
    for idx_th in range(len(certify_list_points) - 1):
        for idx_th_d in range(len(certify_list_points) - 1):
            lb_theta = certify_list_points[idx_th]
            ub_theta = certify_list_points[idx_th + 1]

            lb_theta_d = certify_list_points[idx_th_d]
            ub_theta_d = certify_list_points[idx_th_d + 1]

            if ((lb_theta == 0) and (lb_theta_d == 0)):
                continue

            elif ((lb_theta == 0) and (ub_theta_d == 0)):
                continue

            elif ((ub_theta == 0) and (lb_theta_d == 0)):
                continue
                
            elif ((ub_theta == 0) and (ub_theta_d == 0)):
                continue

            certify_list = certify_list + [[lb_theta, ub_theta, lb_theta_d, ub_theta_d]]

    cert_list_near_origin = [[-0.1, -0.05, -0.1, -0.05], [-0.1, -0.05, -0.05, 0], [-0.1, -0.05, 0, 0.05],  [-0.1, -0.05, 0.05, 0.1],
                [-0.05, 0, -0.1, -0.05], [-0.05, 0, -0.05, 0], [-0.05, 0, 0, 0.05],  [-0.05, 0, 0.05, 0.1],
                [0.0, 0.05, -0.1, -0.05], [0.0, 0.05, -0.05, 0], [0.0, 0.05, 0, 0.05],  [0.0, 0.05, 0.05, 0.1],
                [0.05, 0.1, -0.1, -0.05], [0.05, 0.1, -0.05, 0], [0.05, 0.1, 0, 0.05],  [0.05, 0.1, 0.05, 0.1]]

    cert_list_near_origin_expand = []

    cart_bound_choice = [[-certify_diameter, -0.1, -certify_diameter, certify_diameter], [0.1, certify_diameter, -certify_diameter, certify_diameter], [-certify_diameter, certify_diameter, -certify_diameter, -0.1], [-certify_diameter, certify_diameter, 0.1, certify_diameter]]
    
    for element1 in cert_list_near_origin:
        for element2 in cart_bound_choice:
            element = element1 + element2
            cert_list_near_origin_expand = cert_list_near_origin_expand + [element]
    

    certify_counter_example = []

    pass_flag_all = False
    numer_of_trains_count = 0
    success_flag_near_origin = False
    success_flag_main = False
    number_of_near_origin_fail = 0
    while not pass_flag_all:

        certify_counter_example, success_flag_near_origin = train_main(policy_model, lyap_model, certify_list, certify_counter_example, args, const_c = 0.12)
        success_flag_main = True
        if not success_flag_near_origin:
            certify_counter_example, success_flag_near_origin = train_near_origin(policy_model, lyap_model, cert_list_near_origin_expand, certify_counter_example, args)
            success_flag_main = False
            success_flag_main, certify_counter_example = main_certify_all(policy_model, lyap_model, certify_list, certify_counter_example)

        pass_flag_all = success_flag_main and success_flag_near_origin
        if not success_flag_near_origin:
            number_of_near_origin_fail = number_of_near_origin_fail + 1

        numer_of_trains_count = numer_of_trains_count + 1
    
    training_total_time = time.time() - training_start_time
    f = open("cartpole_result.txt", "a")
    f.write("learning rate:" + str(args.lr_near_origin)+ "\n")
    f.write("args.include_near_origin_cplex:" + str(args.include_near_origin_cplex)+ "\n")
    f.write("seed:"+ str(args.seed) + ":training_time:" + str(training_total_time)+ ":numer_of_trains_count:" + str(numer_of_trains_count) + "\n\n")

    f.close()

    state_to_save_lyap = {'model_state_dict':lyap_model.state_dict()}
    state_to_save_policy = {'model_state_dict':policy_model.state_dict()}
    th.save(state_to_save_lyap, "cartpole_state_to_save_lyap_seed" + str(int(args.seed)) + str(args.lr_near_origin)+ str(args.include_near_origin_cplex) + ".pt")
    th.save(state_to_save_policy, "cartpole_state_to_save_policy_seed" + str(int(args.seed)) + str(args.lr_near_origin)+ str(args.include_near_origin_cplex) + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-near-origin-cplex", default=True, help="whether include near origin cplex in main training")
    parser.add_argument("--lr-near-origin", help="learning rate for near origin areas", type=float, default=0.000125) #both theta and theta_dot are close to origin
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0) 
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="cpu", type=str)

    args = parser.parse_args()

    train_lyapunov(args)
