import numpy as np
import torch.nn as nn
import torch

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.patches as patches

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
plt.subplot(1, 2, 1)
def func_NLC_raw(x1, x2):
    return np.tanh((0.22351787984371185 - 0.98369121551513672 * np.tanh((-1.5544859170913696 - 0.015240113250911236 * x1 + 0.023013174533843994 * x2)) - 1.4729056358337402 * np.tanh((-1.0263288021087646 - 0.017298899590969086 * x1 + 0.027189154177904129 * x2)) - 1.1969460248947144 * np.tanh((-0.97856545448303223 - 0.017603849992156029 * x1 + 0.027988830581307411 * x2)) - 0.95498400926589966 * np.tanh((1.034111499786377 - 0.089762948453426361 * x1 + 0.31161656975746155 * x2)) - 1.4816581010818481 * np.tanh((1.0560840368270874 + 0.33148494362831116 * x1 - 0.015659866854548454 * x2)) - 1.4398871660232544 * np.tanh((1.1782207489013672 - 0.29730242490768433 * x1 - 0.31557342410087585 * x2))))

def func_NLC(x1, x2):
    return func_NLC_raw(x1, x2) - func_NLC_raw(0, 0)
    
def func_UNL_raw(x1, x2):
    return np.tanh((1.2453790903091431 + 0.83704346418380737 * np.tanh((-1.4990885257720947 + 1.9625025987625122 * x1 + 0.021317984908819199 * x2)) - 0.92947345972061157 * np.tanh((-1.4811422824859619 + 0.18486802279949188 * x1 - 0.057087451219558716 * x2)) + 0.22155895829200745 * np.tanh((-0.99772018194198608 - 0.18100647628307343 * x1 - 0.51021850109100342 * x2)) + 0.11722179502248764 * np.tanh((-0.93738692998886108 + 0.69252479076385498 * x1 + 0.41277414560317993 * x2)) + 0.35415962338447571 * np.tanh((-0.82583862543106079 + 0.76746141910552979 * x1 + 0.38658440113067627 * x2)) - 0.83168512582778931 * np.tanh((1.0896867513656616 + 1.261768102645874 * x1 + 0.19598305225372314 * x2))))

def func_UNL(x1, x2):
    return func_UNL_raw(x1, x2) - func_UNL_raw(0, 0)

def func_SOS(theta, theta_dot):
    V = 8.53288811054e-21*theta+2.95518972707e-21*theta_dot+3.06048873761*theta**2+0.59033716601*theta*theta_dot+0.349435517215*theta_dot**2+3.05214303992e-20*theta**3-2.81417742123e-20*theta**2*theta_dot-2.72832317353e-20*theta*theta_dot**2+2.8529442857e-20*theta_dot**3+2.14423767387*theta**4+0.45382434525*theta**3*theta_dot+0.0824392565976*theta**2*theta_dot**2+0.108514030341*theta*theta_dot**3+0.237070176836*theta_dot**4+3.32026124566*theta**6+0.488563910427*theta**5*theta_dot+0.218917824693*theta**4*theta_dot**2+0.10612433807*theta**3*theta_dot**3+0.139003892774*theta**2*theta_dot**4-0.0251451870357*theta*theta_dot**5+0.217149991311*theta_dot**6
    return V

def func_LQR(theta, theta_dot):
    V = theta_dot*(0.0179479842292839*theta_dot + 0.015102669871863*theta) + theta*(0.015102669871863*theta_dot + 1.02764190924855*theta)
    return V

class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet,self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.lyapunov(x)

lyap_model = LyapunovNet()

saved_state = torch.load("pendulum_state_to_save_lyap.pt", map_location=lambda storage, loc: storage)
saved_state = saved_state['model_state_dict']
lyap_model.load_state_dict(saved_state) 

lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
lyapunov_b1 = lyap_model.state_dict()['lyapunov.0.bias'].data.cpu().numpy()
lyapunov_b1 = np.expand_dims(lyapunov_b1, axis=(1,2))
lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
lyapunov_b2 = lyap_model.state_dict()['lyapunov.2.bias'].data.cpu().numpy()
lyapunov_b2 = np.expand_dims(lyapunov_b2, axis=(1,2))

def func_ours_raw(x1_expand, x2_expand):
    layer1_result =  np.maximum(np.tensordot(lyapunov_w1, np.array([x1_expand, x2_expand]), axes = 1) + lyapunov_b1, 0)
    V_NN =  (np.tensordot(lyapunov_w2, layer1_result, axes = 1) + lyapunov_b2)[0]
    return V_NN

def func_ours(x1_expand, x2_expand):
    return func_ours_raw(x1_expand, x2_expand) - func_ours_raw(0, 0)[0][0]

train_diameter = 12
X_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
Y_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
x1_expand, x2_expand = np.meshgrid(X_expand,Y_expand)

V_NLC = func_NLC(x1_expand, x2_expand)
V_UNL = func_UNL(x1_expand, x2_expand)
V_SOS = func_SOS(x1_expand, x2_expand)
V_LQR = func_LQR(x1_expand, x2_expand)
V_ours = func_ours(x1_expand, x2_expand)

# cut_SOS = 7.011072369008178
# cut_UNL = 0.4833272213232254
# cut_LQR = 0.6024753750547
# cut_ours = 24.930882314476364
# cut_NLC = 0.836886144153583

cut_SOS = 7.011072369008178
cut_UNL = 0.4833272213232254 - func_UNL_raw(0, 0)
cut_LQR = 0.6024753750547
cut_ours = 24.930882314476364 - func_ours_raw(0, 0)[0][0]
cut_NLC = 0.836886144153583 - func_NLC_raw(0, 0)

plt.rcParams.update({'font.size': 10})
plt.contour(x1_expand,x2_expand,V_ours-cut_ours,0,linewidths=1.5, colors = 'g')
plt.contour(x1_expand,x2_expand,V_NLC-cut_NLC,0,linewidths=1.5, colors = 'm')
plt.contour(x1_expand,x2_expand,V_UNL-cut_UNL,0,linewidths=1.5, colors = 'tab:purple')
plt.contour(x1_expand,x2_expand,V_SOS-cut_SOS,0,linewidths=1.5, colors = 'k')
plt.contour(x1_expand,x2_expand,V_LQR-cut_LQR,0,linewidths=1.5, colors = 'b')


plt.contour(x1_expand,x2_expand,V_ours,8,linewidths=0.4, colors='k')
c1 = plt.contourf(x1_expand,x2_expand,V_ours,8,alpha=0.4,cmap=cm.coolwarm)
plt.colorbar(c1)

rect = patches.Rectangle((-train_diameter, -train_diameter), train_diameter * 2, train_diameter * 2, linewidth=2, edgecolor='r', facecolor='none')

plt.gca().add_patch(rect)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity')


plt.subplot(1, 2, 2)

def func_NLC(x1, x2):
    return np.tanh((0.45639458298683167 + 1.3460793495178223 * np.tanh((-1.2011017799377441 - 0.34940284490585327 * x1 + 1.0865392684936523 * x2)) - 1.1668504476547241 * np.tanh((-1.0918536186218262 + 0.034315034747123718 * x1 - 0.048630755394697189 * x2)) + 1.1657252311706543 * np.tanh((1.0109463930130005 - 0.06838352233171463 * x1 - 0.003784568514674902 * x2)) - 0.59988445043563843 * np.tanh((1.0811564922332764 - 1.6724585294723511 * x1 - 0.28754439949989319 * x2)) + 0.78500986099243164 * np.tanh((1.2918539047241211 - 0.091193288564682007 * x1 - 0.020296718925237656 * x2)) - 1.4939277172088623 * np.tanh((1.4826024770736694 + 0.59365767240524292 * x1 + 1.8921090364456177 * x2))))

def func_UNL(x1, x2):
    return np.tanh((1.195081353187561 + 0.88736289739608765 * np.tanh((-0.8964008092880249 + 2.7222888469696045 * x1 - 1.3374241590499878 * x2)) - 0.80505263805389404 * np.tanh((0.35582336783409119 - 3.7293601036071777 * x1 - 3.1488761901855469 * x2)) - 1.9080113172531128 * np.tanh((0.46791648864746094 + 2.4647660255432129 * x1 + 1.8052127361297607 * x2)) - 2.0625097751617432 * np.tanh((1.0935239791870117 + 0.054358616471290588 * x1 - 1.8298448324203491 * x2)) + 1.2586170434951782 * np.tanh((1.3948873281478882 - 0.35110947489738464 * x1 - 0.35958966612815857 * x2)) + 1.0920743942260742 * np.tanh((2.3356473445892334 + 2.5617868900299072 * x1 - 1.5179961919784546 * x2))))


def func_SOS(d_e, theta_e):
    V = -1.20252137787e-20*d_e-3.21874913032e-20*theta_e+1.88914593906*d_e**2+1.67029419882*d_e*theta_e+2.02759842794*theta_e**2+9.30338599877e-20*d_e**3+2.80351859383e-19*d_e**2*theta_e-1.27278424415e-19*d_e*theta_e**2+7.63216452035e-20*theta_e**3+0.728222112216*d_e**4+0.888783069252*d_e**3*theta_e+1.59401261221*d_e**2*theta_e**2+0.397348514726*d_e*theta_e**3+0.981099261013*theta_e**4+0.681057463985*d_e**6+1.07407534132*d_e**5*theta_e+2.46591884797*d_e**4*theta_e**2+1.46682720409*d_e**3*theta_e**3+1.67863422482*d_e**2*theta_e**4+0.142457190733*d_e*theta_e**5+0.821991107265*theta_e**6
    return V


def func_LQR(d_e, theta_e):
    V = d_e*(0.894462973809601*d_e + 0.5*theta_e) + theta_e*(0.500000000000001*d_e + 0.670713087739653*theta_e)
    return V

class LyapunovNet(nn.Module):
    def __init__(self):
        super(LyapunovNet,self).__init__()
        self.lyapunov = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.lyapunov(x)

lyap_model = LyapunovNet()

saved_state = torch.load("path_tracking_state_to_save_lyap.pt", map_location=lambda storage, loc: storage)
saved_state = saved_state['model_state_dict']
lyap_model.load_state_dict(saved_state) 

lyapunov_w1 = lyap_model.state_dict()['lyapunov.0.weight'].data.cpu().numpy()
lyapunov_b1 = lyap_model.state_dict()['lyapunov.0.bias'].data.cpu().numpy()
lyapunov_b1 = np.expand_dims(lyapunov_b1, axis=(1,2))
lyapunov_w2 = lyap_model.state_dict()['lyapunov.2.weight'].data.cpu().numpy()
lyapunov_b2 = lyap_model.state_dict()['lyapunov.2.bias'].data.cpu().numpy()
lyapunov_b2 = np.expand_dims(lyapunov_b2, axis=(1,2))

def func_ours_raw(x1_expand, x2_expand):
    layer1_result =  np.maximum(np.tensordot(lyapunov_w1, np.array([x1_expand, x2_expand]), axes = 1) + lyapunov_b1, 0)
    V_NN =  (np.tensordot(lyapunov_w2, layer1_result, axes = 1) + lyapunov_b2)[0]
    return V_NN

def func_ours(x1_expand, x2_expand):
    return func_ours_raw(x1_expand, x2_expand) - func_ours_raw(0, 0)[0][0]

train_diameter = 3.0
X_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
Y_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
x1_expand_ours, x2_expand_ours = np.meshgrid(X_expand,Y_expand)

train_diameter = 1.5
X_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
Y_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
x1_expand_NLC, x2_expand_NLC = np.meshgrid(X_expand,Y_expand)

train_diameter = 0.8
X_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
Y_expand = np.linspace(-train_diameter * 1.1, train_diameter * 1.1, 2200)
x1_expand, x2_expand = np.meshgrid(X_expand,Y_expand)


V_NLC = func_NLC(x1_expand_NLC, x2_expand_NLC) - func_NLC_raw(0, 0)
V_UNL = func_UNL(x1_expand, x2_expand) - func_UNL_raw(0, 0)
V_SOS = func_SOS(x1_expand, x2_expand)
V_LQR = func_LQR(x1_expand, x2_expand)
V_ours = func_ours(x1_expand_ours, x2_expand_ours) - func_ours_raw(0, 0)[0][0]

cut_SOS = 1.44
cut_UNL = 0.627 - func_UNL_raw(0, 0)
cut_LQR = 0.19
cut_ours = 63 - func_ours_raw(0, 0)[0][0]
cut_NLC = 0.56 - func_NLC_raw(0, 0)

plt.rcParams.update({'font.size': 10})
plt.contour(x1_expand_ours, x2_expand_ours,V_ours-cut_ours,0,linewidths=1.5, colors = 'g', linestyles='-')
plt.contour(x1_expand_NLC, x2_expand_NLC,V_NLC-cut_NLC,0,linewidths=1.5, colors = 'm')
plt.contour(x1_expand,x2_expand,V_UNL-cut_UNL,0,linewidths=1.5, colors = 'tab:purple')
plt.contour(x1_expand,x2_expand,V_SOS-cut_SOS,0,linewidths=1.5, colors = 'k')
plt.contour(x1_expand,x2_expand,V_LQR-cut_LQR,0,linewidths=1.5, colors = 'b')


plt.contour(x1_expand_ours,x2_expand_ours,V_ours,8,linewidths=0.4, colors='k')
c1 = plt.contourf(x1_expand_ours,x2_expand_ours,V_ours,8,alpha=0.4,cmap=cm.coolwarm)
plt.colorbar(c1)

train_diameter = 3.0
rect = patches.Rectangle((-train_diameter, -train_diameter), train_diameter * 2, train_diameter * 2, linewidth=2, edgecolor='r', facecolor='none')

plt.gca().add_patch(rect)
plt.xlabel('Distance Error')
plt.ylabel('Angle Error')

plt.legend([plt.Rectangle((0,0),1,2,color='g',fill=False,linewidth = 1.5),plt.Rectangle((0,0),1,2,color='m',fill=False,linewidth = 1.5),\
            plt.Rectangle((0,0),1,2,color='tab:purple',fill=False,linewidth = 1.5),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 1.5),\
            plt.Rectangle((0,0),1,2,color='b',fill=False,linewidth = 1.5),rect],['Our Approach','NLC','UNL','SOS', 'LQR', 'Valid Region'], \
            bbox_to_anchor =(-0.25,-0.32), loc='lower center', borderaxespad=0, ncol = 6)

    
plt.savefig('roa_fig_all.png', dpi=300, bbox_inches='tight')
