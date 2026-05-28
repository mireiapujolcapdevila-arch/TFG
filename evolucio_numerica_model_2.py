import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# PARÀMETRES
b = 3.0
delta = 1e-5

def H(z):
    if z <= 0:
        return -1.0-z
    elif z < 2:
        return z - 1.0
    else:
        return -3.0 / (b-2.0)* (z-b)

def error_tret(c):
    lambda_u = (c + np.sqrt(c**2+4.0)) / 2.0
    z0_l = -1.0 + delta
    w0_l = lambda_u * delta
    
    dH_r = -3.0 / (b-2.0)
    lambda_s =  (c-np.sqrt(c**2-4.0*dH_r)) / 2.0
    z0_r = b - delta
    w0_r = - lambda_s * delta
    
    def esdeveniment_z2(xi, estat): return estat[0] - 2.0 
    esdeveniment_z2.terminal = True
    
    esdeveniment_z2.direction = 1
    sol_l = solve_ivp(lambda xi, y: [y[1], c * y[1]- H(y[0])], (0,100), [z0_l, w0_l], events = esdeveniment_z2, rtol = 1e-9, atol = 1e-9)
    
    esdeveniment_z2.direction = -1
    sol_r = solve_ivp(lambda xi, y: [y[1], c*y[1]- H(y[0])], (0, -100), [z0_r, w0_r], events = esdeveniment_z2, rtol = 1e-9, atol = 1e-9)
    
    if len(sol_l.y_events[0]) == 0 or len(sol_r.y_events[0] ) == 0:
        return 999.0 
    
    w_l = sol_l.y_events[0][0][1]
    w_r = sol_r.y_events[0][0][1]
    
    return w_l-w_r

# CERCA DE LA VELOCITAT
c_optim = brentq(error_tret, 0.1, 5.0)
print(f"Valor optimitzat: c = {c_optim:.6f}\n")

lambda_u = (c_optim + np.sqrt(c_optim**2 + 4.0)) / 2.0 
lambda_s = (c_optim - np.sqrt(c_optim**2 - 4.0 * (-3.0/(b-2.0)))) / 2.0

def esdeveniment_z2(xi, estat): return estat[0] - 2.0
esdeveniment_z2.terminal = True
esdeveniment_z2.direction = 1

sol_l = solve_ivp(lambda xi, y: [y[1], c_optim * y[1] - H(y[0])], (0, 100), [-1.0 + delta, lambda_u * delta], events=esdeveniment_z2, rtol=1e-9, atol=1e-9)
esdeveniment_z2.direction = -1
sol_r = solve_ivp(lambda xi, y: [y[1], c_optim * y[1] - H(y[0])], (0, -100), [b - delta, -lambda_s * delta], events=esdeveniment_z2, rtol=1e-9, atol=1e-9)

xi_l = sol_l.t - sol_l.t[-1]
xi_r = sol_r.t - sol_r.t[-1]

xi_total = np.concatenate((xi_l, xi_r))
z_total = np.concatenate((sol_l.y[0], sol_r.y[0]))
w_total = np.concatenate((sol_l.y[1], sol_r.y[1]))

fig1,ax1 = plt.subplots(figsize= (8,6))

ax1.plot(sol_l.y[0], sol_l.y[1], 'b-', linewidth = 2, label = 'Branca esquerra' )
ax1.plot(sol_r.y[0], sol_r.y[1], 'r-', linewidth = 2, label = 'Branca dreta')
ax1.axvline(x = 2.0, color = 'g', linestyle = '--', label = 'Secció de salt z = 2')
ax1.scatter([2.0], [sol_l.y[1][-1]], color = 'black', zorder = 5, label = f'Encaix (w = {sol_l.y[1][-1]:.4f})')
ax1.set_title(f'Espai de Fases (z,w) amb c ~ {c_optim: .5f}')
ax1.set_xlabel('z')
ax1.set_ylabel('w')
ax1.grid(True)
ax1.legend()
fig1.tight_layout()
plt.show()

fig2,axs2 = plt.subplots(1, 2, figsize= (14,5))

axs2[0].plot(xi_l, sol_l.y[0], 'b-', linewidth = 2, label = 'Branca esquera' )
axs2[0].plot(xi_r, sol_r.y[0], 'r-', linewidth = 2, label = 'Branca dreta')
axs2[0].axvline(x=0.0, color = 'g', linestyle = '--', label = 'Punt de salt z = 2')
axs2[0].set_title(r'Evolució de $z(\xi)$')
axs2[0].set_xlabel(r'$\xi$ (temps de l\'ona)')
axs2[0].set_ylabel('z')
axs2[0].grid(True)
axs2[0].legend()

axs2[1].plot(xi_l, sol_l.y[1], 'b-', linewidth = 2, label = 'Branca esquera' )
axs2[1].plot(xi_r, sol_r.y[1], 'r-', linewidth = 2, label = 'Branca dreta')
axs2[1].axvline(x=0.0, color = 'g', linestyle = '--', label = 'Punt de salt z = 2')
axs2[1].set_title(r'Evolució de $w(\xi)$')
axs2[1].set_xlabel(r'$\xi$ (temps de l\'ona)')
axs2[1].set_ylabel('w')
axs2[1].grid(True)
axs2[1].legend()

fig2.tight_layout()
plt.show()
