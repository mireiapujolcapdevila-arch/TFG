
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

b = 3
delta = 1e-3  

def f(x, z):
    return -x**3 + 3*x**2 + 4*z - 8

def g(y, z):
    return -(y - z)

def h(x, y, z):
    if z <= 0.0:
        return x - z
    elif 0.0 < z < 2.0:
        return x + z
    else:
        return - (x + 2.0) / (b - 2.0) * (z - b)

def x_l(z):
    if(z > 2): z = 2 - 1e-8
    coefs = [-1, 3, 0, 4*z - 8]
    arrels = np.roots(coefs)
    arrels_reals = arrels[np.isreal(arrels)].real
    arrels_valides = [arrel for arrel in arrels_reals if arrel <= 0.0]
    if arrels_valides:
        return arrels_valides[0]
    else:
        return 0.0 


def x_r(z):
    if(z < 1): z = 1 + 1e-8
    coefs = [-1, 3, 0, 4*z - 8]
    arrels = np.roots(coefs)
    arrels_reals = arrels[np.isreal(arrels)].real
    arrels_valides = [arrel for arrel in arrels_reals if arrel >= 2.0]
    if arrels_valides:
        return arrels_valides[0]
    else:
        return 2.0  


def x_r(z):
    if(z < 1): z = 1 + 1e-8
    coefs = [-1, 3, 0, 4*z - 8]
    arrels = np.roots(coefs)
    arrels_reals = arrels[np.isreal(arrels)].real
    arrels_valides = [arrel for arrel in arrels_reals if arrel >= 2.0]
    if arrels_valides:
        return arrels_valides[0]
    else:
        return 2.0  

def H_l(z):
    return h(x_l(z), z, z)

def H_r(z):
    return h(x_r(z), z, z)

def sistema_baix(t, estat, c_val):
    z, w = estat
    return [w, c_val*w - H_l(z)]

def sistema_dalt(t, estat, c_val):
    z, w = estat
    return [w, c_val*w - H_r(z)]

def tall_z_b(t, y):
    return y[0] - 2.0
tall_z_b.terminal = True

def tall_z_d(t, y):
    return y[0] - 2.0
tall_z_d.terminal = True

def velocitat(c):
    c = float(c)
    
    # --- BRANCA ESQUERRA  ---
    
    coefs_eq = [1, -3, -4, 8]
    arrels_eq = np.roots(coefs_eq)
    z_eq = [r.real for r in arrels_eq if np.isreal(r) and r.real <= 0][0]
    
    p1_b = np.array([z_eq, 0.0])  
    
    df_dx = -3 * z_eq**2 + 6 * z_eq
    dx_dz = -4.0 / df_dx
    dh_dz = dx_dz - 1.0  
    
    vap_b = (c + (c**2 - 4.0 * dh_dz)**(1/2)) / 2.0
    vep_b = np.array([1.0, vap_b])
    ci_b = p1_b + vep_b * delta
    
    sol_baix = solve_ivp(lambda t, y: sistema_baix(t, y, c), t_span=(0, 300), y0=ci_b, method='BDF', events=tall_z_b)
    
    # --- BRANCA DRETA ---
    p1_d = np.array([b, 0.0])
    x = x_r(b)  
    vap_d = (c - (c**2 + 4.0 * (x + 2.0) / (b - 2.0))**(1/2)) / 2.0
    vep_d = np.array([1.0, vap_d])
    ci_d = p1_d - vep_d * delta
    
    sol_dalt = solve_ivp(lambda t, y: sistema_dalt(t, y, c), t_span=(0, -300), y0=ci_d, method='BDF', events=tall_z_d)
    
    if len(sol_baix.y_events[0]) == 0 or len(sol_dalt.y_events[0]) == 0:
        return 999.0 if c >= 0 else -999.0
        
    w_baix = sol_baix.y_events[0][0][1]
    w_dalt = sol_dalt.y_events[0][0][1]
    
    return w_baix - w_dalt


# EXECUCIÓ I DIBUIX

c_sol = fsolve(velocitat, x0=[0.3])
c_opt = c_sol[0]
print(f"Velocitat de l'ona trobada (Model Cúbic Ajustat): c = {c_opt:.5f}")

# TRAJECTÒRIES AMB LA c CORRECTA ---
coefs_eq = [1, -3, -4, 8]
arrels_eq = np.roots(coefs_eq)
z_eq = [r.real for r in arrels_eq if np.isreal(r) and r.real <= 0][0]
p1_b = np.array([z_eq, 0.0])  

df_dx = -3 * z_eq**2 + 6 * z_eq
dx_dz = -4.0 / df_dx
dh_dz = dx_dz - 1.0  
vap_b = (c_opt + (c_opt**2 - 4.0 * dh_dz)**(1/2)) / 2.0
vep_b = np.array([1.0, vap_b])
ci_b = p1_b + vep_b * delta

sol_baix = solve_ivp(lambda t, y: sistema_baix(t, y, c_opt), t_span=(0, 300), y0=ci_b, method='BDF', events=tall_z_b, max_step=0.5)

p1_d = np.array([b, 0.0])
x = x_r(b)
vap_d = (c_opt - (c_opt**2 + 4.0 * (x + 2.0) / (b - 2.0))**(1/2)) / 2.0
vep_d = np.array([1.0, vap_d])
ci_d = p1_d - vep_d * delta

sol_dalt = solve_ivp(lambda t, y: sistema_dalt(t, y, c_opt), t_span=(0, -300), y0=ci_d, method='BDF', events=tall_z_d, max_step=0.5)

t_baix = sol_baix.t
z_baix = sol_baix.y[0]
w_baix = sol_baix.y[1]

t_dalt = sol_dalt.t
z_dalt = sol_dalt.y[0]
w_dalt = sol_dalt.y[1]

t_match = t_baix[-1]
t_dalt_offset = t_dalt[-1]  


t_dalt_shifted = (t_dalt - t_dalt_offset + t_match)[::-1]
z_dalt_rev = z_dalt[::-1]
w_dalt_rev = w_dalt[::-1]

t_total = np.concatenate((t_baix, t_dalt_shifted[1:]))
z_total = np.concatenate((z_baix, z_dalt_rev[1:]))
w_total = np.concatenate((w_baix, w_dalt_rev[1:]))

# DIBUIX DE LES GRÀFIQUES 
plt.figure(figsize=(10, 8))

# Gràfica z(t)
plt.subplot(2, 1, 1)
plt.plot(t_total, z_total, 'b-', linewidth=2, label='z(t)')
plt.axvline(x=t_match, color='k', linestyle='--', alpha=0.5, label='Connexió (z=2)')
plt.ylabel('z(t)', fontsize=12)
plt.title(f"Evolució temporal de z(t) i w(t) per c = {c_opt:.4f}", fontsize=14)
plt.legend()
plt.grid(True)

# Gràfica w(t)
plt.subplot(2, 1, 2)
plt.plot(t_total, w_total, 'r-', linewidth=2, label='w(t)')
plt.axvline(x=t_match, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Temps (t)', fontsize=12)
plt.ylabel('w(t)', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Pla de Fases (z vs w)
plt.figure(figsize=(8, 6))
plt.plot(z_baix, w_baix, 'b-', linewidth=2, label='Branca baixa (d\'esquerra a centre)')
plt.plot(z_dalt, w_dalt, 'g-', linewidth=2, label='Branca alta (de dreta a centre)')
plt.plot(2.0, w_baix[-1], 'ro', markersize=6, label='Punt d\'empalmament (z=2)')
plt.xlabel('z', fontsize=12)
plt.ylabel('w = dz/dt', fontsize=12)
plt.title(f"Pla de fases per a l'ona viatgera (c = {c_opt:.4f})", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
# CÀLCUL DE x(t) I y(t) 
y_total = z_total  

x_baix = np.array([x_l(zi) for zi in z_baix])
x_dalt_rev = np.array([x_r(zi) for zi in z_dalt_rev])

x_total = np.concatenate((x_baix, x_dalt_rev[1:]))

# DIBUIX DE TOTES LES VARIABLES RESPECTE EL TEMPS 
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  

fig.suptitle(f"Evolució temporal del sistema complet (c = {c_opt:.4f})", fontsize=16)

axs[0].plot(t_total, z_total, 'b-', linewidth=2, label='z(t)')
axs[0].axvline(x=t_match, color='k', linestyle='--', alpha=0.5, label='Connexió z = 2')
axs[0].set_ylabel('z(t)', fontsize=12)
axs[0].legend(loc='upper left')
axs[0].grid(True)

axs[1].plot(t_total, w_total, 'r-', linewidth=2, label='w(t)')
axs[1].axvline(x=t_match, color='k', linestyle='--', alpha=0.5, label='Connexió z = 2')
axs[1].axvline(x=t_match, color='k', linestyle='--', alpha=0.5)
axs[1].set_ylabel('w(t) = dz/dt', fontsize=12)
axs[1].legend(loc='upper left')
axs[1].grid(True)

axs[2].plot(t_total, x_total, 'g-', linewidth=2, label='x(t)')
axs[2].axvline(x=t_match, color='k', linestyle='--', alpha=0.5, label='Connexió z = 2')
axs[2].axvline(x=t_match, color='k', linestyle='--', alpha=0.5)
axs[2].set_ylabel('x(t)', fontsize=12)
axs[2].set_xlabel('Temps (t)', fontsize=12)
axs[2].legend(loc='upper left')
axs[2].grid(True)

axs[3].plot(t_total, y_total, 'm-', linewidth=2, label='y(t)')
axs[3].axvline(x=t_match, color='k', linestyle='--', alpha=0.5, label='Connexió z = 2')
axs[3].axvline(x=t_match, color='k', linestyle='--', alpha=0.5)
axs[3].set_ylabel('y(t)', fontsize=12)
axs[3].set_xlabel('Temps ', fontsize=12)
axs[3].legend(loc='upper left')
axs[3].grid(True)

plt.tight_layout()
plt.show()