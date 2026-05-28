import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# PARAMETRES
b, eps, delta = 3, 0.5, 0.0001
z_tall = 2.1

# FUNCIONS
def BL(z): return max(0, z - 2)
def BR(z): return max(0, 1 - z)

def f(x, z):
    if x <= -1: return -x - 1 + BL(z)
    if -1 < x < 1: return BL(z)*(1-x)/2 - BR(z)*(x+1)/2 + (1-x**2)*(z + 0.5*x - 1.5)
    return -x + 1 - BR(z)

def g(y, z): return -(y - z)

def h(x, y, z):
    if z <= 0: return x - z
    elif 0 < z <= 2: return x + z
    else: return -(x + 2)/(b - 2) * (y - b)

def sistema_generic(t, estat, c_val):
    x, y, z, w = estat
    return [(1/(eps*c_val))*f(x, z), (1/(eps*c_val))*g(y, z), w, c_val*w - h(x, y, z)]

# CERCA DEL ZERO
def error_c(c_proposta):
    def tall_z(t, estat): return estat[2] - z_tall
    tall_z.terminal = True
    
    # Branca baix
    lambda_u = (c_proposta + np.sqrt(c_proposta**2 + 4)) / 2
    ci_b = [-1, -1 + delta, -1 + delta*(1 + eps*c_proposta*lambda_u), delta*(lambda_u*(1 + eps*c_proposta*lambda_u))]
    sol_b = solve_ivp(lambda t, y: sistema_generic(t, y, c_proposta), (0, 50), ci_b, events=tall_z)
    
    # Branca dalt
    J = np.array([[-1/(eps*c_proposta), 0, 0, 0], [0, -1/(eps*c_proposta), 1/(eps*c_proposta), 0],[0, 0, 0, 1],[0, 3/(b-2), 0, c_proposta]])
    vaps, veps = np.linalg.eig(J)
    v_s = np.real(veps[:, np.where(vaps == np.max(vaps[vaps < 0]))[0][0]])
    ci_d = np.array([1, b, b, 0]) - delta * v_s
    sol_d = solve_ivp(lambda t, y: sistema_generic(t, y, c_proposta), (0, -50), ci_d, events=tall_z)
    
    if sol_b.t_events[0].size > 0 and sol_d.t_events[0].size > 0:
        return sol_b.y_events[0][0, 3] - sol_d.y_events[0][0, 3]
    return 999

# CERCA C ÒPTIMA
res = root_scalar(error_c, bracket=[0.2, 0.4], method='brentq')
c_opt = res.root
print(f"Velocitat optimitzada: c = {c_opt}")

# SOLUCIONS AMB AQUESTA c
def tall_final(t, estat): return estat[2] - z_tall
tall_final.terminal = True

# INTEGREM BRANCA DE L'ESQUERRA
lambda_u = (c_opt + np.sqrt(c_opt**2 + 4)) / 2
ci_b = [-1, -1 + delta, -1 + delta*(1 + eps*c_opt*lambda_u), delta*(lambda_u*(1 + eps*c_opt*lambda_u))]
sol_b = solve_ivp(lambda t, y: sistema_generic(t, y, c_opt), (0, 50), ci_b, events=tall_final, dense_output=True)

# INTEGREM A LA BRANCA DE LA DRETA
J = np.array([[-1/(eps*c_opt), 0, 0, 0], [0, -1/(eps*c_opt), 1/(eps*c_opt), 0],[0, 0, 0, 1],[0, 3/(b-2), 0, c_opt]])
vaps, veps = np.linalg.eig(J)
v_s = np.real(veps[:, np.where(vaps == np.max(vaps[vaps < 0]))[0][0]])
ci_d = np.array([1, b, b, 0]) - delta * v_s
sol_d = solve_ivp(lambda t, y: sistema_generic(t, y, c_opt), (0, -50), ci_d, events=tall_final, dense_output=True)

# GRÀFIQUES
plt.figure(figsize=(10, 5))

t_tall_b = sol_b.t_events[0][0]
t_tall_d = sol_d.t_events[0][0]

t_dalt_ajustat = sol_d.t[::-1] - t_tall_d + t_tall_b
z_dalt_ajustat = sol_d.y[2][::-1]
w_dalt_ajustat = sol_d.y[3][::-1]

plt.subplot(1, 2, 1)
plt.plot(sol_b.t, sol_b.y[2], 'b', label='z (baix)')
plt.plot(t_dalt_ajustat, z_dalt_ajustat, 'r', label='z (dalt)')
plt.axvline(t_tall_b, color='k', linestyle='--', alpha=0.3)
plt.title(f"Evolució Temporal (c = {c_opt:.4f})")
plt.xlabel("Temps ajustat")
plt.ylabel("z")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sol_b.y[2], sol_b.y[3], 'b', label='Branca baix')
plt.plot(z_dalt_ajustat, w_dalt_ajustat, 'r', label='Branca dalt')
plt.scatter([z_tall], [sol_b.y_events[0][0, 3]], color='green', zorder=5, label='Punt de connexió')
plt.title("Retrat de Fases (w , z)")
plt.xlabel("z")
plt.ylabel("w")
plt.legend()
plt.tight_layout()
plt.show()