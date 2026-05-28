import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# PARÀMETRES

eps = 0.5
D = 1.96 * (10**-3)
L = 4.0
Nx = 100
x_xarxa = np.linspace(0, L, Nx)
dx = x_xarxa[1] - x_xarxa[0]
b = 3.0

# FUNCIONS DEL MODEL VECTORIALS

def f_vec(X,Z):
    
    res = np.zeros_like(X)
    m1 = X <= -1.0
    m2 = (X > -1.0) & (X< 1.0)
    m3 = X>=1.0
    
    BLz = np.maximum(0.0, Z -2.0)
    BRz = np.maximum(0.0, 1.0-Z)
    
    res[m1] = -X[m1]-1.0+ BLz[m1]
    
    term1 = BLz[m2] * (1.0 - X[m2]) / 2.0
    term2 = BRz[m2] * (X[m2] + 1.0) / 2.0
    term3 = (1.0 - X[m2]**2)*(Z[m2] + 0.5*X[m2]-1.5)
    res[m2] = term1 - term2 + term3
    res[m3] = -X[m3] + 1.0 - BRz[m3]
    
    return res

def g_vec(Y,Z):
    return -(Y-Z)

def h_vec(X, Y, Z, b=b):
    res = np.zeros_like(X)
    m1 = Z <= 0.0
    m2 = (Z > 0.0) & (Z <= 2.0)
    m3 = Z > 2.0
    res[m1] = X[m1] - Z[m1]
    res[m2] = X[m2] + Z[m2]
    res[m3] = -(X[m3] + 2.0) / (b - 2.0) * (Y[m3] - b)
    return res


# APROXIMACIÓ SEGONA DERIVADA - DIFERÈNCIES CENTRADES SEGON ORDRE

def laplacia(z,dx):
    lap = np.zeros_like(z)
    lap[1:-1] = (z[:-2]-2*z[1:-1]+z[2:]) / dx**2 
    return D * lap

# SISTEMA D'EDOs
def sistema(t, Y):
    X = Y[:Nx]
    Yvar = Y[Nx:2*Nx]
    Z = Y[2*Nx:]
    
    dXdt = (1.0 / eps)*f_vec(X,Z)
    dYdt = (1.0/ eps)*g_vec(Yvar,Z)
    dZdt= laplacia(Z,dx) + h_vec(X,Yvar,Z)
    
    dXdt[0] = dXdt[-1] = 0.0
    dYdt[0] = dYdt[-1] = 0.0
    dZdt[0] = dZdt[-1] = 0.0
    
    return np.concatenate([dXdt, dYdt, dZdt])

# CONDICIONS INICIALS 
X0 = -1.0* np.ones(Nx)
Y0 = -1.0* np.ones(Nx)
Z0 = -1.0* np.ones(Nx)

# INJECCIÓ LOCAL DE POTASSI

centre = Nx // 2
mig = Nx // 10 
Z0[centre-mig:centre+mig] = 2.0

y0 = np.concatenate([X0,Y0,Z0])

#INTEGRACIÓ TEMPORAL 

t_span = (0.0, 280.0)
t_eval = np.linspace(0, 280, 200)
sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)
if sol.success:
    Z_sol = sol.y[2*Nx:].reshape((Nx, len(sol.t))).T
    X_sol = sol.y[:Nx].reshape((Nx, len(sol.t))).T
    Y_sol = sol.y[Nx:2*Nx].reshape((Nx, len(sol.t))).T
    
    # DIAGRAMA DE CALOR
    plt.figure(figsize = (10,6))
    plt.pcolormesh(x_xarxa, sol.t, Z_sol, shading = 'auto', cmap = 'hot')
    plt.colorbar(label = 'z (concentració $K^+$)')
    plt.xlabel('Posició r')
    plt.ylabel('Temps t')
    plt.title('Diagrama de calor de la variable z $K^+$')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize = (10,6))
    plt.pcolormesh(x_xarxa, sol.t, X_sol, shading = 'auto', cmap = 'coolwarm')
    plt.colorbar(label = 'x (potencial neuronal)')
    plt.xlabel('Posició r')
    plt.ylabel('Temps t')
    plt.title('Diagrama de calor de la variable x (potencial membrana neuronal)')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize = (10,6))
    plt.pcolormesh(x_xarxa, sol.t, Y_sol, shading = 'auto', cmap = 'coolwarm')
    plt.colorbar(label = 'y (potencial astròcit)')
    plt.xlabel('Posició r')
    plt.ylabel('Temps t')
    plt.title('Diagrama de calor de la variable y (potencial membrana astròcit)')
    plt.tight_layout()
    plt.show()
    
    # PERFIL FINAL
    
    plt.figure()
    plt.plot(x_xarxa, X_sol[-1,:],label = 'x (neuronal)',lw = 2)
    plt.plot(x_xarxa, Z_sol[-1,:], label = 'z $K^+$', lw = 2)
    plt.xlabel('Posició r')
    plt.ylabel('Valor')
    plt.title(f"Perfil de l'ona en t = {sol.t[-1]:.1f}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# EVOLUCIÓ TEMPORAL DE X PER A CADA NEURONA

plt.figure(figsize = (12,6))
norm = plt.Normalize(vmin = 0, vmax = Nx-1)
sm = plt.cm.ScalarMappable(cmap = 'viridis', norm  = norm)
sm.set_array([])

for i in range(Nx):
    plt.plot(sol.t, X_sol[:,i], lw = 0.8, color = plt.cm.viridis(norm(i)))

plt.xlabel('Temps')
plt.ylabel('X (potencial neuronal)')
plt.title('neurones')
plt.colorbar(sm, ax=plt.gca(), label='Índex de neurona')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
    
# CÀLCUL DE LA VELOCITAT DE PROPAGACIÓ

llindar = 0.0

index_inici = centre + mig + 2
index_fi = Nx -5
punts_estudi  = np.arange(index_inici, index_fi)

temps_activacio = []
posicions = []
  
for i in punts_estudi:
    idx_t = np.where(Z_sol[:,i]>llindar)[0]
    if len(idx_t) > 0:
        temps_activacio.append(sol.t[idx_t[0]])
        posicions.append(x_xarxa[i])
        
if len(temps_activacio) >1:
    coef = np.polyfit(temps_activacio, posicions, 1)
    velocitat = coef[0]*10
    vel = velocitat*60
    print("-" * 50)
    print(f"La velocitat estimada de propagació és: {vel:.5f} $cm^2/min$ [espai/temps]")
    print("-" * 50)
    
    plt.figure(figsize= (8,5))
    plt.plot(temps_activacio, posicions, 'o', label = " Front d'ona ")
    plt.plot(temps_activacio, np.polyval(coef, temps_activacio), '-', 
             label=f'Ajust lineal (v =  {vel:.5f} $cm^2/min$ )', color='red')
    plt.xlabel("Temps d'activació")
    plt.ylabel("Posició r")
    plt.legend()
    plt.grid(True)
    plt.savefig('velocitat_ona.png', dpi=150)
    plt.show()
    c = velocitat/((1.96**(-3))**(1/2))
    print(f"La velocitat adimensional de propagació és: c =  {c:.5f} $cm^2/min$ [espai/temps]")
else:
    print("No s'ha pogut detectar una propagació clara de l'ona amb aquest llindar.")
    