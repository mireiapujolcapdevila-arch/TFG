import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

b = 3.0         
epsilon = 0.5    


def equacions_ordre0(vars):
    c0, xi1 = vars
    
    if c0 >= 2.0 or c0 <= 0:
        return [1e5, 1e5] 
        
    alpha_plus = (c0 + np.sqrt(c0**2 + 4)) / 2
    alpha = c0 / 2
    beta = np.sqrt(4 - c0**2) / 2
    gamma_minus = (c0 - np.sqrt(c0**2 + 12/(b-2))) / 2
    
    z0_xi1 = -np.exp(alpha*xi1)*np.cos(beta*xi1) + ((alpha_plus + alpha)/beta)*np.exp(alpha*xi1)*np.sin(beta*xi1) + 1
    w0_xi1 = alpha_plus*np.exp(alpha*xi1)*np.cos(beta*xi1) + (alpha*(alpha_plus + alpha)/beta + beta)*np.exp(alpha*xi1)*np.sin(beta*xi1)
    
    w0_reg3_0 = gamma_minus * (2 - b)
    
    eq1 = z0_xi1 - 2.0         
    eq2 = w0_xi1 - w0_reg3_0   
    return [eq1, eq2]

c0, xi1_star = fsolve(equacions_ordre0, [1.0, 2.0])
alpha_plus = (c0 + np.sqrt(c0**2 + 4)) / 2
alpha = c0 / 2
beta = np.sqrt(4 - c0**2) / 2
gamma_minus = (c0 - np.sqrt(c0**2 + 12/(b-2))) / 2

print(f"Solució ordre 0")
print(f"c0 = {c0:.5f}")
print(f"xi1* = {xi1_star:.5f}\n")
def equacions_ordre1(vars):
    c1, M1 = vars
    
    K1 = 0.0

    w1_reg1_0 = K1*alpha_plus + (c1*alpha_plus)/(2*alpha_plus - c0)
    

    D = (w1_reg1_0 - alpha*K1 + c1/2) / beta
    
    C2 = (c1/2) * (beta*alpha_plus + (alpha*(1 + alpha*alpha_plus))/beta)
    
    z1_xi1 = np.exp(alpha*xi1_star) * ((K1 - (c1/2)*xi1_star)*np.cos(beta*xi1_star) + (D + C2*xi1_star)*np.sin(beta*xi1_star))
    
    term_cos = alpha*K1 - c1/2 + beta*D + xi1_star*(beta*C2 - alpha*(c1/2))
    term_sin = alpha*D - beta*K1 + C2 + xi1_star*(alpha*C2 + beta*(c1/2))
    w1_xi1 = np.exp(alpha*xi1_star) * (term_cos*np.cos(beta*xi1_star) + term_sin*np.sin(beta*xi1_star))
    
    D_reg3 = (gamma_minus*(c1*(b-2) - 3*c0)) / np.sqrt(c0**2 + 12/(b-2))
    
    z1_reg3_0 = M1
    w1_reg3_0 = gamma_minus*M1 + D_reg3
    eq1 = z1_xi1 - z1_reg3_0
    eq2 = w1_xi1 - w1_reg3_0
    
    return [eq1, eq2]

c1, M1 = fsolve(equacions_ordre1, [0.0, 0.0])
print(f"Solució ordre 1")
print(f"c1 = {c1:.5f}")
print(f"M1 = {M1:.5f}\n")
print(f"Velocitat total c = {c0 + epsilon*c1:.5f}")

K1 = 0.0
w1_reg1_0 = K1*alpha_plus + (c1*alpha_plus)/(2*alpha_plus - c0)
D = (w1_reg1_0 - alpha*K1 + c1/2) / beta
C2 = (c1/2) * (beta*alpha_plus + (alpha*(1 + alpha*alpha_plus))/beta)
D_reg3 = (gamma_minus*(c1*(b-2) - 3*c0)) / np.sqrt(c0**2 + 12/(b-2))

xi_reg1 = np.linspace(-5, 0, 200)
xi_reg2 = np.linspace(0, xi1_star, 200)
xi_reg3_local = np.linspace(0, 5, 200)
xi_reg3_global = xi_reg3_local + xi1_star

Z = []
W = []
XI = []

# REGIÓ Z <0 
z0_1 = np.exp(alpha_plus*xi_reg1) - 1
z1_1 = K1*np.exp(alpha_plus*xi_reg1) + (c1*alpha_plus/(2*alpha_plus - c0))*xi_reg1*np.exp(alpha_plus*xi_reg1)
w0_1 = alpha_plus*np.exp(alpha_plus*xi_reg1)
w1_1 = K1*alpha_plus*np.exp(alpha_plus*xi_reg1) + (c1*alpha_plus/(2*alpha_plus - c0))*(1 + alpha_plus*xi_reg1)*np.exp(alpha_plus*xi_reg1)

Z.extend(z0_1 + epsilon*z1_1)
W.extend(w0_1 + epsilon*w1_1)
XI.extend(xi_reg1)

# REGIÓ 0<Z<2
z0_2 = -np.exp(alpha*xi_reg2)*np.cos(beta*xi_reg2) + ((alpha_plus + alpha)/beta)*np.exp(alpha*xi_reg2)*np.sin(beta*xi_reg2) + 1
z1_2 = np.exp(alpha*xi_reg2) * ((K1 - (c1/2)*xi_reg2)*np.cos(beta*xi_reg2) + (D + C2*xi_reg2)*np.sin(beta*xi_reg2))

w0_2 = alpha_plus*np.exp(alpha*xi_reg2)*np.cos(beta*xi_reg2) + (alpha*(alpha_plus + alpha)/beta + beta)*np.exp(alpha*xi_reg2)*np.sin(beta*xi_reg2)
term_cos = alpha*K1 - c1/2 + beta*D + xi_reg2*(beta*C2 - alpha*(c1/2))
term_sin = alpha*D - beta*K1 + C2 + xi_reg2*(alpha*C2 + beta*(c1/2))
w1_2 = np.exp(alpha*xi_reg2) * (term_cos*np.cos(beta*xi_reg2) + term_sin*np.sin(beta*xi_reg2))

Z.extend(z0_2 + epsilon*z1_2)
W.extend(w0_2 + epsilon*w1_2)
XI.extend(xi_reg2)

# REGIÓ Z > 2
z0_3 = (2 - b)*np.exp(gamma_minus*xi_reg3_local) + b
z1_3 = M1*np.exp(gamma_minus*xi_reg3_local) + D_reg3*xi_reg3_local*np.exp(gamma_minus*xi_reg3_local)

w0_3 = gamma_minus*(2 - b)*np.exp(gamma_minus*xi_reg3_local)
w1_3 = (gamma_minus*M1 + D_reg3*(1 + gamma_minus*xi_reg3_local))*np.exp(gamma_minus*xi_reg3_local)

Z.extend(z0_3 + epsilon*z1_3)
W.extend(w0_3 + epsilon*w1_3)
XI.extend(xi_reg3_global)

# GRÀFIC
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(XI, Z, label=r'$z(\xi)$', color='blue', lw=2)
plt.plot(XI, W, label=r'$w(\xi) = z^\prime(\xi)$', color='red', lw=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(xi1_star, color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', lw=0.5)
plt.axhline(2, color='gray', linestyle=':', alpha=0.8)
plt.xlabel(r'$\xi$')
plt.ylabel('Amplitud')
plt.title(r'Ona per a $\varepsilon = 0.5$ amb ordre $\mathcal{O}(\varepsilon^2)$')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(Z, W, color='purple', lw=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(2, color='gray', linestyle='--', alpha=0.5)
plt.xlabel(r'$z$')
plt.ylabel(r'$w$')
plt.title('Pla de Fases $(z, w)$')
plt.grid(True)

plt.tight_layout()
plt.show()
