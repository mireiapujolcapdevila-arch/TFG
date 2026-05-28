import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
b = 3


def lambda_positiu(c):
    return (1/2)*(c+np.sqrt(c**2-4))


def lambda_negatiu(c):
    return (1/2)*(c-np.sqrt(c**2-4))

def r_positiu(c):
    return (1/2)*(c+np.sqrt(c**2+4))


def gamma_negatiu(c):
    return (1/2)*(c-np.sqrt(c**2+12/(b-2)))

def c_2(c):
    lambda_pos = lambda_positiu(c)
    lambda_neg = lambda_negatiu(c)
    r_pos = r_positiu(c)
    return (r_pos+lambda_pos)/(lambda_neg-lambda_pos)

def c_1(c):
    lambda_pos = lambda_positiu(c)
    lambda_neg = lambda_negatiu(c)
    r_pos = r_positiu(c)
    return (-r_pos-lambda_neg)/(lambda_neg-lambda_pos)

def w(t,c):
    c1 = c_1(c) 
    c2 = c_2(c) 
    lambda_pos = lambda_positiu(c) 
    lambda_neg = lambda_negatiu(c)
    return c1*lambda_pos*np.exp(lambda_pos*t)+c2*lambda_neg*np.exp(lambda_neg*t)

def alpha(c):
    return c/2 
def beta(c):
    return np.sqrt(4-c**2)/2

def w_2(t,c):
    alp = alpha(c)
    bet = beta(c)
    r_pos = r_positiu(c)
    return alp*np.exp(alp*t)*(-np.cos(bet*t)+(r_pos+alp)/bet*np.sin(bet*t))+ np.exp(alp*t)*(bet*np.sin(bet*t)+(r_pos+alp)*np.cos(bet*t))
    
    
def buscar_t(t, c):
    c1 = c_1(c)
    c2 = c_2(c)
    lambda_pos = lambda_positiu(c)
    lambda_neg = lambda_negatiu(c)
    
    return c1*np.exp(lambda_pos*t)+c2*np.exp(lambda_neg*t)-1.0

def trobar_c(c):
    gamma_neg  = gamma_negatiu(c)
    alp = alpha(c)
    bet = beta(c)
    r_pos = r_positiu(c)
    def buscar_t(t):
        #c1 = c_1(c)
        #c2 = c_2(c)
        #lambda_pos = lambda_positiu(c)
        #lambda_neg = lambda_negatiu(c)
        
        return np.exp(alp*t)*(-np.cos(bet*t)+(r_pos+alp)/bet*np.sin(bet*t))-1.0
    t_adient = fsolve(buscar_t, 0.5)[0]
    
    return w_2(t_adient,c)-(2-b)*gamma_neg

c_adient = fsolve(trobar_c, 1.5)[0]
print(c_adient)

#DIBUIX DE LA SOLUCIÓ

c = c_adient
r_pos = r_positiu(c)
gamma_neg = gamma_negatiu(c)
alp = alpha(c)
bet = beta(c)

def F(x, t):
    return np.log(np.abs((1+x)/(1-x))) - 2/(1+x) - 2*t/c

# ---------- tram 1 ----------
t1 = np.linspace(-70, -65, 200)
x1 = -np.ones_like(t1)

# ---------- tram implícit ----------
t2 = np.linspace(-65, 8, 600)

x2 = []
x0 = -0.99  

for t in t2:
    sol = fsolve(F, x0=x0, args=(t,))
    
    # evitar sortir de (-1,1)
    x_sol = np.clip(sol[0], -0.999, 0.999)
    
    x2.append(x_sol)
    x0 = x_sol

x2 = np.array(x2)

# ---------- tram 3 ----------
t3 = np.linspace(8, 70, 200)
x3 = np.ones_like(t3)

# unir-ho tot
t = np.concatenate([t1, t2, t3])
x = np.concatenate([x1, x2, x3])

y = 2*np.ones_like(t)

# gràfica
plt.figure(figsize=(8,4))
plt.xlim(-9,9)
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.title("Perfil de l'ona viatgera durant el sistema lent")
plt.xlabel('temps ràpid')
plt.grid()
plt.legend()

plt.show()


#Busquem la t on la solució definida en z pertany (0,2), i mirem el temps quan aquesta solució assoleix el valor z = 2
def buscar_t_final(t):
    return np.exp(alp*t)*(-np.cos(bet*t)+(r_pos+alp)/bet*np.sin(bet*t))-1.0
t_final = fsolve(buscar_t_final, 0.5)[0]

 

t_esq = np.linspace(-5, 0, 200) #temps que agafem de la solució que està definida per z<0 fins z = 0
t_centre = np.linspace(0, t_final, 200) # temps que agafem la solució que està definida per z = 0 fins z = 2
t_dr = np.linspace(t_final, t_final + 5, 200) #temps que agafem la solució que està definida per z > 2

#ZONA ESQUERRA
z_esq = np.exp(r_pos*t_esq) - 1.0  #solució z definida en z < 0
w_esq = r_pos*np.exp(r_pos*t_esq) #solució w definida en z < 0
x_esq = -1 + 0*t_esq
y_esq = z_esq
#ZONA CENTRAL
z_cen = 1.0+np.exp(alp*t_centre)*(-np.cos(bet*t_centre)+(r_pos+alp)/bet*np.sin(bet*t_centre)) #solució z definida en 0 < z < 2
w_cen = w_2(t_centre,c) #solució w definida en 0 < z < 2
x_cen = -1 + 0*t_centre
y_cen = z_cen
#ZONA DRETA
z_eq_dret = b
z_dre = z_eq_dret +(2.0-z_eq_dret)*(np.exp(gamma_neg*(t_dr-t_final))) #solució z definida en z > 2
w_dre = (2.0-z_eq_dret)*(gamma_neg)*np.exp(gamma_neg*(t_dr-t_final))  #solució w definida en z > 2
x_dre = 1 + 0*t_dr
y_dre = z_dre

t_total = np.concatenate((t_esq, t_centre, t_dr))
z_total = np.concatenate((z_esq, z_cen, z_dre))
w_total = np.concatenate((w_esq, w_cen, w_dre))
x_total = np.concatenate((x_esq, x_cen, x_dre))
y_total = np.concatenate((y_esq, y_cen, y_dre))

#GRÀFIQUES

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

#--- GRÀFICA 1: Perfil de l'ona (Temps vs z i w) ---
ax1.plot(t_total, z_total, label='$z(t)$', color='blue', linewidth=2)
ax1.plot(t_total, w_total, label="$w(t) = z'(t)$", color='red', linestyle='--', linewidth=2)
ax1.plot(t_total, x_total, label='$x(t)$', color='orange', linewidth=2)
ax1.plot(t_total, y_total, label='$y(t)$', color='green', linewidth=2)

# Marquem els punts d'empalmament
ax1.axvline(0, color='gray', linestyle=':', alpha=0.7)
ax1.axvline(t_final, color='gray', linestyle=':', alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axhline(2, color='black', linewidth=0.5, linestyle=':')

ax1.set_title("Perfil de l'Ona Viatgera durant el sistema ràpid")
ax1.set_xlabel("Temps Lent ")
ax1.legend()
ax1.grid(True)
# --- GRÀFICA 2: Pla de fases (z vs w) ---
def z_esq_inici(t,c):
    return np.exp(r_pos*t) - 1.0
def w_esq_inici(t,c):
    return r_pos*np.exp(r_pos*t)
def w_dre_final(t,c):
    return (2.0-b)*(gamma_neg)*np.exp(gamma_neg*(t-t_final)) 
def z_dre_final(t,c):
    return  z_eq_dret +(2.0-z_eq_dret)*(np.exp(gamma_neg*(t-t_final)))
ax2.plot(z_total, w_total, color='purple', linewidth=2)
# Marquem els punts d'equilibri i de transició
ax2.plot(z_esq_inici(-5,c), w_esq_inici(-5,0), 'go', label="Equilibri Inici (-1,0)")
ax2.plot(2, w_2(t_final, c), 'bo', label="Empalmament a $z=2$")
ax2.plot(z_dre_final(t_final+5, c), w_dre_final(t_final + 5,c), 'ro', label=f"Equilibri Final ({z_eq_dret:.1f}, 0)")
# Dibuixem les línies de frontera z=0 i z=2
ax2.axvline(0, color='gray', linestyle=':')
ax2.axvline(2, color='gray', linestyle=':')
ax2.axhline(0, color='black', linewidth=0.5)

ax2.set_title("Pla de Fases $(z, w)$")
ax2.set_xlabel("$z$")
ax2.set_ylabel("$w$")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

