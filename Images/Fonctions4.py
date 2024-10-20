import numpy as np
import matplotlib.pyplot as plt
from Fonctions import *
from Fonctions2 import *
from Fonctions3 import *
from numpy.fft import fft2, ifft2

# prox(u) = argmin_F(x) = (1/2*lambd)*||x-u||^2 + g(x)
# h = lambd

# Opérateur proximal pour g(x) = |x| (soft-thresholding)
def prox_l1(u, lambd):
    return np.sign(u) * np.maximum(np.abs(u) - lambd, 0)

# Opérateur proximal pour g(x) = |x|^2
def prox_l2(u, lambd):
    return u / (1 + 2 * lambd)

# Opérateur proximal pour g(x) = ||x-f||^2
def prox_l3(u, f, lambd):
    return (u + lambd * f) / (1 + lambd)

# Opérateur proximal pour g(x) = ||Ax-f||^2
def prox_l4(u, A, f, lambd):
    # Résolution du système (I + h * A^T * A)x = u + lambd * A^T * f
    I = np.eye(A.shape[1])
    J = I + lambd * A.T @ A
    K = u + lambd * A.T @ f
    return np.linalg.solve(J, K)

# Opérateur proximal pour g(u) = indicatrice de C (projection sur C)
def prox_l5(u, C=np.array([-1,1])):
    return np.clip(u, C[0], C[1])  # Projection sur l'intervalle [C[0], C[1]] 

# Opérateur proximal pour g(x) = ||grad(x)||_1
def prox_l6(u, lambd, tau, K):

    z = np.zeros((2, *u.shape))  # Initialisation de z (variable duale)
    
    for k in range(K):
        # projection sur ||z||∞ ≤ 1
        grad_z = -2 * gradient(div(z) + u / lambd)
        z = prox_l5(z - tau * grad_z)
    
    return u + lambd* div(z)

def prox_l7(u, lambd):
    return u + lambd * laplacian(u) # Equation de la chaleur 

def prox_l8(u, lambd): # perona malick

    grad_u = gradient(u)
    norm_grad_u = norm(grad_u)
    anisotropic_diffusion = div(g_PM(norm_grad_u) * grad_u)
    u = u + lambd * anisotropic_diffusion
    return u

def forward_backward(u, lambd, tau, K, tol = 1e-7):

    # Initialisation
    x = np.zeros_like(u)
    
    for k in range(K):
    
        grad_f = (x - u)/lambd
        x_half = x - tau * grad_f  # Descente de gradient
        
        # Proxy
        x = prox_l1(x_half, lambd)

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def forward_backward_2(u, A, lambd, tau, K, tol = 1e-7):

    x = np.zeros_like(u)
    
    for k in range(K):
        grad_f = ((x-u)*A) /lambd
        x_half = x - tau * grad_f  # Descente de gradient
        
        #Proxy
        x = prox_l1(x_half, lambd)

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def forward_backward_3(u, A, lambd, tau, K, tol = 1e-7):

    x = np.zeros_like(u)
    
    for k in range(K):
        grad_f = convolve(convolve(x, A)-u, A.T) /lambd
        x_half = x - tau * grad_f  # Descente de gradient
        
        #Proxy
        x = prox_l1(x_half, lambd)
    
        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break    
    
    return x

def fista_tv(u, lambd, tau, K, tol = 1e-7):
    
    y = np.copy(u)
    x_old = np.copy(u)
    t = 1  # Paramètre d'accélération
    
    for k in range(K):
        
        #Descente de gradient
        grad_f = (y - u)/lambd
        x_half = y - tau * grad_f

        # Prox_tv
        x = prox_l6(x_half, lambd, tau, K)
        
        # Accélération
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def fista_tv_2(u, A, lambd, tau, K, tol = 1e-7):
    
    y = np.copy(u)
    x_old = np.copy(u)
    t = 1  # Paramètre d'accélération
    
    for k in range(K):
        
        # Forward step: Gradient descent sur (1/2*lambd)||Ay - u||^2
        grad_f = ((y-u)*A) /lambd
        x_half = y - tau * grad_f

        # Backward step: TV prox (soft-thresholding)
        x = prox_l6(x_half, lambd, tau, K)
        
        # Mise à jour du paramètre d'inertie
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break        
    
    return x

def fista_tv_3(u, A, lambd, tau, K, tol = 1e-7):

    y = np.copy(u)
    x_old = np.copy(u)
    t = 1
    
    for k in range(K):
        
        # Descente de gradient
        grad_f = convolve(convolve(y, A)-u, A.T) /lambd
        x_half = y - tau * grad_f

        # Prox_TV
        x = prox_l6(x_half, lambd, tau, K)
        
        # Mise à jour du paramètre d'inertie
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x
    
        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
        
    return x

def PGM(u, lambd, tau, K, tol = 1e-7):

    x = np.zeros_like(u)
    
    for k in range(K):
        
        # Descente de gradient
        grad_f = (x - u)/lambd
        x_half = x - tau * grad_f
        
        # Prox
        x = prox_l7(x_half, tau)
        
        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
     
    return x

def PGM_2(u, A,  lambd, tau, K, tol = 1e-7):

    x = np.zeros_like(u)
    
    for k in range(K):
        
        # Descente de gradient
        grad_f = ((x-u)*A) /lambd
        x_half = x - tau * grad_f 
        
        # Prox
        x = prox_l7(x_half, tau)

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def PGM_3(u, A,  lambd, tau, K, tol = 1e-7):

    x = np.zeros_like(u)
    
    for k in range(K):
        
        # Descente de gradient
        grad_f = convolve(convolve(x, A)-u, A.T) /lambd
        x_half = x - tau * grad_f 
        
        # Prox
        x = prox_l7(x_half, tau)

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def APGM(u, lambd, tau, K, tol = 1e-7):
    
    x_old = np.copy(u)
    y = np.copy(u)
    t = 1
    
    for k in range(K):

        # Descente de Gradient
        grad_f = (y - u) /lambd
        x_half = y - tau * grad_f

        # Prox
        x = prox_l8(x_half, tau)
        
        # Accélération
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x
        
        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x  
    
def APGM_2(u, A, lambd, tau, K, tol = 1e-7):

    x_old = np.copy(u)
    y = np.copy(u)
    t = 1
    
    for k in range(K):

        # Descente de Gradient
        grad_f = ((y-u) * A) /lambd
        x_half = y - tau * grad_f

        # Prox
        x = prox_l8(x_half, tau)
        
        # Accélération
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x
        
        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break

    return x    


def APGM_3(u, A, lambd, tau, K, tol = 1e-7):

    x_old = np.copy(u)
    y = np.copy(u)
    t = 1
    
    for k in range(K):

        # Descente de Gradient
        grad_f = convolve(convolve(y, A)- u, A.T) /lambd
        x_half = y - tau * grad_f

        # Prox
        x = prox_l8(x_half, tau)
        
        # Accélération
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = x

        # Critère de convergence
        if np.linalg.norm(x - x_half) < tol:
            break
    
    return x

def primal_dual_algorithm(u, lambd, tau, sigma, K, theta=1.0, tol = 1e-7):

    x = np.copy(u)  # Variable primale
    y = np.zeros_like(u)  # Variable duale
    x_bar = np.copy(x)  # Variable relaxée

    for k in range(K):
        # y
        term = y + sigma * x_bar
        y = term - sigma * prox_l1(term / sigma, 1/ sigma)  # prox_l1
        
        # x
        x_new = prox_l6(y, lambd, tau, K) # Proximal de G
        
        # Relaxation
        x_bar = x_new + theta * (x_new - x)  # Relaxation avec theta

        # Mise à jour
        x = np.copy(x_new)
    
        # Critère de convergence
        if np.linalg.norm(x - y) < tol:
            break
    
    return x

def ADMM(u, lambd, rho, K=100, tol=1e-4):

    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)

    for k in range(K):
        # Mise à jour de x
        x = (u + rho * (y - z / rho)) / (1 + rho)
        
        # Mise à jour de y (soft-thresholding)
        y = prox_l1(x - z / rho, lambd / rho)
        
        # Mise à jour de la variable duale z
        z = z + rho * (x - y)
        
        # Vérification du critère de convergence
        if np.linalg.norm(x - y) < tol:
            break

    return z

def ADMM_2(A, u, lambd, tau, rho=1.0, K=100, tol=1e-7):
    
    m, n = u.shape
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)


    for k in range(K):
        # Mise à jour de y
        y = (A * u + rho * x + z) / (A + rho)
        
        # Mise à jour de x (soft-thresholding)
        x = prox_l6(y - z / rho, lambd / rho, tau, K)
        
        # Mise à jour de la variable duale z
        z = z + rho * (x - y)
        
        # Critère de convergence
        if np.linalg.norm(x - y) < tol:
            break

    return x
       

def ADMM_3(A, u, lambd, tau, rho=1.0, K=100, tol=1e-7):

    m, n = u.shape
    x = np.zeros_like(u)
    y = np.copy(u)
    z = np.zeros_like(u)

    for k in range(K):
        # Mise à jour de y
        y = y - tau * (convolve(convolve(y, A) - u, A.T) +  rho * (y - x - z/rho))
        
        # Mise à jour de x (soft-thresholding)
        x = prox_l6(y - z / rho, lambd / rho, tau, K)
        
        # Mise à jour de la variable duale z
        z = z + rho * (x - y)
        
        # Critère de convergence
        if np.linalg.norm(x - y) < tol:
            break
        
    return x