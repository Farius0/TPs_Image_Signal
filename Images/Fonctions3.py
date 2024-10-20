from tqdm import tqdm
from Fonctions import *
import numpy as np
import itertools


# Equation de la chaleur

def heat_equation(f, dt, K):
    u = np.copy(f) 

    for k in range(K):
        u = u + dt * laplacian(u)
        
    return u

# Contours

def grad_edge(u, eta):
    ''' détecter les contours dans une image en utilisant le gradient de l'image et un seuil de sensibilité '''
    grad_u = gradient(u)
    norm_grad_u = norm(grad_u)
    
    contours = np.zeros_like(u, dtype=int)
    contours[norm_grad_u > eta] = 1
    
    return contours

def change_sign(J):
    ''' Détecte les changements de signe dans une image et génère une carte booléenne pour indiquer où ces changements se produisent '''
    bool_map = np.full_like(J, False, dtype=bool)
    
    prod_ver = (J[:-1, :] * J[1:, :]) <= 0
    diff_abs_ver = abs(J[:-1, :]) - abs(J[1:, :])
    
    prod_hor = (J[:, :-1] * J[:, 1:]) <= 0
    diff_abs_hor = abs(J[:, :-1]) - abs(J[:, 1:])
    
    bool_map[:-1, :] |= prod_ver & (diff_abs_ver <= 0)
    bool_map[1:, :] |= prod_ver & (diff_abs_ver > 0)
    
    bool_map[:, :-1] |= prod_hor & (diff_abs_hor <= 0)
    bool_map[:, 1:] |= prod_hor & (diff_abs_hor > 0)
    
    return bool_map

def lap_edge(u):
    ''' détecter les contours dans une image en utilisant le Laplacien de l'image '''
    laplacian_u = laplacian(u)
    
    contours = np.zeros_like(u, dtype=int)
    
    bool_map = change_sign(laplacian_u)
    contours[bool_map] = 1
    
    return contours

def Marr_Hildreth(u, eta):
    ''' Combine deux critères de détection de contours (changement de signe et Laplacien) pour détecter des contours dans une image '''
    cond1 = grad_edge(u, eta)
    cond2 = lap_edge(u)
    
    return cond1 & cond2  

def g_exp(xi, alpha=1):
    ''' effectuer une sorte de filtrage passe-bas, lissant l'image et réduisant le bruit de haute fréquence '''
    return np.exp(- (xi / alpha)**2)

def g_PM(xi, alpha=1):
    
    ''' effectuer une sorte de filtrage passe-haut, préservant les détails fins de l'image tout en atténuant les basses fréquences'''
    return 1 / np.sqrt((xi / alpha)**2 + 1)

def Perona_Malik(f, dt, K, alpha, g=g_PM):
    ''' Restauration d'images en utilisant la diffusion anisotrope '''
    u = np.copy(f)
    for k in range(K):
        grad_u = gradient(u)
        norm_grad_u = norm(grad_u)
        
        anisotropic_diffusion = div(g(norm_grad_u, alpha=alpha) * grad_u)
        u = u + dt * anisotropic_diffusion
        
    return u  

def search_K_alpha_opt(f, u, K_vals, alpha_vals):
    '''recherche exhaustive pour trouver les combinaisons optimales de valeurs pour les paramètres "K" (nombre d'itérations) et "alpha" (intensité de la diffusion) dans l'algorithme de diffusion anisotrope de Perona-Malik. '''
    dt = 1/8
    
    n_K = len(K_vals)
    n_alpha = len(alpha_vals)
    
    map_K_alpha_PSNR = np.empty((n_K, n_alpha))
    
    for n, (K, alpha) in tqdm(enumerate(itertools.product(K_vals, alpha_vals)), total=(n_K * n_alpha)):
        Im_rec_PM = Perona_Malik(f, dt, K, alpha)
        PSNR_Im_rec = PSNR(u, Im_rec_PM)
        
        i, j = divmod(n, n_alpha)
        map_K_alpha_PSNR[i, j] = PSNR_Im_rec
        
    return map_K_alpha_PSNR

def Perona_Malik_enhanced(f, dt, K, alpha, s, g=g_PM):
    u = np.copy(f)
    G = gaussian_kernel(dt, s)
    
    for k in range(K):        
        grad_u = gradient(u)
        
        conv = convolve(u, G)
        
        grad_conv = gradient(conv)
        norm_grad_conv = norm(grad_conv)
        
        anisotropic_diffusion = div(g(norm_grad_conv, alpha=alpha) * grad_u)
        u = u + dt * anisotropic_diffusion
        
    return u