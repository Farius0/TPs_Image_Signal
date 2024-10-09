
from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def scalar_product(u, v):
    if u.ndim > 2:
        return np.sum(u * v, axis=0)
    else:
        return np.sum(u * v)
    
# Norme

def norm(u):
    return np.sqrt(scalar_product(u, u))

# gradient de u

def gradient(u):
    m, n = u.shape
    grad_u = np.zeros((2, m, n))
    
    grad_u[0, :-1, :] = u[1:] - u[:-1]
    
    grad_u[1, :, :-1] = u[:, 1:] - u[:, :-1]
    
    return grad_u


def div(p):
    m, n = p.shape[1:]
    
    div_1 = np.zeros((m, n))
    div_1[:-1, :] = p[0, :-1, :]
    div_1[1:, :] -= p[0, :-1, :]
    
    div_2 = np.zeros((m, n))
    div_2[:, :-1] = p[1, :, :-1]
    div_2[:, 1:] -= p[1, :, :-1]
    
    return div_1 + div_2


def laplacian(u):
    return div(gradient(u))


def convergence_criteria(u0, u1, conv_crit):
    if (norm(u0 - u1)/norm(u0)< conv_crit):
        return True
    else :
        return False


def load_img(fname, dirname='images/'):
    Im_PIL = Image.open(dirname + fname)
    return np.array(Im_PIL)


def add_gaussian_noise(I, s):
    m, n = I.shape
    
    I_out = I + s * np.random.randn(m, n)
    
    return I_out

def MSE(u_truth, u_estim):
    m, n = u_truth.shape
    return np.sum((u_truth - u_estim)**2) / (m * n)

def PSNR(u_truth, u_estim):
    mse = MSE(u_truth, u_estim)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def search_lambda_opt(f, u, lambda_values, K, eps):
    '''Recherche exhaustive pour trouver la valeur optimale de lambda (λ) dans une fonction de débruitage.'''
    psnr_values = []

    for lambd in lambda_values:
        # Appliquez votre fonction de débruitage avec la valeur de lambda actuelle
        u_denoised = Denoise_TV(f, K, lambd, eps, tau = 1.0 / (lambd + 4.0))

        # Calculez le PSNR entre l'image débruitée et l'image de référence u
        psnr = PSNR(u, u_denoised)
        psnr_values.append(psnr)

    # Trouvez la valeur optimale de lambda
    optimal_lambda = lambda_values[np.argmax(psnr_values)]

    return optimal_lambda

def gaussian_kernel(size, sigma):
    variance = sigma**2
    half_size = size // 2
    
    range_matrix = np.arange(- half_size, half_size + 1)
    X, Y = np.meshgrid(range_matrix, range_matrix)
    
    return np.exp(- (X**2 + Y**2) / (2 * variance)) / (2 * np.pi * variance)

def convolve(f, G):
    return convolve2d(f, G, mode='same', boundary='symm')

def Denoise_Tikhonov(f, K, lamb, tau=None):
    if tau is None :
        tau = 1/(lamb +4)
    u_K =  np.copy(f)
    for i in range(1, K+1):
        u_K1 = u_K + tau*(lamb*(f-u_K) + laplacian(u_K))
        u_K = u_K1
    return u_K1

def Denoise_TV(f, K, lamb, eps, tau):
    tau = 1/(lamb +4)
    u_K = np.copy(f)
    for i in range(1,K+1):
        u_K1 = u_K + tau*(lamb*(f-u_K)+div(gradient(u_K)/np.sqrt(norm(gradient(u_K))**2 + eps)))
        u_K=u_K1
    return u_K1

def Denoise_Tikhonov_Fourier(f, lamb):
    x = np.fft.fft2(np.copy(f))
    y = np.copy(x)
    m,n = f.shape
    for p in range(m):
        for q in range(n):
            y[p,q] = lamb*x[p,q]/(lamb + 4*(np.sin(np.pi*p/m)**2 + np.sin(np.pi*q/n)**2))
    u= np.fft.ifft2(y)
    return np.real(u)

def Deconvolution_TV(f, G, tau, eps, K, lamb):
    u_K =np.copy(f)
    for i in range(1,K):
        u_K1 = u_K + tau*(lamb*convolve(f-convolve(u_K,G),G)+ div(gradient(u_K)/np.sqrt(norm(gradient(u_K))**2 + eps**2)))
        u_K = u_K1
    return u_K1

def Inpainting_TV(f, M, tau, eps, K, lamb):
    u_K = np.copy(f)
    for i in range(K):
        u_K1 = u_K + tau*(lamb*(f-u_K)*M + div(gradient(u_K)/np.sqrt(norm(gradient(u_K))**2 + eps**2)))
        u_K = u_K1
    return u_K1

def Inpainting_Tikhonov(f, M, tau, K, lamb):
    uk = np.copy(f)
    for i in range(K):
        uK_1 = uk + tau*(lamb*(f-uk)*M + laplacian(uk))
        uk = uK_1
    return uK_1

def Denoise_g1(f, K, lamb, eps, tau):
    if tau is None:
        tau=1/(lamb+4)
    u_k = np.copy(f)
    u = u_k
    for i in range(K):
        u = u_k + tau * (lamb * (f - u_k) + 2*u_k/((1 + u_k**2)**2))
        u_k = u 
    return u

def Denoise_g2(f, K, lamb, eps, tau):
    if tau is None:
        tau=1/(lamb+4)
    u_k = np.copy(f)
    u = u_k
    for i in range(K):
        u = u_k + tau * (lamb * (f - u_k) + 2*u_k/(1 + u_k**2))
        u_k = u 
    return u