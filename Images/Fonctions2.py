import numpy as np
from scipy.signal import convolve2d
from Fonctions import *

# Gaussian noise

def add_gaussian_noise(I, s):
    m, n = I.shape
    I_out = I + s * np.random.randn(m, n)
    return I_out

# Gaussian kernel 

def gaussian_kernel(size, sigma):
    variance = sigma**2
    half_size = size // 2
    range_matrix = np.arange(- half_size, half_size + 1)
    X, Y = np.meshgrid(range_matrix, range_matrix)
    return np.exp(- (X**2 + Y**2) / (2 * variance)) / (2 * np.pi * variance)

# Convolution

def convolve(f, G):
    return convolve2d(f, G, mode='same', boundary='symm')


# Denoise_Tikhonov

def Denoise_Tikhonov(f, K, lamb, tau=None):
    if tau is None :
        tau = 1/(lamb +4)
    u =  np.copy(f)
    for i in range(1, K+1):
        u = u + tau*(lamb*(f-u) + laplacian(u))
    return u

# Denoise_TV

def Denoise_TV(f, K, lamb, eps, tau):
    tau = 1/(lamb +4)
    u = np.copy(f)
    for i in range(1,K+1):
        u = u + tau*(lamb*(f-u)+div(gradient(u)/np.sqrt(norm(gradient(u))**2 + eps)))
    return u

# Denoise_Tikhonov_Fourier

def Denoise_Tikhonov_Fourier(f, lamb):
    x = np.fft.fft2(np.copy(f))
    y = np.copy(x)
    m, n = f.shape
    for p in range(m):
        for q in range(n):
            y[p,q] = lamb*x[p,q]/(lamb + 4*(np.sin(np.pi*p/m)**2 + np.sin(np.pi*q/n)**2))
    u= np.fft.ifft2(y)
    return np.real(u)

# Deconvolution_TV

def Deconvolution_TV(f, G, tau, eps, K, lamb):
    u =np.copy(f)
    for i in range(1,K):
        u = u + tau*(lamb*convolve(f-convolve(u,G),G)+ div(gradient(u)/np.sqrt(norm(gradient(u))**2 + eps**2)))
    return u

# Impainting_TV

def Inpainting_TV(f, M, tau, eps, K, lamb):
    u = np.copy(f)
    for i in range(K):
        u = u + tau*(lamb*(f-u)*M + div(gradient(u)/np.sqrt(norm(gradient(u))**2 + eps**2)))
    return u

# Impainting_Tikhonov

def Inpainting_Tikhonov(f, M, tau, K, lamb):
    u = np.copy(f)
    for i in range(K):
        u = u + tau*(lamb*(f-u)*M + laplacian(u))
    return u

# Denoise_g1

def Denoise_g1(f, K, lamb,eps, tau):
    if tau is None:
        tau=1/(lamb+4)
    u = np.copy(f)
    for i in range(K):
        div_grad = div(2 * gradient(u) * norm(gradient(u)) / ((eps + norm(gradient(u))**2)**2))
        u = u + tau * (lamb * (f - u) + div_grad)
    return u

# Denoise_g2

def Denoise_g2(f, K, lamb,eps, tau):
    if tau is None:
        tau=1/(lamb+4)
    u = np.copy(f)
    for i in range(K):
        div_grad = div(2 * gradient(u) * norm(gradient(u)) / (eps + norm(gradient(u))**2))
        u = u + tau * (lamb * (f - u) + div_grad)
    return u