from PIL import Image
import numpy as np
import deepinv as dinv
import torch


def load_img(fname, dirname='images/'):
    Im_PIL = Image.open(dirname + fname)
    return np.array(Im_PIL)

def scalar_product(u, v):
    if u.ndim > 2:
        return np.sum(u * v, axis=0)
    else:
        return np.sum(u * v)
    
# Norme de u

def norm(u):
    return np.sqrt(scalar_product(u, u))

# gradient de u

def gradient(u):
    m, n = u.shape
    grad_u = np.zeros((2, m, n))
    
    grad_u[0, :-1, :] = u[1:] - u[:-1]
    
    grad_u[1, :, :-1] = u[:, 1:] - u[:, :-1]
    
    return grad_u

# divergence de u

def div(p):
    m, n = p.shape[1:]
    
    div_1 = np.zeros((m, n))
    div_1[:-1, :] = p[0, :-1, :]
    div_1[1:, :] -= p[0, :-1, :]
    
    div_2 = np.zeros((m, n))
    div_2[:, :-1] = p[1, :, :-1]
    div_2[:, 1:] -= p[1, :, :-1]
    
    return div_1 + div_2

# laplacian de u

def laplacian(u):
    return div(gradient(u))

# critère de convergence

def convergence_criteria(u0, u1, conv_crit):
    if (norm(u0 - u1)/norm(u0)< conv_crit):
        return True
    else :
        return False
    
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
        u_denoised = Denoise_TV(f, K, lambd, eps, tau = 1.0 / (lambd + 4.0))
        psnr = PSNR(u, u_denoised)
        psnr_values.append(psnr)
    optimal_lambda = lambda_values[np.argmax(psnr_values)]
    return optimal_lambda

def numpy_to_tensor(im_col):
   
    if im_col.ndim == 2:  # Image en niveaux de gris
        return torch.from_numpy(im_col).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        return torch.from_numpy(im_col).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

def tensor_to_numpy(im_tsr):

    if im_tsr.size(1) == 1:  # Image en niveaux de gris
        return im_tsr.squeeze(0).squeeze(0).numpy()  # (H, W)
    else:
        return im_tsr.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)


def denoise(img_tensor, sigma = .8):
 
    noise = dinv.physics.Denoising()
                                                                    
    noise.noise_model = dinv.physics.GaussianNoise(sigma = sigma)

    return noise(img_tensor)

def blur(img_tensor, sigma = (2, 2), angle = 45):

    Filt = dinv.physics.blur.gaussian_blur(sigma = sigma, angle = angle)

    Flou_oper = dinv.physics.Blur(Filt)

    return tensor_to_numpy(Flou_oper(img_tensor.float())), tensor_to_numpy(Filt)

# load an inpainting operator that masks 50% of the pixels and adds Gaussian noise

def inpaint(img_tensor, mask = torch.rand(256, 256) > 0.4, sigma=.05):

    Inpaint = dinv.physics.Inpainting(mask = mask, tensor_size= img_tensor.shape[1:],
                    noise_model=dinv.physics.GaussianNoise(sigma=sigma))

    return tensor_to_numpy(Inpaint(img_tensor)), tensor_to_numpy(mask)