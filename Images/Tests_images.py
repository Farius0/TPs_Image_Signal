import numpy as np
import matplotlib.pyplot as plt
from Fonctions import *
from Fonctions2 import *
from Fonctions3 import *
from Fonctions4 import *
import torch
import deepinv as dinv
import cv2

################################################################ Chargement des images

Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")

Im_butterfly, Im_leaves, Im_starfish = Im_butterfly/255.0, Im_leaves/255.0, Im_starfish/255.0

Im_data, Im_parrot, Mask_parrot = load_img("cameraman.png"), load_img('Im1.png'), load_img('Im1_mask.png')

Im_data, Im_parrot, Mask_parrot = Im_data/255, Im_parrot / 255.0 , Mask_parrot / 255.0

##################################################################### Test Bruitage

Im_noised = add_gaussian_noise(Im_data, s= 0.1)

plt.imshow(Im_data, cmap='gray')

plt.imshow(Im_noised, cmap='gray')

# plt.savefig('images\Im_noised.png', bbox_inches='tight', pad_inches=0.4)

plt.imshow(forward_backward(Im_noised, 1/8, 1/8, 100), cmap ='gray')

# plt.savefig('images\Im_denoised_l1_FB.png', bbox_inches='tight', pad_inches=0.4)

plt.imshow(fista_tv(Im_noised, 1/8, 1/8, 50), cmap ='gray')

# plt.savefig("images\Im_denoised_TV_FISTA.png", bbox_inches='tight', pad_inches=0.4)

plt.imshow(Im_starfish)

Rx, Gx, Bx = cv2.split(Im_starfish)

Ry, Gy, By = add_gaussian_noise(Rx, 0.3), add_gaussian_noise(Gx, 0.3), add_gaussian_noise(Rx, 0.3)

Rz, Gz, Bz = fista_tv(Ry, 1/8, 1/8, 100), fista_tv(Gy, 1/8, 1/8, 100), fista_tv(By, 1/8, 1/8, 100)

plt.imshow(cv2.merge([Ry, Gy, By]))

plt.imshow(cv2.merge([Rz, Gz, Bz]))

R_, G_, B_ = cv2.split(Im_butterfly)

Rt, Gt, Bt = add_gaussian_noise(R_, 0.1), add_gaussian_noise(G_, 0.1), add_gaussian_noise(R_, 0.1)

Rs, Gs, Bs = fista_tv(Rt, 1/8, 1/8, 100), fista_tv(Gt, 1/8, 1/8, 100), fista_tv(Bt, 1/8, 1/8, 100)

plt.imshow(cv2.merge([Rt, Gt, Bt]))

plt.imshow(cv2.merge([Rs, Gs, Bs]))

######################################################################## Test Floutage

G = gaussian_kernel(15, sigma = 5)

plt.imshow(G)

Im_convolved = convolve(Im_data, G)

Im_convolved_noised = add_gaussian_noise(Im_convolved, 0.1)

plt.imshow(Im_convolved, cmap="gray")

plt.imshow(Im_convolved_noised, cmap="gray")

plt.imshow(fista_tv_3(Im_convolved, G, 0.0001, 0.0001, 70, 1e-7), cmap ='gray')

Butt_rx, G_rx = blur(numpy_to_tensor(R_), sigma = (5, 5), angle = 45)

Butt_gx, G_gx = blur(numpy_to_tensor(G_), sigma = (5, 5), angle = 45)

Butt_bx, G_bx = blur(numpy_to_tensor(B_), sigma = (5, 5), angle = 45)

plt.imshow(cv2.merge([Butt_rx, Butt_gx, Butt_bx]))

R_d, G_d, B_d = fista_tv_3(Butt_rx, G_rx, 0.0001, 0.0001, 70, 1e-7), fista_tv_3(Butt_gx, G_gx, 0.0001, 0.0001, 70, 1e-7), fista_tv_3(Butt_bx, G_bx, 0.0001, 0.0001, 70, 1e-7)

plt.imshow(cv2.merge([R_d, G_d, B_d]))

###################################################################### Test Inpainting

plt.imshow(Im_parrot, cmap="gray")

plt.imshow(ADMM_2(Mask_parrot, Im_parrot, 1, 0.03, 0.01, 60, 1e-7), cmap ='gray')

# mask = torch.rand(256, 256) > 0.4

mask = torch.ones_like(numpy_to_tensor(Rx))

# mask[:, :, 64:-64, 64:-64] = 0

mask[:,:, 0::24, :] = 0

mask[:,:, : , 0::24] = 0

plt.imshow(tensor_to_numpy(mask))

RM_x, M_x = inpaint(numpy_to_tensor(Rx), mask = mask , sigma=.05)

RM_y, M_y = inpaint(numpy_to_tensor(Gx), mask = mask, sigma=.05)

RM_z, M_z = inpaint(numpy_to_tensor(Bx), mask = mask, sigma=.05)

plt.imshow(cv2.merge([RM_x, RM_y, RM_z]))

R, G, B = ADMM_2(M_x, RM_x, 1, 0.03, 0.01, 50, 1e-7), ADMM_2(M_y, RM_y, 1, 0.03, 0.01, 50, 1e-7), ADMM_2(M_z, RM_z, 1, 0.03, 0.01, 50, 1e-7)

plt.imshow(cv2.merge([R, G, B]))

PSNR(Im_starfish[:,:,0], R)