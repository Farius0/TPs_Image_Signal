# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
from numpy.fft import fft, ifft



# Construction du signal u

# Déclarations des variables constantes 
N = 2048
k_2 = 152
k_1 = 11

# Espace du temps
t = np.arange(N)

 # Implémentation du signal u

def u(n) :
    signal = (n<N/2)*np.sin(2*np.pi*k_1*n/N) + (n>= N/2)*np.sin(2*np.pi*k_2*n/N)
    return  signal
u=u(t)

# Tracer de u
plt.figure(figsize=(10 ,6))
plt.plot(u)
plt.title('Signal de u(n)')
plt.xlabel('t')
plt.ylabel('Amplitude de u(n)')
plt.legend()
plt.grid(True)
plt.show()

# Calcul de la transformée de fourier de u

Fu = fft(u)

# Son spectre de phase est donné par: 
sp_u = np.abs(Fu)
ph_u = np.angle(Fu)


plt.figure(figsize = (10, 6))

plt.plot(sp_u,'b')
plt.title("spectre d'amplitude")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid()
plt.show()


# Calcul et affichage des spectres v1 et v2


n = np.arange(0,N)
v1 , v2 = np.array_split(u ,2) 

# Calcul des transformées de Fourier de v1 et v2
Fv1 = fft(v1)
Fv2 = fft(v2)

# Calcul des spectres de v1 et v2
sp_v1 = np.abs(Fv1)
sp_v2 = np.abs(Fv2)

# Répresentation de Fourier de v1 et de v2
plt.plot(Fv1.real, label = " partie réel")
plt.plot(Fv1.imag, label =" partie imaginaire")
plt.title('Fourier de v1')
plt.xlabel('t')
plt.ylabel('Amplitude de u(n)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(Fv2.imag, label =" partie imaginaire" )
plt.plot(Fv2.real, label =" partie réel" )
plt.title('Fourier de v2')
plt.xlabel('t')
plt.ylabel('Amplitude de u(n)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize = (10, 6))


plt.subplot(1, 2, 1)
plt.plot(sp_v1[:N//4],'r')
plt.title("spectre d'amplitude de v1")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid(True)


plt.subplot(1,2,2)
plt.plot(sp_v2[:N//4],'b')
plt.title("spectre d'amplitude de v2")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid(True)

plt.tight_layout()
plt.show()


g = np.where(np.arange(N)<N/2,1,0)

v1_tild = np.sin(2*np.pi*k_1*n/N)
v2_tild = np.sin(2*np.pi*k_2*n/N)

TFDv1_tild = fft(v1_tild)
TFDv2_tild = fft(v2_tild)

# Expression de u en fonction de v1_tild et v2_tild 

u_1 = g*v1_tild +(1-g)*v2_tild 

TFDg = fft(g)
sp_g = np.abs(TFDg)

plt.plot(g)
plt.title("spectre d'amplitude de g")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid(True)
plt.show()

plt.figure(figsize=(10 ,6))
plt.plot(u_1)
plt.title('Signal de u_1(n)')
plt.xlabel('t')
plt.ylabel('Amplitude de u_1(n)')
plt.legend()
plt.grid(True)
plt.show()

# Spectre d'amplitude de g 

plt.plot(sp_g,'b')
plt.title("spectre d'amplitude de g")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid(True)

# Recherche de Fourier de u en utilisant la convolution  circulaire

v1_filt = np.zeros(N,dtype = np.complex128)
v2_filt = np.zeros(N,dtype = np.complex128)

for i in range(N):
    for k in range(N):
        v1_filt[i] += TFDg[k]*TFDv1_tild[i-k]
        v2_filt[i] += (1-TFDg[k])*TFDv2_tild[i-k]

        
v1_filt =(1/N)*v1_filt
        
v2_filt =(1/N)*v2_filt

Fu_2 = v1_filt + v2_filt

plt.figure(figsize=(12, 6))
plt.plot(Fu_2.real, label="Partie réelle")
plt.plot(Fu_2.imag, label="Partie imaginaire") 
plt.title("TF_u en fonction de g, v1_tld et v2_tld")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude");
plt.grid(True)
plt.legend()
plt.show()



# Recherche de Fourier de v1 et v2 en utilisant la convolution  circulaire


v1_filt2 = np.zeros(N//2,dtype = np.complex128)
v2_filt2 = np.zeros(N//2,dtype = np.complex128)
g_red = fft(g[:N//2])

for i in range(N//2):
    for k in range(N//2):
        v1_filt2[i] += g_red[k]*Fv1[i-k]
        v2_filt2[i] += g_red[k]*Fv2[i-k]
        
Fv1_2 =(2/N)*v1_filt2
Fv2_2 =(2/N)*v2_filt2

Fv1_2_amp = np.abs(Fv1_2)
Fv2_2_amp = np.abs(Fv2_2)

plt.figure(figsize=(12, 6))
plt.plot(Fv1_2_amp[:N//4], label="Spectre d'amplitude de v1")
plt.title("Spectre d'amplitude de v1 et v2 avec filtre g")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(Fv2_2_amp[:N//4], label="Spectre d'amplitude de v2")
plt.title("Spectre d'amplitude de v1 et v2 avec filtre g")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# Définition du signal shirp

kv = k_1 + n*(k_2 - k_1)/N

w = np.sin(2*np.pi*n*kv/N)

TFDw =  fft(w)
sp_w = np.abs(TFDw)

plt.plot(w)
plt.title("Signal de chirp de w")
plt.xlabel('t')
plt.ylabel('Amplitudes')
plt.grid(True)
plt.show()

plt.plot(sp_w)
plt.title("Spectre du chirp w")
plt.xlabel('Fréquences')
plt.ylabel('Amplitudes')
plt.grid(True)
plt.show()


# Définir le filtre dérivateur h = 𝛿1 − 𝛿0


h = np.zeros(N)
h[0] = -1 
h[1] = 1 
 
# Générer un bruit e qui suit une distribution gaussienne
e = np.random.normal(loc=0, scale=0.1, size=N) 

# Calculer la transformée de Fourier de h pour obtenir H(omega)
TFDh = np.fft.fft(h, N)

# Observation du signal obtenu à la sortie
y_conv = np.zeros(N)
for i in range(N):
    for k in range(N):
        y_conv[i] += h[k]*w[i-k]
               
     
y = y_conv + e

# Calculer la transformée de Fourier de y en tenant compte du bruit e  
TFDy = fft(y)

# Calculer la transformée de Fourier du bruit e pour obtenir E(omega)
TFDe = fft(e)

# Estimer H(omega) à partir de Y(omega), W(omega) et E(omega)
TFDh = (TFDy - TFDe) / TFDw

# Calculer l'estimation de h en effectuant la transformée de Fourier inverse de H(omega)
h_estimé = ifft(TFDh)
print(h_estimé)

plt.plot(h_estimé)


def h_estim(k1, k2, e=e):
    kv = k_1 + n * (k_2 - k_1) / N
    w = np.sin(2 * np.pi * n * kv / N)
    y_conv = np.zeros(N)
    for i in range(N):
        for k in range(N):
            y_conv[i] += h[k]*w[i-k]
    y = y_conv + e  
    TFDy = fft(y)
    TFDe = fft(e)
    TFDh = (TFDy - TFDe) / TFDw
    h_estime = ifft(TFDh)
    return h_estime


plt.figure(figsize=(10,10), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(h_estim(1, 512, e=e))
plt.title("Estimation de h avec k1 = 1 et k2 = 512")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(h_estim(512, 1, e=e))
plt.title("Estimation de h avec k1 = 512 et k2 = 1")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(h_estim(512, 512, e=e))
plt.title("Estimation de h avec k1 = 512 et k2 = 512")
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")

plt.grid(True)
plt.tight_layout()
plt.show()


A = np.linalg.norm(h_estim(1, 512) - h)
print(A)
B = np.linalg.norm(h_estim(512, 1) - h)
print(B)
C = np.linalg.norm(h_estim(512, 512) - h)
print(C)

################################################################# Spectogramme

# Fonction pour calculer le spectrogramme
def spect(signal, L, M):
    # Initialisation de la matrice du spectrogramme
    spectrogram = np.zeros((L, M), dtype=np.complex128)
    
    # Calcul du spectrogramme
    for m in range(M):
        # Définition de la trame en fonction de m
        trame = signal[m * L: (m + 1) * L]
        # Calcul de la transformée de Fourier de la trame
        spectrogram[:, m] = np.fft.fft(trame)
    
    return np.abs(spectrogram)

# Définition des paramètres
N = 2048
M = 128
L = 16
k1 = 11  # Valeur de k1
k2_values = [11, 100,  1024, 1500, 2048 ]  # Valeurs de k2 à tester

# Génération du signal u en fonction de k1 et k2
def signal_u(N, k1, k2):
    n = np.arange(N)
    u = np.zeros(N)
    u[:N//2] = np.sin(2 * np.pi * k1 * n[:N//2] / N)
    u[N//2:] = np.sin(2 * np.pi * k2 * n[N//2:] / N)
    return u

# Génération du signal w en fonction de k1 et k2
def signal_w(N, k1, k2):
    n = np.arange(N)
    w = np.sin(2 * np.pi * n * (k1 + (k2 - k1) * np.arange(N) / N) / N)
    return w

# Affichage des spectrogrammes pour différentes valeurs de k2
for k2 in k2_values:
    # Génération du signal u en fonction de k1 et k2
    u = signal_u(N, k1, k2)
    
    # Génération du signal w en fonction de k1 et k2
    w = signal_w(N, k1, k2)
    
    # Calcul du spectrogramme pour u
    spectrogram_u = spect(u, L, M)
    
    # Calcul du spectrogramme pour w
    spectrogram_w = spect(w, L, M)
    
    # Affichage du spectrogramme de u
    plt.figure()
    plt.imshow(spectrogram_u, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Spectrogramme de u (k2={k2})")
    plt.xlabel("Trame")
    plt.ylabel("Fréquence")
    plt.colorbar(label='Amplitude')
    plt.show()
    
    # Affichage du spectrogramme de w
    plt.figure()
    plt.imshow(spectrogram_w, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Spectrogramme de w (k2={k2})")
    plt.xlabel("Trame")
    plt.ylabel("Fréquence")
    plt.colorbar(label='Amplitude')
    plt.show()


ML_values = [(1024, 2),(512,4), (64,32), (8,256), (4,512), (2, 1024)] # valeurs de M et L à tester
for  L, M  in ML_values:
    u = signal_u(N, k1, 152 )
    
    # Génération du signal w en fonction de k1 et k2
    w = signal_w(N, k1, 152)
    
    # Calcul du spectrogramme pour u
    spectrogram_u = spect(u, L, M)
    
   
    # Calcul du spectrogramme pour w
    spectrogram_w = spect(w, L, M)
    
    # Affichage du spectrogramme de u
    plt.figure()
    plt.imshow(spectrogram_u, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Spectrogramme de u (M,L={M,L})")
    plt.xlabel("Trame")
    plt.ylabel("Fréquence")
    plt.colorbar(label='Amplitude')
    plt.show()
    
     
     
def spectrogram2(u, L, M, r):
    T_u = np.zeros((L, M), dtype = np.complex128)
    for m in range(M):
        T_u[:,m] = fft(u[m*r:m*r+L])
    return np.abs(T_u)
    # Affichage du spectrogramme de w
    plt.figure()
    plt.imshow(spectrogram_w, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Spectrogramme de w (M,L={M,L})")
    plt.xlabel("Trame")
    plt.ylabel("Fréquence")
    plt.colorbar(label='Amplitude')
    plt.show()


Tu1 = spectrogram2(u, 16, 128, 8)
Tw1 = spectrogram2(w, 16, 128, 8)


plt.figure(figsize=(10, 4))
plt.imshow(Tu1, aspect='auto', origin='lower', cmap='jet')           
plt.title('Spectrogramme amélioré de u')
plt.xlabel('Temps (trames)')            
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.imshow(spectrogram_u, aspect='auto', origin='lower', cmap='jet')
plt.title('Spectrogramme de u')
plt.xlabel('Temps (trames)')
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')

plt.subplot(2, 1, 2)
plt.imshow(Tu1, aspect='auto', origin='lower', cmap='jet')           
plt.title('Spectrogramme amélioré de u')
plt.xlabel('Temps (trames)')            
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 4))
plt.imshow(Tw1, aspect='auto', origin='lower', cmap='viridis')           
plt.title('Spectrogramme amélioré de w')
plt.xlabel('Temps (trames)')            
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')
plt.show()



plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.imshow(spectrogram_w, aspect='auto', origin='lower', cmap='viridis')
plt.title('Spectrogramme de w')
plt.xlabel('Temps (trames)')
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')

plt.subplot(2, 1, 2)
plt.imshow(Tw1, aspect='auto', origin='lower', cmap='viridis')           
plt.title('Spectrogramme amélioré de w)')
plt.xlabel('Temps (trames)')            
plt.ylabel('Fréquence')
plt.colorbar(label='Amplitude')

plt.tight_layout()
plt.show()


def spectrogram3(u, L, M, r):
    T_u = np.zeros((L, M), dtype = np.complex128)
    for m in range(M):
        T_u[:,m] = fft(u[m*r:m*r+L])
    return T_u


def inv_spect(Tu, r):
    L,M = Tu.shape; N = L * M
    u = np.zeros((N), dtype = np.complex128)
    for m in range(M):
            u[m*r:m*r+L] += ifft(Tu[:,m])
    return u/(L/r)

Tu_2 = spectrogram3(u, 16, 128, 4)
u_final = inv_spect(Tu_2, 4)


# Afficher le signal u_final
plt.figure(figsize=(12,4), dpi=100)
plt.plot(u_final)
plt.title("Signal u_final")
plt.xlabel("Fréquences")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()










