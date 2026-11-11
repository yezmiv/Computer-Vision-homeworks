import cv2
import numpy as np
import os

#degradation parameters
seed  = 42     #seme fixed 
alpha = 0.6 #contrast control if is <1 contrast is reduced
beta = 0    #brightness control
sigma = 15     #gaussian noise 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True) #create the images directory 
os.makedirs(os.path.join(BASE_DIR, "degraded"), exist_ok=True) #create the degraded images directory
os.makedirs(os.path.join(BASE_DIR, "confronto"), exist_ok=True) #create the confronto images directory

images = ["parigi.png", "cavallo.png", "mare.png"]
#img = cv2.imread("images/parigi.png", 1) 
#print("Image shape:", img.shape)


#functions to compute metrics
def mse(a, b):
    return np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) #mean squared error

def mae(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) #mean absolute error

def resize_h(img, h=500): #i want to resize keeping aspect ratio for the panel
    f = h / img.shape[0] 
    return cv2.resize(img, (int(img.shape[1] * f), h))

def label(img, testo, h=500): #resize and white text 
    img = resize_h(img, h)
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)  #just black bar
    cv2.putText(bar, testo, (5, 22), #write label
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
    return np.vstack([bar, img])


righe_csv = ["image,method,PSNR,MSE,MAE"]  #header for the CSV file  



for nome_img in images:
    img = cv2.imread("images/" + nome_img, 1)
    base = os.path.splitext(nome_img)[0]
    print("\nImage shape:", img.shape, "-", nome_img)
    #print("\n", nome_img)

    #DEGRADATION: gaussian noise and low contrast
    img_degraded = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  #constrast reduction
    rng = np.random.default_rng(seed) #reproducible random generator
    rumore = rng.normal(0, sigma, img.shape) #noise gaussiano
    img_degraded = np.clip(img_degraded.astype(np.float32) + rumore, 0, 255).astype(np.uint8) #add noise and clip to [0,255]
   
    cv2.imwrite(os.path.join(BASE_DIR, "degraded", f"{base}_degraded.png"), img_degraded)

    #METHODS: 1-baseline, 2-linear contrast stretching, 3-histogram equalization, 4-denoising + contrast recovery
   
    #1 
    img_baseline = img_degraded.copy() #just copy
    cv2.imwrite(f"results/{base}_method1_baseline.png", img_baseline)

    
    #2 linear contrast stretching

    #2A on LUMINANCE 
    lab = cv2.cvtColor(img_degraded, cv2.COLOR_BGR2LAB) #covert to LAB color space to isolate luminance
    L, a, b = cv2.split(lab)
    L_stretched = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX) #stretching on L channel for min-max 
    img_luminance = cv2.cvtColor(cv2.merge((L_stretched, a, b)), cv2.COLOR_LAB2BGR) #back to BGR original color 

    cv2.imwrite(f"results/{base}_method2A_luminance.png", img_luminance)
    #cv2.imshow("Method 2A Luminance", img_luminance)


    #2B Per-Channel BGR
    B, G, R = cv2.split(img_degraded) #divifo l'immagine degradata nei suoi 3 canali nativi

    #stretching indipendent for all channels
    blue = cv2.normalize(B, None, 0, 255, cv2.NORM_MINMAX)
    green = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX)
    red = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)

    img_channel = cv2.merge((blue, green, red)) #recombine the channels back into a single image

    cv2.imwrite(f"results/{base}_method2B_channel.png", img_channel)
    #cv2.imshow("Method 2B Per Channel", img_channel)



    #3 Histogram Equalization (on luminance channel)
    
    lab_eq = cv2.cvtColor(img_degraded, cv2.COLOR_BGR2LAB)
    L_eq, a_eq, b_eq = cv2.split(lab_eq)
    L_equalized = cv2.equalizeHist(L_eq) #equalization on L channel only
    img_equalized = cv2.cvtColor(cv2.merge((L_equalized, a_eq, b_eq)), cv2.COLOR_LAB2BGR) 

    cv2.imwrite(f"results/{base}_method3_histogram_equalization.png", img_equalized)
    #cv2.imshow("Method3 Histogram(LAB)", img_equalized)

    #4 Denoising + Contrast Recovery
    #4A BILATERAL DENOISE + LUMINANCE STRETCHING
   
    img_denoised = cv2.bilateralFilter(img_degraded, d=10, sigmaColor=80, sigmaSpace=80) #Denoising with bilateral filter

    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_stretched = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    img_bilateral_stretch = cv2.cvtColor(cv2.merge((L_stretched, a, b)), cv2.COLOR_LAB2BGR)

    cv2.imwrite(f"results/{base}_method4A_bilateral_stretch.png", img_bilateral_stretch)
    #cv2.imshow("Method 4A - Bilateral + Stretching", img_bilateral_stretch)

    #4B NON-LOCAL-MEANS DENOISING + LUMINANCE STRETCHING
    img_nlm = cv2.fastNlMeansDenoisingColored( img_degraded, None,
        h=13,          #filtro sul luminanza
        hColor=13,     #sul colori
        templateWindowSize=7, searchWindowSize=21 )

    lab = cv2.cvtColor(img_nlm, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_stretched = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)

    img_nlm_stretch = cv2.cvtColor(cv2.merge((L_stretched, a, b)), cv2.COLOR_LAB2BGR)
    cv2.imwrite(f"results/{base}_method4B_nlm_stretch.png", img_nlm_stretch)
    #cv2.imshow("Method 4B - NLM + Stretching", img_nlm_stretch)

    
    #EVALUATION
    # --- Dizionario: nome del metodo -> immagine risultante ---
    metodi = {
        "1_baseline": img_baseline,
        "2A_luminance": img_luminance,
        "2B_channel": img_channel,
        "3_histogram_eq": img_equalized,
        "4A_bilateral_stretch": img_bilateral_stretch,
        "4B_nlm_stretch": img_nlm_stretch,
    }


    print("\nMetodo                  PSNR      MSE      MAE")
    for nome, risultato in metodi.items():
        # Le metriche si calcolano SEMPRE rispetto all'originale (img)
        psnr = cv2.PSNR(img, risultato)
        errore_quad = mse(img, risultato)
        errore_abs = mae(img, risultato)

        print(f"{nome:22s} {psnr:6.2f}  {errore_quad:8.2f}  {errore_abs:7.2f}")
        righe_csv.append(f"{nome_img},{nome},{psnr:.2f},{errore_quad:.2f},{errore_abs:.2f}")

    #PANEL FINALE
    pannello = np.hstack([
        label(img, "Original"), #original
        label(img_degraded, "Degraded"), #degraded
        label(img_baseline, "1 baseline"), #1
        label(img_luminance, "2A luminance"), #2A
        label(img_channel, "2B channel"), #2B
        label(img_equalized, "3 histogram eq"), #3
        label(img_bilateral_stretch, "4A bilateral"), #4A
        label(img_nlm_stretch, "4B nlm"), #4B
    ])
    cv2.imwrite(f"confronto/{base}_confronto.png", pannello)  
        #cv2.imwrite(f"results/{base}_confronto_{nome}.png", np.hstack([img, img_degraded, risultato]))  #figura affiancata
        #cv2.imshow(f"Confronto {nome}", np.hstack([img, img_degraded, risultato]))
    
    cv2.imshow(f"{nome_img} all photos", pannello)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()        
        

#salvo la tabella in un CSV
with open(f"results/metrics.csv", "w") as f:
    f.write("\n".join(righe_csv))




