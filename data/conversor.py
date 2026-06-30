import pickle
import numpy as np
from PIL import Image
import os

# Carrega o arquivo
with open(r'C:\Users\perei\Downloads\cifar-100-python\cifar-100-python\test', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

images = data[b'data']
labels = data[b'fine_labels']
filenames = data[b'filenames']

# Pasta de saída
output_dir = 'imagens_cifar100'
os.makedirs(output_dir, exist_ok=True)

# Converte e salva todas
for i in range(len(images)):
    img_array = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img_array.astype(np.uint8))
    
    filename = filenames[i].decode('utf-8')  # ex: "apple_s_000001.png"
    img.save(os.path.join(output_dir, filename))
    
    if i % 1000 == 0:
        print(f"{i}/{len(images)} imagens salvas...")

print("Concluído!")