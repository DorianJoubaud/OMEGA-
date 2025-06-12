import imageio
import os
from natsort import natsorted

# 1. Lister tous les fichiers d'images
filenames = [f for f in os.listdir() 
             if f.startswith('frontiere_comparaison_cycle_') and f.endswith('.png')]
print(os.getcwd())

# 2. Trier par ordre naturel (cycle_10, cycle_20, ..., cycle_100)
filenames = natsorted(filenames)

# 3. Charger chaque image et l'ajouter à une liste
images = []
for fn in filenames:
    images.append(imageio.imread(fn))

# 4. Sauvegarder en GIF. 'duration=0.5' signifie 0.5 seconde par image.
gif_path = 'frontieres_animation.gif'
imageio.mimsave(gif_path, images, duration=1.0)

print(f"Animation GIF créée : {gif_path}")
