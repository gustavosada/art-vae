import tensorflow as tf
import numpy as np
import model
import tkinter
from PIL import Image
from PIL import ImageTk


model_filename = 'experiments/nintendo64_20_32_120'
n = 20

vae, encoder, decoder = model.build()

vae.load_weights(model_filename+"_vae.h5")
encoder.load_weights(model_filename+"_enc.h5")
decoder.load_weights(model_filename+"_dec.h5")

dataset = np.load("fashion_data.npy")
dataset_size = len(dataset)

x = np.array([dataset[15],])
z = encoder.predict(x)

x_new = decoder.predict(z)

gui = tkinter.Tk()

scales = []

def refresh(self):
    new_z = []
    for scale in scales:
        new_z.append(scale.get())
    new_z = np.array([new_z,])
    x_new = decoder.predict(new_z)
    image = Image.fromarray(np.uint8(255*x_new[0]))
    imagetk = ImageTk.PhotoImage(image)
    panel.configure(image=imagetk)
    panel.image = imagetk

imagetk = ImageTk.PhotoImage(Image.fromarray(np.uint8(255*x_new[0])))
panel = tkinter.Label(gui, image = imagetk)
panel.pack(side = "bottom", fill = "both", expand = "yes")

for latent in z[0]:
    scale = tkinter.Scale(gui, from_=-2, to=2, resolution=0.001, orient=tkinter.HORIZONTAL, command=refresh)
    scale.set(latent)
    scale.pack()
    scales.append(scale)

gui.mainloop()
