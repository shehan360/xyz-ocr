import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from OCREngine import runOcrEngine


def btnClick(event=None):
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print(filename)
    T.delete('1.0', tk.END)
    T.insert(tk.END, "Processing image at:"+filename+"\n")

    T.delete('1.0', tk.END)

    implicit_seg_str, explicit_seg_string = runOcrEngine(filename, touch_char_segementation_model,
                                                         connected_character_recognition_model,
                                                         single_char_model)
    T.insert(tk.END, "Explicit segmentation:\n")
    T.insert(tk.END, explicit_seg_string)

    T.insert(tk.END, "\n\nImplicit segmentation:\n")
    T.insert(tk.END, implicit_seg_str)


r = tk.Tk()
r.title('Handwritten OCR')

button = tk.Button(r, text='Upload Image', width=25, command=btnClick)
button.pack()
S = tk.Scrollbar(r)
T = tk.Text(r, height=30, width=50)
S.pack(side=tk.RIGHT, fill=tk.Y)
T.pack(side=tk.LEFT, fill=tk.Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)
T.insert(tk.END, "Loading models....\n")
connected_character_recognition_model = load_model('./Models/connected_character_recognition_8cnn.h5')
T.insert(tk.END, "Connected character recognition model loaded!\n")
single_char_model = load_model('./Models/single_char_model_6cnn.h5')
T.insert(tk.END, "Single character classification model loaded!\n")
touch_char_segementation_model = load_model(
    './Models/implicit_segmentation_model.hdf5')
T.insert(tk.END, "Implicit segmentation model loaded!")
r.mainloop()






