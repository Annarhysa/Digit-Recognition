from tkinter import *
import numpy as np
from PIL import ImageGrab
from prediction import predict

window = Tk()
window.title("Handwritten Digit Recognition")
out1 = Label()

def UI():
    global out1
    widget = canvas

    #canvas co-ordinates
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + window.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    #resizing captuured image for the model
    img = ImageGrab.grab().crop((x,y,x1,y1)).resize((28,28))

    #image to grayscale
    img = img.convert('L')

    #matrix to vector conversion of image for the model
    x = np.asarray(img)
    vec = np.zeros((1,784))
    k = 0
    for i in range(28):
        for j in range(28):
            vec[0][k] = x[i][j]
            k+=1
    
    #adding theta values from the txt file
    t1 = np.loadtxt('Theta1.txt')
    t2 = np.loadtxt('Theta2.txt')

    #calling the function for prediciting the value for the image
    pred = predict(t1, t2, vec/255)

    #displaying the predicted digit on label
    out1 = Label(window, text = "Digit = " + str(pred[0]), font=('Algerian', 20))
    out1.place(x=230, y=420)


#function to clear the canvas
def clear_canvas():
    global canvas, out1
    canvas.delete('all')
    out1.destroy()

lastx, lasty = None, None

#to start the canvas
def activation(event):
    global lastx, lasty
    canvas.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    canvas.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y



# Label
L1 = Label(window, text="Handwritten Digit Recoginition", font=('MS Sans Serif', 25), fg="blue")
L1.place(x=35, y=10)
 
# Button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('MS Gothic', 15), bg="red", fg="black", command=clear_canvas)
b1.place(x=120, y=370)
 
# Button to predict digit drawn on canvas
b2 = Button(window, text="2. Prediction", font=('MS Gothic', 15), bg="green", fg="red", command=UI)
b2.place(x=320, y=370)
 
# Setting properties of canvas
canvas = Canvas(window, width=350, height=290, bg='black')
canvas.place(x=120, y=70)

#launching the application
canvas.bind('<Button-1>', activation)
window.geometry("600x500")
window.mainloop()