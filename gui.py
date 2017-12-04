import tkinter as tk
from tkinter import Button
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import sys
import cv2
import numpy as np

import traceback

from segment import build_mask


base_filename = None
base_img = None
points = [] # TODO: figure out better way to do this



def click(event):
	# need to swap for row, col
	x, y = event.x, event.y
	points.append((y, x))

def stylize():

	# TODO: bug where last button click gets added as well
	points.pop()


	mask = build_mask(base_img, points)
	
	name, ext = os.path.splitext(base_filename)
	outfile = name + '_mask' + ext
	cv2.imwrite(outfile, mask * (255//3))

	print('Done Processing')
	sys.exit()

def main():
	global base_filename
	global base_img

	#This creates the main root of an application
	root = tk.Tk()
	root.title("Segmented Style Transfer")
	root.configure(background='grey')

	root.withdraw()
	base_filename = filedialog.askopenfilename()
	root.deiconify()

	if not base_filename:
		print('ERROR: please select base and tagged image!')
		sys.exit(-1)

	try:
		base_img = cv2.imread(base_filename)
	except:
		traceback.print_exc()
		print('ERROR: ' + base_filename +' is invalid input image!')
		sys.exit(-1)


	#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
	img = ImageTk.PhotoImage(Image.open(base_filename))

	#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
	panel = tk.Label(root, image = img)

	#The Pack geometry manager packs widgets in rows or columns.
	panel.pack(side = "bottom", fill = "both", expand = "yes")

	button = Button(root, text="Stylize", command=stylize)
	button.pack(side = "bottom")
	
	root.bind("<Button>", click)
	root.mainloop()


if __name__ == '__main__':
	main()