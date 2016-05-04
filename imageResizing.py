"""
Project: Content-Aware Image Resizing
Team Member: Yunpeng Fan, Yan Chen
Project Description: Effective resizing of images should not only use geometric constraints,
			but consider the image content as well. We present a simple
			image operator called seam carving that supports content-aware
			image resizing for both reduction and expansion.

"""
from PIL import Image
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
import numpy as np
from math import fabs
import sys
import cv2

inf = 40000


# use sobel filter to calculate energy function for each pixel
def sobelFilter (im):

	im = im.convert("F")
	
	width, height = im.size 
	im_arr = np.reshape( im.getdata( ), (height, width) )
	sobel_arr = generic_gradient_magnitude( im, derivative=sobel )
	gradient = Image.new("F", im.size)
	gradient.putdata( list( sobel_arr.flat ) )
	return gradient

# used to transpose image so that we don't have to implement finding
# seams on both vertical and horizontal directions
def transposeImage(im):
	
	width, height = im.size
	cost = np.zeros( im.size )	
	im_arr = np.reshape( im.getdata( ), (height, width) )
	im_arr = np.transpose(im_arr)
	im = Image.new(im.mode, (height, width) )	
	im.putdata( list( im_arr.flat) )
	return im
	
# using calculated energy function to calculate min energy for each pixel
# and using Dynamic programming to implement finding seams
# https://en.wikipedia.org/wiki/Seam_carving
def findHS (im):
	
	width, height = im.size
	
	cost = np.zeros(im.size)
	
	im_arr = np.reshape( im.getdata( ), (height, width) )
	im_arr = np.transpose(im_arr)
	for y in range(height):
		cost[0,y] = im_arr[0, y]
	
		
	for x in range(1, width):
		for y in range(height):
			if y == 0:
				min_val = min( cost[x-1,y], cost[x-1,y+1] )
			elif y < height - 2:
				min_val = min( cost[x-1,y], cost[x-1,y+1] )
				min_val = min( min_val, cost[x-1,y-1] )
			else:
				min_val = min( cost[x-1,y], cost[x-1,y-1] )
			cost[x,y] = im_arr[x,y] + min_val
			
	min_val = inf
	path = [ ]
	
	for y in range(height):
		if cost[width-1,y] < min_val:
			min_val = cost[width-1,y]
			min_ptr = y 
			
	pos = (width-1,min_ptr)
	path.append(pos) 
	
	while pos[0] != 0:
		val = cost[pos] - im_arr[pos]
		x,y = pos
		if y == 0:
			if val == cost[x-1,y+1]:
				pos = (x-1,y+1) 
			else:
				pos = (x-1,y)
		elif y < height - 2:
			if val == cost[x-1,y+1]:
				pos = (x-1,y+1) 
			elif val == cost[x-1,y]:
				pos = (x-1,y)
			else:
				pos = (x-1,y-1)
		else:
			if val == cost[x-1,y]:
				pos = (x-1,y)
			else:
				pos = (x-1,y-1) 
		
		path.append(pos)
 
	
	return path

# use transpose function combined with findHS to find vertical seams 
def findVS ( im ): 
	
	im = transposeImage(im)
	u = findHS(im)
	for i in range(len(u)):
		temp = list(u[i])
		temp.reverse()
		u[i] = tuple(temp)
	return u

# function used to mark seams
def mark_seam (img, path):

	pix = img.load()
	path = flatten(path)

	if img.mode == "RGB": 
		for pixel in path:
			pix[pixel] = (255,0,255)
	else:
		for pixel in path:
			pix[pixel] = 0

	return img
			
# if find a seam, just remove it from image
def deleteHS (img, path):

	width, height = img.size
	i = Image.new(img.mode, (width, height-1))
	inImage  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for y in range(height):
		for x in range(width):
			if (x,y) not in path_set and x not in seen_set:
				output[x,y] = inImage[x,y]
			elif (x,y) in path_set:
				seen_set.add(x)
			else:
				output[x,y-1] = inImage[x,y]
	

	return i
			
# if find a seam, just remove it from image
def deleteVS (img, path):


	width, height = img.size
	i = Image.new(img.mode, (width-1, height))
	inImage  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for x in range(width):
		for y in range(height):
			if (x,y) not in path_set and y not in seen_set:
				output[x,y] = inImage[x,y]
			elif (x,y) in path_set:
				seen_set.add(y)
			else:
				output[x-1,y] = inImage[x,y]
		
	return i


# if find a seam, just add it from image
def addVS(img, path):

	width, height = img.size
	i = Image.new(img.mode, (width + 1, height))
	inImage  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for x in range(width):
		for y in range(height):
			if (x,y) not in path_set and y not in seen_set:
				output[x,y] = inImage[x,y]
			elif (x,y) in path_set and y not in seen_set:
				output[x,y] = inImage[x,y]
				seen_set.add( y )
				if x < width -2:
					output[x+1,y] = calAvg(inImage[x,y], inImage[x+1,y])
				else:
					output[x+1,y] = calAvg(inImage[x,y], inImage[x-1,y])
			else:
				output[x+1,y] = inImage[x,y]
				

	return i

# if find a seam, just add it from image
def addHS(img, path):

	width, height = img.size
	i = Image.new(img.mode, (width, height+1) )
	inImage  = img.load()
	output = i.load()
	path_set = set(path)
	seen_set = set()
	for y in range(height):
		for x in range(width):
			if (x,y) not in path_set and x not in seen_set:
				output[x,y] = inImage[x,y]
			elif (x,y) in path_set and x not in seen_set:
				output[x,y] = inImage[x,y]
				seen_set.add( x )
				if y < height -2:
					output[x,y+1] = calAvg(inImage[x,y], inImage[x,y+1])
				else:
					output[x,y+1] = calAvg(inImage[x,y], inImage[x,y-1])
			else:
				output[x,y+1] = inImage[x,y]
				

	return i		

# function used to calculate average between 2 neighbors
# used in adding seams
def calAvg (u, v):

	w = list(u)
	for i in range(len(u)):
		w[i] = (u[i] + v[i]) / 2
	return tuple(w)
	
	
def argmin(sequence, vals):

	return sequence[ vals.index(min(vals)) ]

# main function used to resize image 
def resize(inImage_img, resolution, output, mark):
	
	inImage = Image.open(inImage_img)
	copy1 = inImage.copy()
	width, height = inImage.size
	marked = [ ]

	while width > resolution[0]:
		u = findVS(sobelFilter(inImage))
		if mark:
			marked.append(u)
		inImage = deleteVS(inImage, u)
		width = inImage.size[0]
	
	# if adding seams, we have create a copy of original image
	# and get seams from reduced seams in the copy
	# then added all seams to create new image
	# otherwise the finding seams will keep finding same seam
	if width < resolution[0]:
		seamSet = []
		while width < resolution[0]:
			# print width
			u = findVS(sobelFilter(copy1))
			copy1 = deleteVS(copy1, u)
			width += 1
			seamSet.append(u)
			if mark:
				marked.append(u)
		for s in seamSet:
			inImage = addVS(inImage, s)
						
	while height > resolution[1]:
		v = findHS(sobelFilter(inImage))
		if mark:
			marked.append(v)
		inImage = deleteHS(inImage,v)
		height = inImage.size[1]


	# if adding seams, we have create a copy of original image
	# and get seams from reduced seams in the copy
	# then added all seams to create new image
	# otherwise the finding seams will keep finding same seam
	if height < resolution[1]:
		copy2 = inImage.copy()
		seamSet = []
		while height < resolution[1]:
			v = findHS(sobelFilter(copy2))
			copy2 = deleteHS(copy2, v)
			height += 1
			seamSet.append(v)
			# if mark:
			# 	marked.append(v)
		for s in seamSet: 
			inImage = addHS(inImage,s)

		
	inImage.save(output, "JPEG")
	
	if mark and marked != [ ]:
		mark_seam(Image.open(inImage_img), marked).show( )
		
def flatten(lst):

	for i in lst:
		if type(i) == list:
			for i in flatten(i):
				yield i
		else:
			yield i

def main():
	from optparse import OptionParser
	import os 
	usage = "usage: %prog -i [inImage image] -r [width] [height] -o [output name] \n" 
	usage+= "where [width] and [height] are the resolution of the new image"
	parser = OptionParser(usage=usage)
	parser.add_option("-i", "--image", dest="in_image", help="Input Image File")
	parser.add_option("-r", "--resolution", dest="resolution", help="Output Image size [width], [height]", nargs=2)
	parser.add_option("-o", "--output", dest="output", help="Output Image File Name")
	parser.add_option("-m", "--mark", dest="mark", help="Mark Seams Targeted. Only works for deleting", action="store_true")
	(options, args) = parser.parse_args()
	if not options.in_image or not options.resolution:
		print "Incorrect Usage; please see python resize.py --help"
		sys.exit(2)
	if not options.output:
		output = os.path.splitext(options.in_image)[0] + ".resize.jpg"
	else:
		output = options.output
	if options.mark:
		mark = True
	else:
		mark = False
	try: 
		in_image = options.in_image
		resolution = ( int(options.resolution[0]), int(options.resolution[1]) )
	except:
		print "Incorrect Usage; please see python resize.py --help"
		sys.exit(2)
		
	resize(in_image, resolution, output, mark)
	

if __name__ == "__main__":
	main()
	