import image_slicer
import argparse
import subprocess
import classify
import deepdraw
import PIL.Image
import numpy as np
from resizeimage import resizeimage

#get input image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image to inpaint")
args = ap.parse_args()

#resize image 
basewidth = 1080
img = PIL.Image.open(args.image)
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save('resized_image.jpg')

#split image
tiles = image_slicer.slice('resized_image.jpg',9)
score=[0,0,0,0,0,0,0,0,0]

#get least quality tile
for tile in tiles:
	tile.image.save('tile.jpg')
	score[tile.number-1] = float(subprocess.check_output(['./brisque_cpp/brisquequality','-im','tile.jpg']))
index=score.index(max(score))

#get label of image
classlabel=classify.run(args.image).split(' ', 1)[1]
print classlabel

#get classid of label
searchfile = open("labelidmap.txt", "r")
for line in searchfile:
    if classlabel in line: classid=int(line.split(':',1)[0])
searchfile.close()
print classid

#perform deepdraw
#tiles[index].image=deepdraw.run(tiles[index].image, classid)
image=deepdraw.run(tiles[index].image,classid)
tiles[index].image=PIL.Image.fromarray(image.astype('uint8'))
#im.save('new.png')
im=image_slicer.join(tiles)
im.save('final.png')
'''
for tile in tiles:
	tile.image.show()
'''
''''
im = PIL.Image.new('RGB', get_combined_size(tiles), None)
im.save('final.jpg')
im = PIL.Image.open('final.jpg')
columns, rows = calc_columns_rows(len(tiles))
for tile in tiles:
	im.paste(tile.image, tile.coords)
	print tile.number, tile.coords
im.save('final.jpg')
'''
