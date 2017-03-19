import image_slicer
import argparse
import subprocess
import classify
import deepdraw
import PIL.Image
import numpy as np

#get input image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image to inpaint")
args = ap.parse_args()

#split image
tiles = image_slicer.slice(args.image,9)
score=[0,0,0,0,0,0,0,0,0]

#get least quality tile
for tile in tiles:
	tile.image.save('tile.jpg')
	score[tile.number-1] = float(subprocess.check_output(['./brisque_cpp/brisquequality','-im','tile.jpg']))
index=score.index(max(score))
print max(score),index
#tile.generate_filename(prefix='image',path=False)
#tiles[index].image.save('area.jpg')

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
tiles[index].image=deepdraw.run(tiles[index].image, classid)
#newimage=tiles[index].generate_filename(prefix='image',path=False)
#PIL.Image.fromarray(np.uint8(new)).save(newimage)
image=image_slicer.join(tiles)
image.save('final.png')
