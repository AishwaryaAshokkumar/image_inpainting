#!/bin/bash

label=$(python classify.py -i $1 | awk '{for (i=2; i<NF; i++) printf $i " "; print $NF}' | tail -n 1)
echo "$label"
classid=$(awk  -v test="$label" -F ":" '$2~test {print $1}' labelidmap.txt)
echo "$classid"
python deepdraw.py -c $classid -i $1

