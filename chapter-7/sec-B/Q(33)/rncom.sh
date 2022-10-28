#!/bin/bash

#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /sdcard/dinesh/line2
texfot pdflatex cir.tex
termux-open cir.pdf


#Test Python Installation
#Uncomment only the following line
python3 /sdcard/dinesh/line2/line2.py
