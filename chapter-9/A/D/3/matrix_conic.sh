#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line


python3 /home/bhavani/Documents/matrix/matrix_conic/conic.py

cd /home/bhavani/Documents/matrix/matrix_conic
pdflatex conic.tex
xdg-open conic.pdf


#Test Python Installation
#Uncomment only the following line
