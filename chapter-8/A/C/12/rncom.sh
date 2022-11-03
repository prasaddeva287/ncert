#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math
#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /sdcard/circle
python3 assign5.py
texfot pdflatex assign5.tex
termux-open assign5.pdf


#Test Python Installation
#Uncomment only the following line
#python3 gvv_math_eg.tex

