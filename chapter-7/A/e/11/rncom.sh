#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /sdcard/Download/module1/line/code1
texfot pdflatex line.tex
python3 main.py
termux-open line.pdf 
#Test Python Installation
#Uncomment only the following line
#python3 /data/data/com.termux/files/home/storage/shared/training/math/codes/tri_sss.py

