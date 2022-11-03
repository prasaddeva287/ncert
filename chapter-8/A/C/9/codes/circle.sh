#!/bin/bash
cd /storage/9B93-1913/Download/circleassign
python3 cir.py
pdflatex circle1.tex
termux-open circle1.pdf
