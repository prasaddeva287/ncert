#!/bin/bash

cd docs
texfot pdflatex line.tex
termux-open line.pdf
cd ..
cd codes
python3 line.py
