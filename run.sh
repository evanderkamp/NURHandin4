#!/bin/bash

echo "Run handin 4 Evelyn van der Kamp s2138085"

# First exercise
echo "Run the first script ..."
python3 NUR_handin4Q1.py

echo "Run the second script ..."
python3 NUR_handin4Q2.py

echo "Run the third script ..."
python3 NUR_handin4Q3.py

echo "Generating the pdf"

pdflatex Handin4.tex
bibtex Handin4.aux
pdflatex Handin4.tex
pdflatex Handin4.tex
