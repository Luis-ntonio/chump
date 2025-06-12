#!/bin/bash

git lfs track "*.pt"
git add .gitattributes
git add commit.sh
git add *.pt
git add *.py
git add *.ipynb
git commit -m "Add model and code files"
git push -u origin main