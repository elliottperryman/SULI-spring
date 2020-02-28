find . -maxdepth 1 -type d \( ! -name '.*' \) -exec bash -c "cd '{}';
echo 'PROCESSING DIRECTORY'; pwd;
rm junk*; 
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb; 
jupyter nbconvert --to python *.ipynb;
" \;

# if you need to change a bunch of shit in a dir
# perl -pi -w -e 's/Calculating/..\/Calculating/g;' demonstration/*.ipynb
