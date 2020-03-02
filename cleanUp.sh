
## Clean out all subdirectories
find . -maxdepth 1 -type d \( ! -name '.*' \) -exec bash -c "cd '{}';
echo ''; echo 'PROCESSING DIRECTORY'; pwd; echo '';
rm junk* 2> /dev/null; 
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb 2> /dev/null; 
jupyter nbconvert --to python *.ipynb 2> /dev/null;
" \;

## Clean out this directory 
echo ''; echo 'PROCESSING DIRECTORY'; pwd; echo '';
rm junk* 2> /dev/null; 
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb 2> /dev/null; 
jupyter nbconvert --to python *.ipynb 2> /dev/null;


# if you need to change a bunch of shit in a dir
# perl -pi -w -e 's/Calculating/..\/Calculating/g;' demonstration/*.ipynb
