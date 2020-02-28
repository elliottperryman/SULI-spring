find . -maxdepth 1 -type d \( ! -name '.*' \) -exec bash -c "cd '{}';
echo 'PROCESSING DIRECTORY'; pwd;
rm junk*; 
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb; 
jupyter nbconvert --to python *.ipynb;
" \;


