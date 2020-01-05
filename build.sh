cat index.md > README.md

for f (begginer/**/*.ipynb) {
  `which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to html $f
  mv "$f:r.html" "$f:h/index.html"
}
