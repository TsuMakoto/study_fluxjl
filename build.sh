cat index.md > README.md

for f (begginer/**/*.ipynb) {
  `which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to markdown $f
  mv "$f:r.html" "$f:h/index.html"
}

`which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to markdown begginer/1/mnist.ipynb
