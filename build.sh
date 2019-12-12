cat index.md > README.html
`which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to html begginer/mnist.ipynb
mv begginer/mnist.html begginer/index.html
