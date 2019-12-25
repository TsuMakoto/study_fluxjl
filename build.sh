cat index.md > README.md
`which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to html begginer/1/mnist.ipynb
`which docker` run --rm --mount type=bind,src=`pwd`,dst=/home/jovyan jupyter/datascience-notebook jupyter nbconvert --to html begginer/2/mnist.ipynb
mv begginer/1/mnist.html begginer/1/index.html
mv begginer/2/mnist.html begginer/2/index.html
