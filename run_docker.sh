sudo docker run --runtime=nvidia --gpus all --rm -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/nsiete -it riso8500/nsiete_project bash
