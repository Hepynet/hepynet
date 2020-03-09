docker run -it --rm \
  --gpus all \
  -v $(pwd):/work \
  -v /net/ustc_03/yangz/lfv:/data \
  --net=host \
  lfv/pdnn:v1
