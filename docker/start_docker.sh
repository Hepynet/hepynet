docker run -it --rm \
  --gpus all \
  -v $(git rev-parse --show-toplevel):/work \
  -v /net/ustc_03/yangz/lfv:/data \
  --net=host \
  lfv/pdnn:v1
