docker run -it --rm \
  --gpus all \
  -v $(git rev-parse --show-toplevel):/work \
  -v /net/ustc_03/yangz:/data \
  starp/hepynet:v0.4.0
