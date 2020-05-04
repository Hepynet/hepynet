FOR /F "delims=" %%i IN ('git rev-parse --show-toplevel') DO set work_dir=%%i
ECHO %work_dir%

docker run -it --rm ^
  -v %work_dir%:/work ^
  -v D:\Data:/data ^
  -p 8888:8888 ^
  lfv/pdnn:v1 ^