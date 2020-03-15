import os
import sys

# Show tensorflow warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from lfv_pdnn.train.job_utils import *

def main():
  argv_len = len(sys.argv)
  if argv_len <=1 :
    print('No ini file path specified!')
    print('Usage: execute_pdnn_job path/to/ini-file')
  elif argv_len >= 2:
    for i in range(1, argv_len):
      print('#'*80)
      print('Executing: ', sys.argv[i])
      ex_test = job_executor(sys.argv[i])
      ex_test.get_config()
      ex_test.load_arrays()
      ex_test.execute_jobs()
    print('#'*80)
    print('Done!')
    print('#'*80)
