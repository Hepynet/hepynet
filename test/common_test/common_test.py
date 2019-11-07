import os
import sys
sys.path.append("..") # add self-defined module in the parent path
sys.path.append("../..") # add self-defined module in the parent path

from lfv_pdnn.common import common_utils

print("*" * 80)
print("tests read_dict_from_txt")
file_path = os.getcwd() + "/dict_generate_test.txt"
dict_output = common_utils.read_dict_from_txt(file_path, key_type='float', value_type='float')
print("output dictionary")
print(dict_output)