#!/bin/env python

import os
import site

# add package path to python path
package_path = os.getcwd()
print(package_path)
user_site_dir = site.getusersitepackages()
if not os.path.exists(user_site_dir):
  os.makedirs(user_site_dir)
path_cfg_file = open(user_site_dir + "/pdnn.pth", 'w')
path_cfg_file.write(package_path)
path_cfg_file.close()
