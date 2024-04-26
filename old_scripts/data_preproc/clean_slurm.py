#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:22:22 2024

@author: andrewchang
"""

import os

def delete_files_with_extension(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Replace 'directory_path' with the path to your folder
directory_path = 'HPC_slurm'
extension = '.out'

delete_files_with_extension(directory_path, extension)