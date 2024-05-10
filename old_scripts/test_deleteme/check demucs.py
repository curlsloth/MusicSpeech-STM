#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:27:02 2024

@author: andrewchang
"""

import os

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

folder_path = '/Users/andrewchang/NYU_research/MusicSpeech-STM/metaTables/vocal_music_demucs_16k/metaData_MTG-Jamendo'
file_names = list_files(folder_path)


def find_duplicates(lst):
    counts = {}
    duplicates = []

    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    for item, count in counts.items():
        if count > 1:
            duplicates.append(item)

    return duplicates

print(find_duplicates(file_names))


def find_elements_with_character(lst, character):
    return [element for element in lst if character in element]

# Example usage:
character = '_'
result = find_elements_with_character(file_names, character)
print(result)


def find_missing_elements(lst):
    # Extract numbers from list elements
    numbers = [int(element.split('row')[1].split('.csv')[0]) for element in lst]
    
    # Generate a list of expected numbers
    expected_numbers = range(max(numbers) + 1)
    
    # Find missing numbers
    missing_numbers = [num for num in expected_numbers if num not in numbers]
    
    # Generate missing elements
    missing_elements = ['row{}.csv'.format(num) for num in missing_numbers]
    
    return missing_elements

# Example usage:
missing = find_missing_elements(file_names)
print(missing)