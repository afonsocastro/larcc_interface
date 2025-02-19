#!/usr/bin/env python3

import json


def count_dicts_in_list(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, list):
        return len(data)
    else:
        return 0


def get_last_element_cm(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, list) and data:
        last_element = data[-1]
        test_dict = last_element.get("test", {})
        cm_value = test_dict.get("cm", "Key not found")
        return test_dict, cm_value
    else:
        return {}, "List is empty or invalid"


if __name__ == "__main__":
    json_file = "v1_4/100_times_train_test_transformer_v1_4.json"
    try:
        total_dicts = count_dicts_in_list(json_file)
        print(f"Total number of dictionaries in the list: {total_dicts}")

        test_dict, cm_value = get_last_element_cm(json_file)
        print(f"Last element 'test' key: {test_dict}")
        print(f"Value of 'cm' inside 'test': {cm_value}")

    except Exception as e:
        print(f"Error: {e}")