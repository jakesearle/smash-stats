import json
import math
import os.path
import re
import sys
from difflib import get_close_matches
from statistics import mean, median
from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

JSON_FILENAME = "data.json"
ALTERNATE_NAMES = {
    "olimar_white_2": "olimar",
    "olimar_white_1": "olimar",
    "olimar_r/b/y_2": "olimar",
    "olimar_white": "olimar",
    "olimar_r/b/y_1": "olimar",
    "olimar_r/b/y": "olimar",
    "olimar_purple_2": "olimar",
    "olimar_purple_1": "olimar",
    "olimar_purple": "olimar",
    "pokemon_trainer_charizard": "charizard",
    "pokemon_trainer_ivysaur": "ivysaur",
    "pokemon_trainer_squirtle": "squirtle",
    "dk": "donkey_kong",
    "samus/dark_samus": [
        "samus",
        "dark_samus"
    ],
    "peach/daisy": [
        "peach",
        "daisy"
    ],
    "marth/lucina": [
        "marth",
        "lucina"
    ],
    "roy/chrom": [
        "roy",
        "chrom"
    ],
    "pit/dark_pit": [
        "pit",
        "dark_pit"
    ],
    "ryu/ken": [
        "ryu",
        "ken"
    ],
    "simon/richter": [
        "simon",
        "richter"
    ],
    "mii_sword_fighter": "mii_swordfighter",
    "dedede": "king_dedede",
    "pokemon_trainer_all": [
        "charizard",
        "ivysaur",
        "squirtle"
    ],
    "nana": "ice_climbers",
    "popo": "ice_climbers",
    "ice_climbers_partner": "ice_climbers",
    "ice_climbers_leader": "ice_climbers",
    "minmin": "min_min",
    "steve_wood": "steve",
    "r.o.b": "r.o.b.",
    "metaknight": "meta_knight",
    "p.t._[squirtle]": "squirtle",
    "p.t._[ivysaur]": "ivysaur",
    "p.t._[charizard]": "charizard",
    "rosa_no_luma": "rosalina_&_luma",
    "rosalina": "rosalina_&_luma",
}
IGNORABLE_CATEGORIES = [
    "hard_land",
    "grab_range",
    "weight",
    "#1",
    "#2",
    "#3",
    "grab",
    "grab_post-shieldstun",
    "item_throw_forward",
    "item_throw_back",
    "rank",
    "key",
    "jump_z-drop_front",
    "jump_z-drop_behind",
    "soft_land_universal",
    "attack_range"
]
NO_MATCH_VAL = 0.25


def compare(my_char, char_dict):
    # Find all key types
    category_types = set()
    for name, stats in char_dict.items():
        category_types.update(stats.keys())
    my_char_stats = char_dict[my_char]

    differences = []
    # Iterate through each character
    for other_char, data in char_dict.items():
        # Find matching types
        other_char_stats = data
        # Get the difference between each attribute
        temp_sum = 0
        for type in category_types:
            if type not in my_char_stats or type not in other_char_stats:
                temp_sum += NO_MATCH_VAL
                continue
            my_val = my_char_stats[type]
            other_val = other_char_stats[type]
            temp_sum += (my_val - other_val) ** 2
        distance = math.sqrt(temp_sum)
        differences.append((distance, other_char))
    differences.sort()
    print("Most similar characters: ")
    for i, (diff, name) in enumerate(differences):
        print(f"#{i + 1}: {name} ({format(diff, '.4f')})")
    print('')
    return differences


def is_floatable(obj):
    try:
        float(obj)
        return True
    except ValueError:
        return False


def get_dictionary():
    if not os.path.exists(JSON_FILENAME):
        print("File not found. Please run \"pull\" and save results")
        return None
    with open(JSON_FILENAME, 'r') as file:
        character_dictionary = json.load(file)
        return character_dictionary


def get_image(path):
    return OffsetImage(plt.imread(path), zoom=.25)


def get_matrix(dictionary):
    # Get a list of all characters and features
    master_chars = {}
    master_features = {}
    for char, data in dictionary.items():
        if char not in master_chars:
            master_chars[char] = len(master_chars)
        for feature, value in data.items():
            if feature not in master_features:
                master_features[feature] = len(master_features)

    # Fill out the matrix
    matrix = [[None for j in range(len(master_chars))] for i in range(len(master_features))]
    for char, data in dictionary.items():
        for feature, value in data.items():
            i = master_features[feature]
            j = master_chars[char]
            if isinstance(value, list):
                value = mean(value)
            matrix[i][j] = value

    # Fill out 'None' values with the median of that row
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if cell is None:
                matrix[i][j] = nullable_median(row)
    return matrix, [char for char in master_chars.keys()]


def get_pca(matrix):
    matrix = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2)
    # principal_df = pd.DataFrame(data=principal_components, columns=['x', 'y'])
    return pca.fit_transform(matrix)


# Edits in-place
def normalize(dictionary):
    # Find all key types
    category_types = set()
    for name, stats in dictionary.items():
        category_types.update(stats.keys())
    # For each type
    for type in category_types:
        # Find min/max
        minimum, maximum = sys.float_info.max, sys.float_info.min
        for data in dictionary.values():
            if type in data:
                val = data[type]
                if isinstance(val, list):
                    val = mean(val)
                minimum = min(minimum, val)
                maximum = max(maximum, val)
        # Skip if there is no variation
        if minimum == maximum:
            # Delete the ones that don't need to be normalized
            for char, data in dictionary.items():
                if type in data:
                    data.pop(type)
            continue
        # Normalize based off of that
        for char, data in dictionary.items():
            if type in data:
                val = data[type]
                if isinstance(val, list):
                    val = mean(val)
                dictionary[char][type] = (val - minimum) / (maximum - minimum)


def nullable_median(nums):
    return median([num for num in nums if num is not None])


def parse_flags(flags_str):
    return list(flags_str[1:])


def plot(matrix, char_list):
    fig, ax = plt.subplots()
    x = matrix[0]
    y = matrix[1]
    paths = [f"./icons/'{char}'.png" for char in char_list]
    ax.scatter(x, y, c='w')

    for x0, y0, path in zip(x, y, paths):
        ab = AnnotationBbox(get_image(path), (x0, y0), frameon=False)

        ax.add_artist(ab)
    ax.grid()
    plt.show()


def sanitize(string):
    string = string.strip().lower()
    # Replace delimiters
    string = re.sub(f'[+\\s(,]', '_', string)
    # Delete weird stuff
    string = re.sub(f"[)~\\s]", '', string)
    # Delete double underscores
    string = re.sub(f'__', '_', string)
    # Wierd slash thing
    string = re.sub(f'_/_', '/', string)
    # Weird trailing stuff
    string = re.sub(f'(_s|_~|_)$', '', string)
    # TODO: replace '/u2014' thing
    return string


def scrape():
    character_dictionary = {}
    url = "https://ultimateframedata.com/stats"
    html = urlopen(url)
    site = BeautifulSoup(html.read(), features="html.parser")
    tables = [table for table in site.findAll("div", {"class": "statstablecontainer"})]
    for table in tables:
        prefix = sanitize(table.find("h2").text)
        headers = [sanitize(header.text) for header in table.find("thead").findAll("th")]
        rows = table.find("tbody").findAll("tr")
        for row in rows:
            cells = [cell for cell in row.findAll("td")]
            current_char = None
            for i, cell in enumerate(cells):
                matching_header = headers[i]
                cell_content = sanitize(cell.text)
                # Convert content if possible
                if is_floatable(cell_content):
                    cell_content = float(cell_content)
                # Skip useless headers
                if matching_header in IGNORABLE_CATEGORIES:
                    continue
                # Log current character that I'm on
                elif matching_header == 'character':
                    current_char = cell_content
                    # Finds better names if they exist
                    if current_char in ALTERNATE_NAMES:
                        current_char = ALTERNATE_NAMES[current_char]
                    # List-izes for single strings so the for loop works
                    if isinstance(current_char, str):
                        current_char = [current_char]
                    # Iterate through characters and add if they need to exist
                    for char in current_char:
                        if char not in character_dictionary:
                            character_dictionary[char] = {}
                elif isinstance(cell_content, float):
                    # For each character in list
                    for char in current_char:
                        key = f"{prefix}_{matching_header}"
                        # Key already exists
                        if key in character_dictionary[char]:
                            prev_content = character_dictionary[char][key]
                            if isinstance(prev_content, float):
                                character_dictionary[char][key] = [prev_content, cell_content]
                            elif isinstance(prev_content, list):
                                character_dictionary[char][key].append(cell_content)
                            else:
                                raise Exception("Unrecognized value type")
                        # New key
                        else:
                            character_dictionary[char][key] = cell_content
    return character_dictionary


def main():
    while True:
        cmd = input("Enter a command: ")
        argv = cmd.split()
        argc = len(argv)

        if argv[0] == 'normalize':
            # Get dictionary
            character_dictionary = get_dictionary()
            if character_dictionary is None:
                continue
            # Normalize
            normalize(character_dictionary)
            # Write
            with open(JSON_FILENAME, 'w+') as file:
                json_formatted_str = json.dumps(character_dictionary, indent=2)
                file.write(json_formatted_str)

        elif argv[0] == 'pull':
            character_dictionary = scrape()
            if argc > 1:
                flags = parse_flags(argv[1])
                if 'n' in flags:
                    normalize(character_dictionary)
                if 'w' in flags:
                    with open(JSON_FILENAME, 'w+') as file:
                        json_formatted_str = json.dumps(character_dictionary, indent=2)
                        file.write(json_formatted_str)
                if 'p' in flags:
                    print(character_dictionary)

        elif argv[0] == 'print num-keys':
            character_dictionary = get_dictionary()
            if character_dictionary is None:
                continue
            entries = sorted([(k, len(v)) for k, v in character_dictionary.items()], key=lambda x: x[1])
            for char, num in entries:
                print(f"\t\"{char}\": {num}")

        elif argv[0] == 'cmp':
            if argc > 1:
                char = argv[1]
            else:
                print("Please input a character")
                continue
            character_dictionary = get_dictionary()
            # Make sure the dictionary exists
            if character_dictionary is None:
                continue
            # Make sure the character exists
            if char not in character_dictionary:
                close = get_close_matches(char, character_dictionary.keys())[0]
                if input(f"\"{char}\" is not in the list. Did you mean \"{close}\"? (Y/n): ").strip().lower()[0] == 'y':
                    char = close
                else:
                    continue
            compare(char, character_dictionary)

        elif argv[0] == "cmp-all":
            # Make indices of character
            character_dictionary = get_dictionary()
            sorted_chars = sorted([k for k in character_dictionary.keys()])
            char_index_map = {char: i for i, char in enumerate(sorted_chars)}
            chart = [[None for i in range(len(char_index_map))] for j in range(len(char_index_map))]
            for char in character_dictionary:
                cmp_list = compare(char, character_dictionary)
                i = char_index_map[char]
                for cmp_val, cmp_char in cmp_list:
                    j = char_index_map[cmp_char]
                    chart[i][j] = cmp_val
            print(chart)

        elif argv[0] == 'plot':
            char_dict = get_dictionary()
            matrix, char_list = get_matrix(char_dict)
            # char_list.sort()
            matrix_t = np.array(matrix).T.tolist()
            pca = get_pca(matrix_t)
            pca_t = np.array(pca).T.tolist()
            plot(pca_t, char_list)

        elif argv[0] == 'quit':
            break


if __name__ == '__main__':
    main()
