import matplotlib.pyplot as plt
import csv
import re
import os
import pickle
from collections import Counter
import numpy as np


def filter_bpgn_file(bpgn_file, output_file):
    #check if outputfile already exists. If so, delete it.
    if os.path.exists(output_file):
        os.remove(output_file)
    else:
        print("Can not delete the file as it doesn't exists")
    f = open(bpgn_file, "r")
    linereader = f.readlines()
    cache = {}
    line_counter = 0
    list_of_game_outcomes = []
    firstround = True
    elo_counter = 0
    elo_distribution = []
    elo_list = []
    low_elo_counter = 0
    removed_matches = 0
    matches_not_checkmated_nor_resigned_nor_drawn = 0
    matches_with_too_few_moves = 0
    saved_games = 0
    for x in linereader:
        line_counter += 1
        if firstround or not("Event" in x):
            firstround = False
            value_of_curly_bracket = re.search(r'(?<=^{)(\S+)(?:\s)(\S+)', x) #gets the result description in the curly bracket like 'resigned' or 'checkmated'
            if('1A.' in x):
                if(x.startswith("{")):
                    moves = re.search(r"\}(.*)", x).group(1)
                    cache['moves'] = moves
                else:
                    cache['moves'] = x
            elif (value_of_curly_bracket):
                cache["result_description"] = value_of_curly_bracket.group(2)
                if not cache["result_description"] in list_of_game_outcomes:
                    list_of_game_outcomes.append(cache["result_description"])
                cache["looser"] = value_of_curly_bracket.group(1)
            else:
                value = re.search(r'\"(.*)\"', x) #starts with " and ends with "
                key = re.search(r'(?<=^.)(\w+)', x)

                if(value and key):
                    if 'Elo' in x:
                        y = x.split('Elo',1)[1]
                        elovalue = re.search(r'\"(.*)\"',y)
                        elo_counter += 1
                        if elovalue and elovalue.group(1) != "":
                            elo_list.append(int(elovalue.group(1)))
                    if elo_counter == 4:
                        if elo_list:
                            elo_tuple = (elo_min, elo_max, elo_avg) = (min(elo_list), max(elo_list), sum(elo_list)/len(elo_list))
                            elo_distribution.append(elo_tuple[2])
                        elo_list = []
                        elo_counter = 0
                    if(key.group() in ["WhiteA", "WhiteB", "BlackA", "BlackB"]):
                        playername = re.search(r'^(.*?)\".*', value.group(1))
                        cache[playername.group(1)] = key.group()
                    else:
                        cache[key.group()] = value.group(1)
        else:
            if (cache["result_description"] == "checkmated}" or cache["result_description"] == "resigns}" or cache["result_description"] == "drawn")  and ('moves' in cache) and  ('Result' in cache) and elo_tuple[2] and (elo_tuple[2] >= 1800) and (len(cache['moves']) > 170):
                dataelement = list()
                time = re.search(r'^\w+', cache['TimeControl']).group() #get the time without increment e.g. get '180' from '180+0'
                dataelement.append(time)
                dataelement.append(cache['moves'])
                dataelement.append(cache['Result'])
                dataelement.append(cache['result_description'])
                saved_games += 1
                if(cache["result_description"] == "drawn"):
                    dataelement.append("drawn")
                else:
                    dataelement.append(cache[cache["looser"]])

                with open(output_file, 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(dataelement)
                    csvFile.close()
            else:
                removed_matches += 1
                if not elo_tuple[2] or elo_tuple[2] < 1800:
                    low_elo_counter += 1
                elif 'moves' in cache and len(cache['moves']) <= 170:
                    matches_with_too_few_moves += 1
                else:
                    matches_not_checkmated_nor_resigned_nor_drawn += 1
            cache = {}
    print("all removed games: %i, games with to few moves: %i, games with too low Elo: %i, aborted games: %i, saved games: %i" % (removed_matches, matches_with_too_few_moves, low_elo_counter, matches_not_checkmated_nor_resigned_nor_drawn, saved_games))
    print("creating dataset finished!")


# in the precleaned dataset some games had just one move, which means that they were not played until chessmate, although in the tags they were declared as
# games that were played till the end.
#we check if the moves from the input file has more than one move and save these games into the output file
def clean_dataset_from_games_with_just_one_move(input_file, output_file):
    #check if outputfile already exists. If so, delete it.
    if os.path.exists(output_file):
        os.remove(output_file)
    else:
        print("Can not delete the file as it doesn't exists")
    line = 0
    aborted_games = list()
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if not row or (row[0] in (None, "")):  # checks if row is empty
                # if (row[0] in (None, "")):
                line += 1
            else:
                if(len(row[1]) < 18):
                    aborted_games.append(row[1])
                line += 1
    print(len(aborted_games))

    with open(output_file, 'w') as output_file, open(input_file, 'r') as input_file :
        line = 0
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        for row in reader:
            if not row or (row[0] in (None, "")):  # checks if row is empty
                line += 1
            else:
                if row[1] in aborted_games:
                    continue
                else:
                    writer.writerow(row)
                line += 1


# filter_bpgn_file('export2005.bpgn', 'filtered_dataset_2005.csv')


