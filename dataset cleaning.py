# f = open('data2005.txt', "w+")
import csv
import re
f = open('export2005.bpgn', "r")
f2 = f.readlines()
cache = {}
second_linebreak = False
for x in f2:

    if x != '\n':

        moves = re.search(r'\A1', x) #starts with 1
        value_of_curly_bracket = re.search(r'(?<=^{)(?:\S+\s){1}(\S+)', x) #gets the result description in the curly bracket like 'resigned' or 'checkmated'
        if (moves):
            cache['moves'] = x
        elif (value_of_curly_bracket):
            cache["result_description"] = value_of_curly_bracket.group(1)
        else:
            value = re.search(r'\"(.*)\"', x) #starts with " and ends with "
            # key = re.search(r'([^\s]+)', x)    #everythi9ng after the first whitespace
            key = re.search(r'(?<=^.)(\w+)', x)
            if(value and key):
                cache[key.group()] = value.group(1)
    elif x == '\n':
        if second_linebreak:
            if (not (not ('moves' in cache) or not ('Result' in cache) or cache["result_description"] == "resigns}" or cache["Result"] == '*' )):

                dataelement = list()
                time = re.search(r'^\w+', cache['TimeControl']).group() #get the time without increment e.g. get '180' from '180+0'
                dataelement.append(time)
                dataelement.append(cache['moves'])
                dataelement.append(cache['Result'])

                with open('bughouse_testset.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(dataelement)
                    csvFile.close()

            cache = {}
            second_linebreak = False
        else:
#////////////////////////////////////////////////////////////////
            moves = re.search(r'\A1', x) #starts with 1
            if (moves):
                cache['moves'] = x
            else:
                value = re.search(r'\"(.*)\"', x) #starts with " and ends with "
                key = re.search(r'([^\s]+)', x)    #everythi9ng after the first whitespace
            if(value and key):
                cache[key.group()] = value.group()
#//////////////////////////////////////////////////////////////
            second_linebreak = True

