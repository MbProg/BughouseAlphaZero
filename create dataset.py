import re
from bughouse.BughouseEnv import BughouseEnv
import csv
import pickle

# b = BughouseEnv(0, 0)
# print(b.get_state()._boards_fen)
# b.push_san('e4', 0, 0)
# print(b.get_state()._boards_fen)

with open('bughouse_dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if (line_count == 0):
            first_mover = row[1][
                1]  # gets the player who performs the first move, it is either player A or B (white players on the two boards)
            if (first_mover == 'A'):
                team = 0
                board = 0
            elif (first_mover == 'B'):
                team = 1
                board = 1

            b = BughouseEnv(team, board)  # a new bughouse game is instantiated
            #save the first state in a file
            # with open('bughouse_testset.txt', 'ab') as output:
            #     pickle.dump(b.get_state(), output, pickle.HIGHEST_PROTOCOL)
            #     # writer = csv.writer(csvFile)
            #     # writer.writerow(b.get_state())
            #     output.close()

            #read the first state from the file
            with open('bughouse_testset.txt', 'rb') as input:
                first_state = pickle.load(input)
                print(first_state._boards_fen)

            line_count += 1
        else:
            if (row[0] in (None, "")):
                continue
            # else:
            #     print(f'\t{row[0]} and {row[1]}')
# moves = "1B. d4{118.969} 1b. e6{119.760} 1A. e4{118.406} 2B. Nc3{118.704} 1a. e5{119.672} 2A. Nc3{118.109} 2b. Nf6{118.869} 2a. Bc5{119.313} 3B. Nf3{118.408} 3A. Nf3{117.812} 3b. d5{118.608} 4B. Ng5{118.158} 4b. Qe7{117.477} 5B. e4{117.814} 3a. Nf6{116.688} 4A. Bc4{117.421} 5b. dxe4{116.146} 6B. Ngxe4{117.714} 6b. Nc6{114.944} 4a. d5{113.782} 7B. Nxf6+{116.229} 5A. Nxd5{115.905} 7b. gxf6{113.832} 5a. Bxf2+{112.641} 6A. Kf1{114.655} {C:JStriker(1994) kibitzes: w} 8B. P@b5{112.494} 6a. P@h3{108.860} 8b. Nb4{111.689} 9B. a3{111.244} 7A. Rg1{112.546} 9b. Nd5{110.807} 7a. hxg2+{106.985} 10B. P@e4{108.775} 8A. Rxg2{110.656} 8a. Bh3{106.885} 10b. Nxc3{109.525} 11B. bxc3{108.675} 11b. Rg8{109.275} 9A. P@e7{108.594} 12B. e5{105.300} 9a. Bxg2+{102.932} 10A. Kxg2{107.172} 10a. N@f4+{100.729} 12b. B@h4{103.497} 11A. Nxf4{104.156} 11a. Qxe7{99.619} 13B. P@g3{99.644} 12A. N@f5{99.906} 13b. N@e4{95.075} 14B. R@f3{96.754} 14b. P@g4{83.799} 15B. gxh4{95.020} 15b. gxf3{83.218} 16B. Qxf3{94.160} 16b. P@f5{75.758} 12a. B@g4{66.822} 17B. exf6{93.019} 17b. Nxf6{74.807} 13A. Nxe7{97.625} 18B. Bg5{91.831} 18b. Bg7{73.615} 19B. Bxf6{90.784} 19b. Bxf6{72.904} 13a. exf4{62.307} 14A. Bxf7+{95.172} 20B. N@h6{87.956} 14a. Kf8{58.807} 20b. P@b2{69.759} 21B. Rd1{86.112} 15A. N@g6+{92.312} 15a. hxg6{57.542} 16A. Nxg6+{91.344} 16a. Kxf7{57.214} 17A. Nxh8+{89.969} 17a. Kg8{56.636} 21b. Q@e4+{63.791} 18A. P@f7+{89.109} 18a. Kxh8{55.949} 19A. f8=Q+{87.922} 19a. N@g8{54.230} 22B. Qxe4{81.127} 22b. fxe4{62.919} 23B. Nxg8{80.564} 23b. P@d2+{58.763} 24B. Kxd2{78.751} 24b. R@c1{56.630} 25B. Rxc1{76.297} 25b. bxc1=Q+{55.419} 26B. Kxc1{75.937} 26b. Bxh4{54.387} 20A. Qxg7+{67.219} 20a. Kxg7{53.090} 21A. R@g5+{61.469} 27B. P@g7{64.281} 21a. P@g6{51.403} 22A. Rxg6+{60.406} 22a. Kxg6{50.622} 27b. Bg5+{50.611} 23A. R@g5+{59.453} 23a. Kf7{49.575} 28B. B@e3{62.078} 24A. Ne5+{58.328} 28b. Bxe3+{48.267} 29B. fxe3{61.978} 24a. Ke6{47.028} 25A. Q@f7+{55.094} 25a. Kd6{46.293} 26A. Nc4+{51.469} 26a. Kc6{45.715} 29b. P@b2+{31.864} 27A. P@b5#{42.172}"
# while games:
#     first_mover = games[0][1] #gets the player who performs the first move, it is either player A or B (white players on the two boards)
#     if(first_mover == 'A'):
#         team = 0
#         board = 0
#     elif(first_mover == 'B'):
#         team = 1
#         board = 1
#
#     b = BughouseEnv(team, board) #a new bughouse game is instantiated
#     #ToDo set the remaining time: b.set_time_remaining(time, team, board)
#     #ToDo save the first state b.getstate() in the csv file
# while moves:
#     move = re.search(r"^(.*?)\}.*", moves).group(1) #take the first move of the movesstring
#     does_not_contain_comment = re.search(f'\d+$', move) #if the move ends with a number it does not contain a comment
#     if(not does_not_contain_comment):
#         moves = moves[len(move)+1:] #if we detected a comment we skip it
#         continue
#     moves = moves[len(move)+1:]
#     turn = re.search(r"^(.*?)\..*", move).group(1) #get the turn like '1A' which means that it is the first turn of player A who is in team 0, is white and  on board A
#     player = re.search(r'[a-zA-Z]').group(1) # get just the player
#     if(player.lower() == 'a'):
#         board_number = 0
#         if(player.isupper()):
#             team_number = 0
#         else:
#             team_number = 1
#     elif(player.lower()=='b'):
#         board_number = 1
#         if(player.isupper()):
#             team_number = 1
#         else:
#             team_number = 0
#     move = move[len(turn)+2:]
#     san = re.search(r"^(.*?)\{.*", move).group(1)
#     move = move[len(san)+1:]
#     time = move
#     b.push(move, team_number, board_number)
#     b.set_time_remaining(time, team_number, board_number)
#     #ToDo save the new state in the csv file
