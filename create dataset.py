import constants
import numpy as np
import re
from bughouse.BughouseEnv import BughouseEnv
import csv
import pickle
import zarr
import os
import datetime
# b = BughouseEnv(0, 0)
# l = b.get_state()._boards_fen
# l.append(str(1))
# print(b.get_state()._boards_fen)
# l = ' '.join(l)
# print(l)
# x = ([1 ], {'san' : 1})
# x[0].append(0)
# x[1]['a'] = 2
# print(x)
# moves = "1B. d4{118.969} 1b. e6{119.760} 1A. e4{118.406} 2B. Nc3{118.704} 1a. e5{119.672} 2A. Nc3{118.109} 2b. Nf6{118.869} 2a. Bc5{119.313} 3B. Nf3{118.408} 3A. Nf3{117.812} 3b. d5{118.608} 4B. Ng5{118.158} 4b. Qe7{117.477} 5B. e4{117.814} 3a. Nf6{116.688} 4A. Bc4{117.421} 5b. dxe4{116.146} 6B. Ngxe4{117.714} 6b. Nc6{114.944} 4a. d5{113.782} 7B. Nxf6+{116.229} 5A. Nxd5{115.905} 7b. gxf6{113.832} 5a. Bxf2+{112.641} 6A. Kf1{114.655} {C:JStriker(1994) kibitzes: w} 8B. P@b5{112.494} 6a. P@h3{108.860} 8b. Nb4{111.689} 9B. a3{111.244} 7A. Rg1{112.546} 9b. Nd5{110.807} 7a. hxg2+{106.985} 10B. P@e4{108.775} 8A. Rxg2{110.656} 8a. Bh3{106.885} 10b. Nxc3{109.525} 11B. bxc3{108.675} 11b. Rg8{109.275} 9A. P@e7{108.594} 12B. e5{105.300} 9a. Bxg2+{102.932} 10A. Kxg2{107.172} 10a. N@f4+{100.729} 12b. B@h4{103.497} 11A. Nxf4{104.156} 11a. Qxe7{99.619} 13B. P@g3{99.644} 12A. N@f5{99.906} 13b. N@e4{95.075} 14B. R@f3{96.754} 14b. P@g4{83.799} 15B. gxh4{95.020} 15b. gxf3{83.218} 16B. Qxf3{94.160} 16b. P@f5{75.758} 12a. B@g4{66.822} 17B. exf6{93.019} 17b. Nxf6{74.807} 13A. Nxe7{97.625} 18B. Bg5{91.831} 18b. Bg7{73.615} 19B. Bxf6{90.784} 19b. Bxf6{72.904} 13a. exf4{62.307} 14A. Bxf7+{95.172} 20B. N@h6{87.956} 14a. Kf8{58.807} 20b. P@b2{69.759} 21B. Rd1{86.112} 15A. N@g6+{92.312} 15a. hxg6{57.542} 16A. Nxg6+{91.344} 16a. Kxf7{57.214} 17A. Nxh8+{89.969} 17a. Kg8{56.636} 21b. Q@e4+{63.791} 18A. P@f7+{89.109} 18a. Kxh8{55.949} 19A. f8=Q+{87.922} 19a. N@g8{54.230} 22B. Qxe4{81.127} 22b. fxe4{62.919} 23B. Nxg8{80.564} 23b. P@d2+{58.763} 24B. Kxd2{78.751} 24b. R@c1{56.630} 25B. Rxc1{76.297} 25b. bxc1=Q+{55.419} 26B. Kxc1{75.937} 26b. Bxh4{54.387} 20A. Qxg7+{67.219} 20a. Kxg7{53.090} 21A. R@g5+{61.469} 27B. P@g7{64.281} 21a. P@g6{51.403} 22A. Rxg6+{60.406} 22a. Kxg6{50.622} 27b. Bg5+{50.611} 23A. R@g5+{59.453} 23a. Kf7{49.575} 28B. B@e3{62.078} 24A. Ne5+{58.328} 28b. Bxe3+{48.267} 29B. fxe3{61.978} 24a. Ke6{47.028} 25A. Q@f7+{55.094} 25a. Kd6{46.293} 26A. Nc4+{51.469} 26a. Kc6{45.715} 29b. P@b2+{31.864} 27A. P@b5#{42.172}"
# moves = "1A. e4{179.781} 1a. Nc6{179.670} 2A. Nc3{179.656} 1B. Nf3{178.547} 2a. Nf6{179.570} 1b. Nf6{179.419} 3A. d4{179.172} 3a. e5{179.470} 2B. e3{177.500} 2b. Nc6{179.319} 4A. dxe5{178.594} 4a. Nxe5{179.370} 5A. Nf3{178.494} 5a. Nxf3+{178.880} 3B. d4{175.844} 6A. gxf3{178.182} 6a. d6{178.660} 3b. d5{178.859} 7A. Rg1{178.082} 4B. Ng5{175.125} 7a. Be6{177.010} 4b. e6{177.777} 5B. Bb5{171.938} 8A. Bg5{176.395} 8a. Be7{175.690} 5b. N@e4{174.352} 6B. Bxc6+{169.906} 6b. bxc6{174.252} 9A. B@c4{170.645} 7B. Nxf7{168.515} 9a. N@e5{172.780} 10A. Bxe6{169.160} 7b. Bb4+{169.835} 10a. fxe6{172.680} 8B. Bd2{165.718} 11A. Bf4{165.143} 8b. Bxd2+{168.153} 11a. P@b4{171.530} 9B. Nxd2{164.312} 9b. Kxf7{168.053} 12A. Bxe5{164.284} 12a. dxe5{171.200} 10B. Nxe4{163.265} 10b. Nxe4{166.801} 11B. N@e5+{162.358} 11b. Kf8{166.230} 13A. B@f7+{160.238} 13a. Kxf7{170.270} 14A. N@g5+{158.909} 14a. Ke8{169.280} 12B. B@b4+{149.451} 12b. B@d6{164.818} 15A. Bb5+{143.127} 15a. c6{168.400} 13B. Qf3+{108.482} 13b. P@f5{161.363} 16A. N@c7+{78.876} 16a. Qxc7{167.520} 17A. Nxe6{78.455} 17a. Qd6{166.750} 18A. Nxg7+{73.893} 18a. Kd8{166.250} 14B. Bxd6+{61.482} 14b. cxd6{161.263} 19A. Ne6+{49.252} 15B. P@h6{42.701} 15b. dxe5{161.163} 19a. Kc8{158.950} 16B. hxg7+{41.779} 16b. Kxg7{160.882} 17B. N@h5+{40.654} 17b. Kf7{160.421} 18B. O-O{26.435} 18b. N@g5{158.869} 19B. Qe2{18.763} 20A. Bxc6{21.300} 19b. P@f3{157.387} 20B. gxf3{17.794} 20a. P@d2+{155.870} 21A. Qxd2{19.675} 20b. Rg8{146.962} 21a. Qxd2+{145.270} 22A. Kxd2{19.238} 22a. bxc3+{145.170} 23A. bxc3{19.003} 23a. N@c4+{144.290} 24A. Ke1{18.424} 21B. N@h6+{10.856} 21b. Ke7{144.238} 24a. P@d2+{140.120} 22B. B@g2{9.794} 25A. Ke2{17.816} 22b. Nh3+{142.135} 23B. Kh1{7.935} 23b. Q@g1+{141.794} 24B. Rxg1{6.622} 25a. bxc6{134.080} 24b. Nexf2+{141.694} 25B. Qxf2{3.934} 25b. Nxf2#{141.594}"
# moves = "1A. e4{179.015} 1B. e4{178.938} 1a. Nf6{179.656} 2A. e5{178.515} 1b. e5{178.594} 2B. Nf3{178.518} 2b. Bc5{178.281} 2a. Ne4{177.640} 3B. d4{177.967} 3b. exd4{177.062} 3A. Qe2{176.843} 4B. Nxd4{177.776} 4b. Bxd4{176.797} 5B. Qxd4{177.395} 3a. d5{176.265} 4A. d3{176.077} 5b. Nf6{175.625} 6B. Nc3{177.074} 6b. Nc6{175.078} 7B. Qd1{176.413} 7b. O-O{173.734} 4a. Nxf2{171.874} 5A. Qxf2{175.546} 8B. P@h6{174.601} 5a. e6{170.124} 6A. Qxf7+{174.061} 8b. N@e6{170.390} 6a. Kxf7{168.952} 9B. hxg7{174.161} 7A. N@g5+{173.961} 9b. Nxg7{169.500} 7a. Qxg5{168.171} 8A. Bxg5{173.861} 8a. P@f2+{167.140} 10B. Bh6{171.307} 9A. Kxf2{172.799} 9a. B@e7{166.499} 10b. Nfh5{166.719} 10A. P@f6{170.846} 10a. Bc5+{164.171} 11A. P@e3{169.002} 11a. d4{162.311} 11B. N@f5{163.485} 12A. e4{167.237} 12a. P@e3+{160.046} 11b. Nxf5{162.828} 13A. Ke1{166.440} 12B. exf5{162.794} 12b. P@d2+{161.110} 13B. Bxd2{160.971} 13b. Qe7+{158.766} 13a. P@f2+{152.014} 14A. Ke2{165.065} 14a. fxg1=Q{149.061} 15A. Rxg1{164.965} 14B. Q@e3{154.622} 15a. Nc6{127.592} 16A. fxg7{163.136} 16a. Bxg7{127.045} 17A. Bf4{159.917} 17a. Rf8{121.248} 18A. N@g5+{158.245} 18a. Kg8{120.529} 19A. Nh3{154.698} 14b. Qxe3+{114.609} 19a. Nxe5{115.154} 15B. Bxe3{153.450} 20A. Q@g3{149.526} 15b. Re8{106.202} 16B. Qxh5{151.998} 16b. Rxe3+{104.936} 17B. fxe3{151.537} 17b. P@f2+{103.233} 18B. Kxf2{149.795} 18b. Q@d2+{93.170} 19B. Qe2{145.439} 20a. P@g4{85.810} 19b. Qxe2+{88.858} 20B. Bxe2{143.806} 20b. P@g7{87.451} 21A. Kd1{135.557} 21B. P@h6{132.680} 21b. d5{79.091} 21a. N@h5{74.888} 22B. hxg7{130.286} 22b. Kxg7{75.997} 22A. Bxe5{129.261} 23B. N@h5+{128.053} 23b. Kg8{75.262} 22a. Nxg3{72.341} 24B. Q@g7#{126.511}"
# moves = '1A. Qe7+{118.979} 1B. Nf3{116.510} 1a. Be3{115.283} 1b. exd4{117.312} 2A. Bxc3+{118.879} 2a. bxc3{114.627} 2B. Nxd4{115.620} 3A. Bg4{118.028} 2b. Nc6{116.234} 3a. P@e5{113.846} 3B. Nxc6{114.167} 4A. P@e4{116.686} 3b. bxc6{116.134} 4B. Nc3{114.011} 4a. Rg1{112.424} 5A. exf3{115.695} 4b. Bc5{114.338} 5a. gxf3{111.815} 6A. Bxf3{114.773} 6a. Qxf3{110.924} 5B. P@d4{111.479} 7A. fxe5{114.493} 5b. Bb4{113.479} 7a. N@f5{109.987} 8A. Qd7{113.061} 6B. N@e5{109.744} 8a. Nxg7+{108.940} 9A. Kd8{112.721} 6b. N@e4{110.870} 9a. Bg5+{106.971} 7B. Nxf7{108.525} 10A. Kc8{111.930} 7b. Nxc3{109.542} 8B. bxc3{107.228} 10a. N@c5{103.596} 8b. Bxc3+{108.448} 9B. Bd2{106.150} 11A. e4{108.795} 9b. Bxd4{106.666} 10B. e3{102.837} 11a. Nxd7{98.643} 12A. exf3{107.513} 10b. Qf6{103.509} 12a. Nc5{95.284} 11B. Nxh8{94.259} 13A. P@d7{101.144} 13a. P@a6{94.018} 11b. Qxf2+{101.837} 12B. Kxf2{94.159} 14A. Rb8{99.602} 14a. axb7+{90.940} 15A. Rxb7{98.971} 12b. B@h4+{95.103} 15a. P@a6{88.877} 13B. P@g3{92.612} 16A. R@b1+{96.868} 16a. Bc1{85.581} 13b. P@f7{84.228} 14B. P@g6{88.956} 14b. hxg6{82.338} 17A. Rxa1{83.289} 17a. axb7+{84.316} 15B. P@h7{86.940} 18A. Kb8{82.057} 15b. Nf6{75.525} 18a. Nxd7+{66.003} 19A. Kxb7{81.226} 19a. Nc5+{65.706} 20A. Kb8{80.835} 16B. Qf3{70.550} 20a. Na6+{64.191} 21A. Ka8{80.034} 21a. P@b7+{62.800} 22A. Kxb7{79.683} 22a. Nc5+{62.503} 23A. Kb8{79.162} 23a. Na6+{61.112} 16b. R@f8{68.962} 24A. Ka8{78.290} 24a. P@b7+{59.831} 25A. Kxb7{77.799} 25a. Nc5+{59.238} 26A. Kb8{77.508}'
# print(b.get_state()._boards_fen)
import zarr
from numcodecs import Blosc

class RL_Datapoint():
    def __init__(self, state, policy, value):
        self.state = state
        self.policy = policy
        self.value = value

    def zip_data(self, filepath, cname='lz4', clevel=4):
        compressor = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        store = zarr.ZipStore(filepath, mode="w")
        zarr_file = zarr.group(store=store, overwrite=True)
        zarr_file.create_dataset(
            name="state",
            data=self.state,
            shape=self.state.shape,
            dtype=np.float64,
            chunks=(self.state.shape[0], self.state.shape[1], self.state.shape[2]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,)
        zarr_file.create_dataset(
            name="policy",
            data=self.policy,
            shape=self.policy.shape,
            dtype=np.float64,
            chunks=( self.policy.shape[0]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
        zarr_file.create_dataset(
            name="value", shape=[1,1],chunks=(1,1), dtype=np.float64, data=np.array(self.value).reshape((1,1)), synchronizer=zarr.ThreadSynchronizer()
        )        
        store.close()
class DatapointSaver():
    def __init__(self,filedirectory='dataset/',zip_length=1024, state_shape=(60,8,8),policy_shape=(constants.NB_LABELS,),autosave=True, cname='lz4', clevel=4):
        self.zip_length = zip_length
        self.X = np.empty((zip_length,state_shape[0],state_shape[1],state_shape[2]))
        self.policies = np.empty((zip_length,policy_shape[0]))
        self.values = np.empty((zip_length,1))
        self.counter = 0
        self.cname = cname
        self.clevel = clevel
        self.filedirectory = filedirectory
        self.autosave = autosave
        self.ID = 0
    
    def append(self,rl_datapoint):
        if (self.counter+1==self.zip_length):
            raise ValueError('Numpy array is filled with data and the limit was exceeded.')
        self.X[self.counter,] = rl_datapoint.state
        self.policies[self.counter,] = rl_datapoint.policy
        self.values[self.counter,] = rl_datapoint.value
        self.counter +=1

        # save automatically the data
        if (self.counter+1==self.zip_length) and self.autosave:
            self._zipData()
            self.counter = 0
    
    def _zipData(self):
        compressor = Blosc(cname=self.cname, clevel=self.clevel, shuffle=Blosc.BITSHUFFLE)
        store = zarr.ZipStore(self.filedirectory + str(self.ID) + '.zip', mode="w")
        zarr_file = zarr.group(store=store, overwrite=True)
        zarr_file.create_dataset(
            name="states",
            data=self.X,
            shape=self.X.shape,
            dtype=np.float64,
            chunks=(self.X.shape[0], self.X.shape[1], self.X.shape[2],self.X.shape[3]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,)
        zarr_file.create_dataset(
            name="policies",
            data=self.policies,
            shape=self.policies.shape,
            dtype=np.int16,
            chunks=( self.policies.shape[0], self.policies.shape[1]),
            synchronizer=zarr.ThreadSynchronizer(),
            compression=compressor,
        )
        zarr_file.create_dataset(
            name="values", shape=self.values.shape,chunks=(self.values.shape[0]), dtype=np.float64, data=self.values, synchronizer=zarr.ThreadSynchronizer()
        ) 
        self.ID+=1           
        store.close()
        
def zipData(states, policies,values, filepath, cname='lz4', clevel=4):
    compressor = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    store = zarr.ZipStore(filepath, mode="w")
    zarr_file = zarr.group(store=store, overwrite=True)
    zarr_file.create_dataset(
        name="states",
        data=states,
        shape=states.shape,
        dtype=np.float64,
        chunks=(states.shape[0], states.shape[1], states.shape[2],states.shape[3]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,)
    zarr_file.create_dataset(
        name="policies",
        data=policies,
        shape=policies.shape,
        dtype=np.float64,
        chunks=( policies.shape[0], policies.shape[1]),
        synchronizer=zarr.ThreadSynchronizer(),
        compression=compressor,
    )
    zarr_file.create_dataset(
        name="values", shape=values.shape,chunks=(values.shape[0]), dtype=np.float64, data=values, synchronizer=zarr.ThreadSynchronizer()
    )        
    store.close()


ID = 0
# @param outcome: the result of the game. 0 means that team 1 won and team 2 lost, 1 means that team 1 lost and team 2 won
def create_states_from_moves(moves, time, row, line, value_and_policy_dict, outcome, looser,datapointSaver:DatapointSaver,outputdirectory = 'dataset/'):
    # bughouseEnv.reset()  # reset the object, to reuse it
    # set the initial time for every player
    global ID
    bughouseEnv = BughouseEnv()
    bughouseEnv.set_time_remaining(time, 0, 0)
    bughouseEnv.set_time_remaining(time, 0, 1)
    bughouseEnv.set_time_remaining(time, 1, 0)
    bughouseEnv.set_time_remaining(time, 1, 1)
    line_of_games_with_illegal_moves = list()
    while moves:
        if(moves.isspace()) : break
        move = re.search(r"^(.*?)\}.*", moves)  # take the first move of the movesstring
        if (move):
            move = move.group(1)
            does_not_contain_comment = re.search(f'\.\d\d\d$',
                                                 move)  # if the move ends with a number it does not contain a comment
            if (not does_not_contain_comment):
                moves = moves[len(move) + 1:]  # if we detected a comment we skip it
                continue
            moves = moves[len(move) + 1:]
            turn = re.search(r"^(.*?)\..*", move).group(
                1)  # get the turn like '1A' which means that it is the first turn of player A who is in team 0, is white and  on board A
            player = re.search(r'[a-zA-Z]', turn).group()  # get just the player
            if (player.lower() == 'a'):
                board_number = 0
                if (player.isupper()):
                    team_number = 0
                else:
                    team_number = 1
            elif (player.lower() == 'b'):
                board_number = 1
                if (player.isupper()):
                    team_number = 1
                else:
                    team_number = 0
            move = move[len(turn) + 2:]
            san = re.search(r"^(.*?)\{.*", move).group(1)
            move = move[len(san) + 1:]
            time = float(move)

            #update value_and_policy dictionary
            fen_key = bughouseEnv.get_state()._fen
            fen_key.append(str(team_number))
            fen_key.append(str(board_number))
            fen_key = ' '.join(fen_key)
            looser_team = -1 #the default is a draw
            looser_board = -1
            if (looser == 'WhiteA'):
                looser_team = 0
                looser_board = 0
            elif(looser == 'WhiteB'):
                looser_team = 1
                looser_board = 1
            elif(looser == 'BlackA'):
                looser_team = 1
                looser_board = 0
            elif(looser == 'BlackB'):
                looser_team = 0
                looser_board = 1
            if(looser_team == team_number):
                if(looser_board == board_number):
                    value = -1
                else:
                    value = -0.8
            elif(looser_team == -1): #if the result is a draw, the value is 0 for all players
                    value = 0
            else:
                if(looser_board == board_number):
                    value = 1
                else:
                    value = 0.8

            if outcome == '0-1':
                winning_team = 0
            else:
                winning_team = 1

            #try to push the new move to the board. If move is illegal, break
            try:
                uci_move = bughouseEnv.push_san(san, team_number, board_number, to_uci=True)
                if fen_key in value_and_policy_dict:
                    # value_and_policy_dict[fen_key][0].append(1 if winning_team == team_number else 0)
                    value_and_policy_dict[fen_key][0].append(value)
                    moves_dict = value_and_policy_dict[fen_key][1]
                    if uci_move in moves_dict:
                        moves_dict[uci_move] += 1
                    else:
                        moves_dict[uci_move] = 1
                else:
                    value_and_policy_dict[fen_key] = ([value], {uci_move : 1})
            except:
                # print(row)
                # print(line)
                line_of_games_with_illegal_moves.append(line)
                break
            bughouseEnv.set_time_remaining(time, team_number, board_number)

            policy = np.zeros(constants.NB_LABELS)
            str_policies = np.array(constants.LABELS)
            policy[np.where(np.isin(str_policies, uci_move))] = 1.
            rl_datapoint = RL_Datapoint(bughouseEnv.get_state().matrice_stack, policy, value)
            # rl_datapoint.zip_data(outputdirectory + str(ID) + '.zip')
            datapointSaver.append(rl_datapoint)
            # with open(outputdirectory + str(ID) + '.pkl', 'wb') as output_file:
            #     # pickle.dump(bughouseEnv.get_state(), output_file, pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(rl_datapoint, output_file, pickle.HIGHEST_PROTOCOL)
            #     output_file.close()
            ID+=1
            if (ID%10000 == 0):
                return 1
            # if ID == 100000:
            #     return

            # rl_datapoint = RL_Datapoint(bughouseEnv.get_state(), value_and_policy_dict[fen_key][1],value_and_policy_dict[fen_key][0] )

            # with open(outputfile, 'ab') as output_file:
            #     # pickle.dump(bughouseEnv.get_state(), output_file, pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(rl_datapoint, output_file, pickle.HIGHEST_PROTOCOL)
            #     output_file.close()
    return 0


def create_dataset(input_file_with_moves, outputdirectory = 'dataset/'):
    line = 0
    value_and_policy_dict = {}
    save_step = 1000
    games_count= 0

    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
        print("The dataset directory was created at: %s", outputdirectory)

    zip_length = 10000
    datapointSaver = DatapointSaver(outputdirectory, zip_length=zip_length)
    start_time = datetime.datetime.now()
    a = datetime.datetime.now()
    file_count = 1
    games_per_file_count = 0
    with open(input_file_with_moves, encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # save every thousand rows in one file
            # also the policy and value should be saved in a seperate file 
            if not row or (row[0] in (None, "")):  # checks if row is empty
                line += 1
            else:
                moves = row[1]
                line += 1
                b = BughouseEnv()
                outcome = row[2]
                looser = row[3]
                val = create_states_from_moves(moves, row[0], row, line, value_and_policy_dict, outcome, looser,datapointSaver, outputdirectory)
                if val==1:
                    b = datetime.datetime.now()
                    c = b-a
                    print('')
                    print('ID: ', file_count, ' - Whole Time: ', (b-start_time).total_seconds(), ' - Time: ',c.total_seconds(), ' - Games in this file:', games_per_file_count, '- Avg Games/File', games_count/file_count, ' - Total Games: ', games_count)
                    a = datetime.datetime.now()
                    games_per_file_count = 0
                    file_count +=1 
                games_per_file_count +=1
                games_count+=1
                sys.stdout.write("\r%i" % games_count)

        # if ID == 100000:
        #     return

    return value_and_policy_dict



def __save_value_and_policy(outputfile, value_and_policy_dict):
    f = open(outputfile,'wb')
    pickle.dump(value_and_policy_dict,f)
    f.close()

def read_dataset(state_file):
    list_of_objects = []
    with open(state_file,'rb') as csv_file:
        while True:
            try:
                list_of_objects.append(pickle.load(csv_file))
            except EOFError:
                break
    return list_of_objects

def load_zip(filepath):
    dataset = zarr.group(store=zarr.ZipStore(filepath, mode="r"))
    rl = RL_Datapoint(np.array(dataset['state']),np.array(dataset['policy']),np.array(dataset['value'])[0][0])
    return rl

def read_value_and_policy_dict(file = 'value_and_policy_dict.pkl'):
    list_of_objects = []
    with open(file,'rb') as pkl_file:
        value_policy_dict = pickle.load(pkl_file)

    return value_policy_dict

def createDataset(state_file,value_policy_file):
    rl_datapoints = read_dataset(state_file)
    examples = []
    counter = 0
    for rl_datapoint in rl_datapoints:
        if isinstance(rl_datapoint,str):
            continue

        values, policyDict = rl_datapoint.values,rl_datapoint.policy
        value = np.mean(values)

        policy = np.mean([np.mean(elem) for elem in list(policyDict.values())])

        if len(list(policyDict.values()))> 1:
            sldkjf = 1

        # create the policy vector
        keys = np.array(list(policyDict.keys()))
        policy_values = np.array(list(policyDict.values()))

        if len(policy_values[policy_values>1]):
            skdjfj=1

        policy = np.zeros(constants.NB_LABELS)
        str_policies = np.array(constants.LABELS)
        policy[np.where(np.isin(str_policies, keys))] = policy_values
        policy = policy / sum(policy)


        examples.append((rl_datapoint.state.matrice_stack,policy,value))
        counter+=1
        if counter == 10:
            break
    return examples
import sys
# i = 9
# sys.stdout.write("\r%i" % i)
# i = 10
# sys.stdout.write("\r%i" % i)
# i = 11
# sys.stdout.write("\r%i" % i)
# sys.stdout.write("\r%i" % (i+1))
# pl = load_zip('dataset/0.zip')
# l = read_dataset(r'dataset\0.pkl')
# print(l)
value_policy_dict = create_dataset('filtered_dataset_small.csv','dataBundleCompressed/')
print('Data preprocessing finished.')

# import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from bughouse.keras.NNet import NNetWrapper as nn
# from bughouse.BugHouseGame import BugHouseGame as Game
# g = Game()
# nnet = nn(g)
# for i in range (0,23):

#     examples = createDataset(r'dataset\bughouse_testset_' + str(i) + '.csv',r'dataset\BACK_value_and_policy_dict_36.pkl')
#     length = len(examples)
#     train_set = examples[:(int(length*0.9))]
#     val_set = examples[(int(length*0.9)):]
# # test_set = examples[(int(length*0.85)):]




#     nnet.train(train_set,val_set,'VGG16_'+str(i))
#     if i == 12:
#         break
# value_policy_dict = read_value_and_policy_dict(r'dataset\BACK_value_and_policy_dict_1.0.pkl')
# create_states_from_moves(moves, b, 180)




