from websocket import create_connection
import time
import re

ws = create_connection("ws://127.0.0.1/websocketclient")
print("Sending 'Hello, World'...")
ws.send("Hello, World")
print("Sent")
print("Receiving...")
########################################################
#i) two words 'move e2e4' -> extract only the second word
def extractFromMoveFEN(moveString):
    return moveString.split(1)

def makeMove(moveAsFEN):
    startCommand = 'ws.send( move '
    endCommand = ')'
    return startCommand+str(moveAsFEN)+endCommand

def getOpponentMove(moveString):
    return (extractFromMoveFEN(moveString))

##### Functions for extracting game information
#1. Which color do I have base on partner number
def ownColor(partnerNumberString):
    if (partnerNumberString == 'partner 0') or (partnerNumberString == 'partner 2'):
        return 'black'
    else:
        return 'white'

def beginGame(ownColor):
    if ownColor == 'white':
        return True
    else:
        return False


def actualTime():
    return time.time()


#this function convert the given time in seconds
def convertTimeInMilliseconds(timeFromNodeServer):
    #divide by string not allowed
    extractTime = re.sub("\D", "", timeFromNodeServer)
    return (float(extractTime)/10000.00)*60


#difference between two time points
def timeDifference(end, start):
    return round(end-start, 3)


########################################################
dataList = []
# result = []
partner = ''
gameTimeFromServer = 0.0



#########################################################
sent = True
while True:
    result = ws.recv()

    # print("Received '%s'" % result)
    dataList.append(result)
    # dataList.append("datalist:"+result)

    if result == 'protover 4':
        #ws.send("feature")
        ws.send("feature san=1, time=1, variants=bughouse, otherboard=1")
        dataList = []
        continue
        #if result == 'accepted':
         #   print("Yeahh, wait for game")

    if result.find("time"):
        partner = result
        continue

    if result.find("otime"):
        gameTimeFromServer = result

    #if "partner 1" in dataList and sent and "go" in dataList:
        #msg = "move e2e4"
        #ws.send(msg)
        #sent = False

    print("dataList2:", dataList)
    print("Partner is: ", partner)
    print("Time for game is:", gameTimeFromServer)
    print("########################################################################################")
    print('I have color: ', ownColor(partner))
    print('Do I begin? ', beginGame(ownColor(partner)))
    print('The game should take: %12.3f seconds.' %convertTimeInMilliseconds(gameTimeFromServer))

    #since every necessary needed information is extracted -> start the game!

    timeBeginWhite = 0.0
    timeEndWhite = 0.0
    restTimeWhite = 0.0

    timeBeginBlack = 0.0
    timeEndBlack = 0.0
    restTimeBlack = 0.0

    whiteMove = True
    blackMove = False

    while result != '{The other board finished}':
        if beginGame(ownColor(partner)):
            timeBeginWhite = actualTime()
            makeMove('moveASString')
            timeEndWhite = actualTime() + 0.001

            timeEndWhite = timeDifference(timeEndWhite, timeBeginWhite)
            restTimeWhite = convertTimeInMilliseconds(gameTimeFromServer) - timeEndWhite

            whiteMoveFirst = False

            #wait till black make a move
            if 'move' in result:
                wholeMove = result
                getOpponentMove(wholeMove)
                whiteMoveFirst = True

                continue

        else:
            #wait until a server send a move from opponent
            if 'move' in result:
                wholeMove = result
                getOpponentMove(wholeMove)
                timeBeginBlack = actualTime()
                makeMove('thisIsAMoveFromEngine')
                timeEndBlack = actualTime() + 0.001

                timeEndBlack = timeDifference(timeEndBlack, timeBeginBlack)
                restTimeBlack = convertTimeInMilliseconds(gameTimeFromServer) - timeEndBlack

                whiteMoveFirst = True

                continue



ws.close()