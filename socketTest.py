from websocket import create_connection

ws = create_connection("ws://127.0.0.1/websocketclient")
print("Sending 'Hello, World'...")
ws.send("Hello, World")
print("Sent")
print("Receiving...")

dataList = []
# result = []
sent = True
while True:
    result = ws.recv()
    # print("Received '%s'" % result)
    dataList.append(result)
    # dataList.append("datalist:"+result)

    if result == 'protover 4':
        ws.send("feature")
        if result == 'accepted':
            print("Yeahh, wait for game")

            # print(dataList)

    if "partner 1" in dataList and sent and "go" in dataList:
        msg = "e2e4"
        ws.send(msg)
        sent = False

    print("dataList2:", dataList)
ws.close()