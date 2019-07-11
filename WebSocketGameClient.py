from websocket import create_connection
import bughouse.constants

class WebSocketGameClient():
    def __init__(self, url = "ws://127.0.0.1/websocketclient"):
        self.url = url
        self.ws = None
        self.message_log = []
        self.send_log = []
        self.connected = False
        self.game_started = False
        self.my_turn = None
        self.my_board = None
        self.my_team = None
        self.max_time = None
        self.my_action_stack = []
        self.my_action_ptr = 0
        self.partner_action_stack = []
        self.partner_action_ptr = 0

    def connect(self):
        self.ws = create_connection(self.url)
        while True:
            message = self.ws.recv()
            print(">> Message from server: ", message)
            self.message_log.append(message)
            if not self.game_started:
                print(">> Message from server: ", message)
                if message == 'protover 4':
                    self.ws.send('feature san=1, time=1, variants="bughouse", otherboard=1, myname="debug_engine", colors=1, time=1, done=1')
                    print("Connecttion to BugHouseGame Server established.")
                    self.connected = True

                if self.my_turn is None:
                    if message == "partner 0":
                        # I am player 1 and am at the bot right
                        self.my_turn = False
                        self.my_team = bughouse.constants.BOTTOM
                        self.my_board = bughouse.constants.RIGHT
                    elif message == "partner 1":
                        # I am player 0 and am at the bot left
                        self.my_turn = True
                        self.my_team = bughouse.constants.BOTTOM
                        self.my_board = bughouse.constants.LEFT
                    elif message == "partner 2":
                        # I am player 3 and am at the top right
                        self.my_turn = True
                        self.my_team = bughouse.constants.TOP
                        self.my_board = bughouse.constants.RIGHT
                    elif message == "partner 3":
                        # I am player 2 and am at the bot left
                        self.my_turn = False
                        self.my_team = bughouse.constants.TOP
                        self.my_board = bughouse.constants.LEFT

                if self.max_time is None:
                    if message.split(' ', 1)[0] == "time":
                        self.max_time = float(message.split(' ', 1)[1])

                if message == "go" or message == "playother":
                    print("Game is starting")
                    self.game_started = True

            elif self.game_started:
                if message.split(' ', 1)[0] == "move":
                    self.my_action_stack.append(message.split(' ', 1)[1])
                if message.split(' ', 1)[0] == "pmove":
                    self.partner_action_stack.append(message.split(' ', 1)[1])
                # ToDO implement game ended

    def check_my_stack(self) -> bool:
        if len(self.my_action_stack) <= self.my_action_ptr:
            return False
        else:
            return True

    def pop_my_stack(self) -> str:
        if len(self.my_action_stack) <= self.my_action_ptr:
            return
        else:
            ret_str = self.my_action_stack[self.my_action_ptr]
            self.my_action_ptr += 1
            if not self.check_my_stack():
                self.my_turn = True
            return ret_str

    def check_partner_stack(self) -> bool:
        if len(self.partner_action_stack) <= self.partner_action_ptr:
            return False
        else:
            return True

    def pop_partner_stack(self) -> str:
        if len(self.partner_action_stack) <= self.partner_action_ptr:
            return
        else:
            ret_str = self.partner_action_stack[self.partner_action_ptr]
            self.partner_action_ptr += 1
            return ret_str

    def send_action(self, action: str):
        if self.ws is not None:
            message = "move " + action
            self.send_log.append(message)
            self.ws.send(message)
            self.my_turn = False

    def send_message(self, message: str):
        if self.ws is not None:
            self.send_log.append(message)
            self.ws.send(message)

    def _game_ended_reset(self):
        # ToDO implement for continous games
        pass