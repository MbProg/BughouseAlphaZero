from websocket import create_connection
import time
import bughouse.constants

class WebSocketGameClient():
    def __init__(self, args = None):
        if args is not None:
            self.tick_time = args.tick_time
        else:
            self.tick_time = 0.05
        self.url = args.url
        self.ws = None
        self.message_log = []
        self.send_log = []
        self.connected = False
        self.game_started = False
        self.player_ready = False
        self.my_turn = None
        self.my_board = None
        self.my_color = None
        self.my_team = None
        self.max_time = None
        self.my_action_stack = []
        self.my_action_ptr = 0
        self.partner_action_stack = []
        self.partner_action_ptr = 0
        self.id_stack = []
        self.id_ptr = 0

        self.win_counter = 0
        self.lose_counter = 0

    def connect(self):
        self.ws = create_connection(self.url)
        while True:
            message = self.ws.recv()
            print(">> Message from server: ", message)
            self.message_log.append(message)
            if not self.game_started:
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
                        self.my_color = bughouse.constants.BLACK
                    elif message == "partner 1":
                        # I am player 0 and am at the bot left
                        self.my_turn = True
                        self.my_team = bughouse.constants.BOTTOM
                        self.my_board = bughouse.constants.LEFT
                        self.my_color = bughouse.constants.WHITE
                    elif message == "partner 2":
                        # I am player 3 and am at the top right
                        self.my_turn = True
                        self.my_team = bughouse.constants.TOP
                        self.my_board = bughouse.constants.RIGHT
                        self.my_color = bughouse.constants.WHITE
                    elif message == "partner 3":
                        # I am player 2 and am at the bot left
                        self.my_turn = False
                        self.my_team = bughouse.constants.TOP
                        self.my_board = bughouse.constants.LEFT
                        self.my_color = bughouse.constants.BLACK

                if self.max_time is None:
                    if message.split(' ', 1)[0] == "time":
                        self.max_time = float(message.split(' ', 1)[1])

                if message == "go" or message == "playother":
                    print("Game is starting")
                    self.game_started = True

            elif self.game_started:
                message_split = message.split(' ', 1)[0]
                # print(message_split)
                if message_split == "move":
                    action_string = message.split(' ', 1)[1]
                    if self.fix_action_input:
                        action_string = self._fix_action_string(action_string)
                    self.my_action_stack.append(action_string)
                    self.id_stack.append(0)
                if message_split == "pmove":
                    action_string = message.split(' ', 1)[1]
                    if self.fix_action_input:
                        action_string = self._fix_action_string(action_string)
                    self.partner_action_stack.append(action_string)
                    self.id_stack.append(1)
                if message_split == '0-1':
                    if self.my_color == bughouse.constants.BLACK:
                        self.win_counter += 1
                        print("WIN")
                        self._game_ended_reset()
                    else:
                        self.lose_counter += 1
                        print("LOSE")
                        self._game_ended_reset()
                if message_split == '1-0':
                    if self.my_color == bughouse.constants.WHITE:
                        self.win_counter += 1
                        print("WIN")
                        self._game_ended_reset()
                    else:
                        self.lose_counter += 1
                        print("LOSE")
                        self._game_ended_reset()
            time.sleep(self.tick_time)

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

    def check_stack_id(self) -> bool:
        if len(self.id_stack) <= self.id_ptr:
            return False
        else:
            return True

    def pop_stack_id(self) -> str:
        if len(self.id_stack) <= self.id_ptr:
            return
        else:
            stack_id = self.id_stack[self.id_ptr]
            self.id_ptr += 1
            return stack_id

    def send_action(self, action: str):
        if self.ws is not None and self.game_started and self.player_ready:
            message = "move " + action
            self.send_log.append(message)
            self.ws.send(message)
            self.my_turn = False
            return True
        return False

    def send_message(self, message: str):
        if self.ws is not None:
            self.send_log.append(message)
            self.ws.send(message)
            return True
        return False

    def ready_check(self):
        self.player_ready = True

    def _game_ended_reset(self):
        self.player_ready = False
        self.game_started = False
        self.my_turn = None
        self.my_board = None
        self.my_color = None
        self.my_team = None
        self.max_time = None
        self.id_stack = []
        self.id_ptr = 0
        self.my_action_stack = []
        self.my_action_ptr = 0
        self.partner_action_stack = []
        self.partner_action_ptr = 0

    def _fix_action_string(self, action_string: str):
        if '@' in action_string:
            suffix,appendix = action_string.split('@', 1)
            return suffix.upper() + '@' + appendix
        return action_string

