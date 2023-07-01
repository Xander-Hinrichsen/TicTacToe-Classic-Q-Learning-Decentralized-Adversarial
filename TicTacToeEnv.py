import numpy as np

class TicTacToeEnvironment():
    def __init__(self, mode='X', points_for_tie=False, tie_points=0.5):
        self.mode = mode
        self.table = np.zeros((3,3), dtype=np.uint8)
        self.observation_space = self.table.reshape(-1)
        self.points_for_tie = points_for_tie
        self.tie_points = tie_points
    def checkwin(self):
        for i in range(3):
            if self.table[0,i] == self.table[1,i] and self.table[1,i] == self.table[2,i] and self.table[1,i] != 0:
                return True, self.table[0,i]
            if self.table[i, 0] == self.table[i,1] and self.table[i,1] == self.table[i,2] and self.table[i,1] != 0:
                return True, self.table[i, 0]
        if self.table[0,0] == self.table[1,1] and self.table[1,1] == self.table[2,2] and self.table[0,0] != 0:
            return True, self.table[1,1]
        if self.table[0,2] == self.table[1,1] and self.table[1,1] == self.table[2,0] and self.table[0,2] != 0:
            return True, self.table[1,1]
        return False, 0
    def board_full(self):
        return not 0 in self.table
    def encode_table(self):
        flat_table = self.table.reshape(-1)
        encoding = ''
        for i in range(len(flat_table)):
            encoding += str(int(flat_table[i]))
        return encoding
    def observation(self):
        return self.encode_table()
    def reset(self):
        self.table = np.zeros((3,3))
        return self.encode_table()
    def encoding_to_idx(self, encoding):
        values = np.zeros(9)
        for i in range(9):
            values[i] = encoding[i]
        return int(np.sum(values* 3**np.arange(9)))
    def step(self, action, player):
        if action < 0 or action  >= 9:
            raise ValueError('action space an int in range [0,9)')
        valid_move = False
        to_place = 1 if player == 'X' else 2
        if self.table[action // 3, action % 3] == 0:
            self.table[action // 3, action % 3] = to_place
            valid_move = True
        is_win, winner_num = self.checkwin()
        board_full = self.board_full()
        terminated = True if (is_win or board_full) else False
        if is_win and winner_num == to_place:
            #print('player', player, 'won!')
            reward = 1
        elif winner_num == 2 and board_full and self.points_for_tie:
            reward = self.tie_points
        else:
            reward = 0
        return self.encode_table(), reward, terminated, board_full
    def print_board(self):
        char_table = np.array([[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']])
        for i in range(3):
            for j in range(3):
                if self.table[i,j] == 1:
                    char_table[i,j] = 'X'
                elif self.table[i,j] == 2:
                    char_table[i,j] = 'O'
            
        print(char_table[0,0], '|', char_table[0,1], '|', char_table[0,2])
        print('---------')
        print(char_table[1,0], '|', char_table[1,1], '|', char_table[1,2])
        print('---------')
        print(char_table[2,0], '|', char_table[2,1], '|', char_table[2,2])

    def play_against_policy(self, policy, mode='X', deterministic=True):
        player_piece = 1 if mode == 'X' else 2
        ai_piece = 2 if mode == 'X' else 1
        for i in range(9):
            self.print_board()
            if (mode == 'X' and (i % 2) == 0) or (mode == 'O' and (i % 2) == 1):
                print('choose your move (0-8)')
                move = -1
                while (move < 0 or move >= 9):
                    move = int(input())
                self.table[move // 3, move % 3] = player_piece
            else:
                print('press enter for ai move')
                input()
                if deterministic:
                    ai_move = np.argmax(policy[self.encoding_to_idx(self.encode_table())])
                ##sample from Q-values as if they were probability distribution
                else:
                    dist = policy[self.encoding_to_idx(self.encode_table())]
                    #turn to probability distribution
                    dist_probs = dist / np.sum(dist)
                    ##sample the action
                    ai_move = np.random.choice(np.arange(9), p=dist_probs)
                self.table[ai_move // 3, ai_move % 3] = ai_piece
            game_over, _ = self.checkwin()
            if game_over:
                break
        self.print_board()            