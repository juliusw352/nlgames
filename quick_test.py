import numpy as np
import time
from nlgames.Xorgame import Xorgame
from toqito.nonlocal_games.xor_game import XORGame

prob = np.array([[0.25,0.25],[0.25,0.25]])
pred = np.array([[0,0],[0,1]])

game = Xorgame(pred, prob)
game2 = XORGame(prob, pred, 1)

nlg_sum = 0.0
tq_sum = 0.0
for i in range(10):
    start = time.time()
    print(game.nsval_single())
    end = time.time()
    nlg_time = end-start
    nlg_sum = nlg_sum + nlg_time
for i in range(10):
    start = time.time()
    print(game2.nonsignaling_value())
    end = time.time()
    tq_time = end-start
    tq_sum = tq_sum + tq_time


print("Toqito: " + str(tq_sum/10))
print("Nlgames: " + str(nlg_sum/10))
