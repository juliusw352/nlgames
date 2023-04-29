import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from nlgames.Xorgame import Xorgame

def test_cvalue():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_equal(game.cvalue(1), 0.75)

def test_cvalue_repetition():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_equal(game.cvalue(2), 0.625)

def test_qvalue():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_almost_equal(game.qvalue(1), 0.85, 2)

def test_nsvalue():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_equal(game.nsval_single(), 1)

def test_nsvalue_rep():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_equal(game.nsval_rep_upper_bound(2), 1)

def test_qvalue_repetition():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    game = Xorgame(pred, prob)

    np.testing.assert_almost_equal(game.qvalue(2), 0.85**2, 2)

def test_to_nonlocal_game():
    prob = np.array([[0.25,0.25],[0.25,0.25]])
    pred = np.array([[0,0],[0,1]])

    specific_game = Xorgame(pred, prob)

    generic_pred = specific_game.to_nonlocal_game()
    generic_game = NonlocalGame(prob, generic_pred, 1)


    np.testing.assert_equal(specific_game.cvalue(1), generic_game.classical_value())
