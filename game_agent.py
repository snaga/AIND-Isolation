"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

from logger import trace

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        print("\niterative: %s" % self.iterative)
        print("method: %s" % self.method)
        print("search depth: %d" % self.search_depth)
        print("game: \n%s" % game.to_string())
        print("game.counts: %s" % str(game.counts))
        print("legal_moves: %s" % legal_moves)
        print("time_left: %s" % time_left())

        self.time_left = time_left

        # TODO: finish this function!
        best_move = None
        best_score = None

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves   
        if len(legal_moves) == 0:
            return (-1,-1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depths = list(range(1, 5))
            else:
                depths = [self.search_depth]
            print("depths: %s" % depths)
            for depth in depths:
                print("depth %d: start" % depth)
                print("time_left: %s" % time_left())
                for next_move in legal_moves:
                    print("next_move: %s" % str(next_move))
                    if self.method == 'minimax':
                        (s, m) = self.minimax(game, depth, maxdepth=depth)
                    else:
                        raise NotImplementedError
                    print("score: %s" % s)
                    if not best_score or s > best_score:
                        best_score = s
                        best_move = next_move
                print("depth %d: end" % depth)
                print("time_left: %s" % time_left())
                print("best move %s best score %f" % (best_move, best_score))

        except Timeout:
            # Handle any actions required at timeout, if necessary
            print("get_move: timeout reached.")
            return best_move
            #raise NotImplementedError

        print("game.counts: %s" % str(game.counts))
        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True, maxdepth=2):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        p = '[%d:%s] %s' % (depth, 'max' if maximizing_player else 'min', ''.join(['  ' for x in range(0,maxdepth-depth)]))
        trace(p + "depth = %d, max player = %s" % (depth, maximizing_player))
        """
        探索する最下層に到達した場合、
        maximizing層ならアクティブなプレイヤー（自分）のスコアを、
        minimizing層なら非アクティブなプレイヤー（自分）のスコアを計算して返す
        """
        if depth <= 0:
            score = self.score(game, game.active_player if maximizing_player else game.inactive_player)
#            trace(p + "stop: score active = %f, inactive = %f" % (self.score(game, game.active_player), self.score(game, game.inactive_player)))
            return score, (-1,-1)

        legal_moves = game.get_legal_moves()
        """
        打てる手が無くなった時、maximizing層なら-1（負け）を、
        minimizing層なら1（勝ち）を返す
        """
        if len(legal_moves) == 0:
            return -1 if maximizing_player else 1, (-1, -1)
        trace(p + 'legal_moves: %s' % legal_moves)

        _score = None
        _best_move = None
        """
        打てる手の中から一つを選んで、先に進めて、再帰的にminimaxを実行する。
        maximizing層で、かつ、それまでの打ち手よりもスコアが高かったら、
        その打ち手を best move とする。
        minimizing層で、かつ、それまでの打ち手よりもスコアが低かったら、
        その打ち手を best move とする。
        """
        for next_move in legal_moves:
            trace(p + "next_move: %s" % str(next_move))
            next_state = game.forecast_move(next_move)
            (s,m) = self.minimax(next_state, depth-1, False if maximizing_player else True)
            if _score is None:
                _score = s
                _best_move = next_move
                continue
            if (maximizing_player and s > _score) or (not maximizing_player and s < _score):
                _score = s
                _best_move = next_move
        trace(p + 'best move %s score %f' % (str(_best_move), _score))
        trace(p + "game.counts: %s" % str(game.counts))
        return _score, _best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        maxdepth = 2
        p = '[%d:%s] %s' % (depth, 'max' if maximizing_player else 'min', ''.join(['  ' for x in range(0,maxdepth-depth)]))
        """
        探索する最下層に到達した場合、
        maximizing層ならアクティブなプレイヤー（自分）のスコアを、
        minimizing層なら非アクティブなプレイヤー（自分）のスコアを計算して返す
        """
        if depth <= 0:
            trace(p + "stop by depth: score active = %f, inactive = %f" % (self.score(game, game.active_player), self.score(game, game.inactive_player)))
            return self.score(game, game.active_player if maximizing_player else game.inactive_player), (-1,-1)

        legal_moves = game.get_legal_moves()
        """
        打てる手が無くなった時、maximizing層なら-1（負け）を、
        minimizing層なら1（勝ち）を返す
        """
        if len(legal_moves) == 0:
            trace(p + "stop by no legal moves")
            return -1 if maximizing_player else 1, (-1, -1)
        trace(p + 'legal_moves: %s' % legal_moves)

        best_score = None
        best_move = None
        """
        打てる手の中から一つを選んで、先に進めて、再帰的にalphabetaを実行する。
        maximizing層で、かつ、それまでの打ち手よりもスコアが高かったら、
        その打ち手を best move とする。
        minimizing層で、かつ、それまでの打ち手よりもスコアが低かったら、
        その打ち手を best move とする。
        """
        for next_move in legal_moves:
            trace(p + "next_move: %s" % str(next_move))
            next_state = game.forecast_move(next_move)
            (s, m) = self.alphabeta(next_state, depth-1, alpha, beta, not maximizing_player)
            trace(p + "alphabeta: score %f move %s alpha %f beta %f" % (s, m, alpha, beta))
            if (not best_score or
                    (maximizing_player and s > best_score) or
                    (not maximizing_player and s < best_score)):
                best_score = s
                best_move = next_move
                trace(p + "updating: best score %f best move %s" % (best_score, best_move))

            """
            maximizing層の場合、下のノードから戻ってきたスコアをalphaとする
            minimizing層の場合、下のノードから戻ってきたスコアをbetaとする
            """
            if maximizing_player:
                alpha = s
            else:
                beta = s

            """
            maximizing層の場合、それまでの best score が beta より大きかったら
            残りのノードを pruning する
            """
            if maximizing_player and (best_score >= beta):
                trace(p + "pruning by beta")
                return beta, best_move
            """
            minimizing層の場合、それまでの best score が alpha より小さかったら
            残りのノードを pruning する
            """
            if not maximizing_player and best_score <= alpha:
                trace(p + "pruning by alpha")
                return alpha, best_move
        trace(p + 'best move %s score %f' % (str(best_move), best_score))

        """
        alpha/betaが初期値の場合、best score を設定する
        """
        if maximizing_player and beta == float('inf'):
            trace(p + "updating beta: %f -> %f" % (beta, best_score))
            beta = best_score
        if not maximizing_player and alpha == float('-inf'):
            trace(p + "updating alpha: %f -> %f" % (alpha, best_score))
            alpha = best_score

        return best_score, best_move
