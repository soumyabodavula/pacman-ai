# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import sys
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        food_dist = sys.maxsize # distance to closest food from new position
        food_pos = newPos
        for food in foodList:  
            if food_dist > util.manhattanDistance(newPos, food):
                food_pos = food
                food_dist = util.manhattanDistance(newPos, food)

        ghost_dist = sys.maxsize
        ghost_pos = newPos
        for ghost in newGhostStates: # distance to closest ghost from new pos
            if ghost_dist > util.manhattanDistance(newPos, ghost.getPosition()):
                ghost_pos = ghost.getPosition()
                ghost_dist = util.manhattanDistance(newPos, ghost.getPosition())

        food_pos = (float(food_pos[0]), food_pos[1])
        dist_food_ghost = util.manhattanDistance(food_pos, ghost_pos) # distance from closest food to closest ghost
        
        # .001 added to avoid division by zero error
        return successorGameState.getScore() + (1/(food_dist+.001)) - (1/(ghost_dist+.001)) - 1/(dist_food_ghost+.001)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, depth, agentNum, gameState):
        numAgents = gameState.getNumAgents()

        if agentNum >= numAgents: # initalized to 0 in getAction
            agentNum = 0 # reset to pacman's turn
            depth += 1 # go to new layer in tree, initialized to 0 in getAction
        
        if depth == self.depth: # max depth reached
            return (Directions.STOP, self.evaluationFunction(gameState))
        
        if gameState.isWin() or gameState.isLose():
            return(Directions.STOP, self.evaluationFunction(gameState))

        optimal_score = None
        optimal_action = None

        if agentNum == 0: # pacman's turn
            pacmanActions = gameState.getLegalActions(agentNum)
            for action in pacmanActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.minimax(depth, agentNum+1, suc_state)[1]
                if optimal_score is None or suc_score > optimal_score:
                    optimal_score = suc_score # maximize
                    optimal_action = action
        
        if agentNum != 0:
            agentActions = gameState.getLegalActions(agentNum)
            for action in agentActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.minimax(depth, agentNum+1, suc_state)[1] # next ghost's turn
                if optimal_score is None or suc_score < optimal_score:
                    optimal_score = suc_score # minimize
                    optimal_action = action
        
        if optimal_action is None: # no successor states => leaf state
            return (None, self.evaluationFunction(gameState))
        
        return (optimal_action, optimal_score)


    def getAction(self, gameState):
        """
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP} 

        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        start_depth = 0
        start_agent_turn = 0
        optimal_action = self.minimax(start_depth, start_agent_turn, gameState)[0]
        return optimal_action

        util.raiseNotDefined()
    
   

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax(self, depth, agentNum, gameState, alpha, beta):
        numAgents = gameState.getNumAgents()
        if agentNum >= numAgents: # initalized to 0 in getAction
            agentNum = 0 # reset to pacman's turn
            depth += 1 # go to new layer in tree, initialized to 0 in getAction
        
        if depth == self.depth: # max depth reached
            return (Directions.STOP, self.evaluationFunction(gameState))
        
        if gameState.isWin() or gameState.isLose():
            return(Directions.STOP, self.evaluationFunction(gameState))

        optimal_score = None
        optimal_action = None

        if agentNum == 0: # pacman's turn
            pacmanActions = gameState.getLegalActions(agentNum)
            for action in pacmanActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.minimax(depth, agentNum+1, suc_state, alpha, beta)[1]

                if optimal_score is None or suc_score > optimal_score:
                    optimal_score = suc_score # maximize
                    optimal_action = action
                
                alpha = max(alpha, suc_score)
                if alpha > beta:
                    break

        if agentNum != 0:
            agentActions = gameState.getLegalActions(agentNum)
            for action in agentActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.minimax(depth, agentNum+1, suc_state, alpha, beta)[1] # next ghost's turn
                if optimal_score is None or suc_score < optimal_score:
                    optimal_score = suc_score # minimize
                    optimal_action = action

                beta = min(beta, suc_score)
                if beta < alpha:
                    break
        
        if optimal_action is None: # no successor states => leaf state
            return (None, self.evaluationFunction(gameState))
        
        return (optimal_action, optimal_score)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        start_depth = 0
        start_agent_turn = 0
        alpha = -1 * sys.maxsize
        beta = sys.maxsize
        optimal_action = self.minimax(start_depth, start_agent_turn, gameState, alpha, beta)[0]
        return optimal_action

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, depth, agentNum, gameState):
        numAgents = gameState.getNumAgents()

        if agentNum >= numAgents: # initalized to 0 in getAction
            agentNum = 0 # reset to pacman's turn
            depth += 1 # go to new layer in tree, initialized to 0 in getAction
        
        if depth == self.depth: # max depth reached
            return (Directions.STOP, self.evaluationFunction(gameState))
        
        if gameState.isWin() or gameState.isLose():
            return(Directions.STOP, self.evaluationFunction(gameState))

        optimal_score = None
        optimal_action = None
        p = 0

        if agentNum == 0: # pacman's turn
            pacmanActions = gameState.getLegalActions(agentNum)
            for action in pacmanActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.expectimax(depth, agentNum+1, suc_state)[1]
                if optimal_score is None or suc_score > optimal_score:
                    optimal_score = suc_score # maximize
                    optimal_action = action
        
        if agentNum != 0:
            agentActions = gameState.getLegalActions(agentNum)
            if len(agentActions) != 0:
                p = 1/len(agentActions)
            for action in agentActions:
                suc_state = gameState.generateSuccessor(agentNum, action)
                suc_score = self.expectimax(depth, agentNum+1, suc_state)[1] # next ghost's turn

                if optimal_score is None:
                    optimal_score = 0.0
                optimal_score += p * suc_score
                optimal_action = action
        
        if optimal_action is None: # no successor states => leaf state
            return (None, self.evaluationFunction(gameState))
        
        return (optimal_action, optimal_score)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        start_depth = 0
        start_agent_turn = 0
        optimal_action = self.expectimax(start_depth, start_agent_turn, gameState)[0]
        return optimal_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodList = newFood.asList()
    food_dist = sys.maxsize # distance to closest food from new position
    food_pos = newPos
    for food in foodList:  
        if food_dist > util.manhattanDistance(newPos, food):
            food_pos = food
            food_dist = util.manhattanDistance(newPos, food)

    ghost_dist = sys.maxsize
    ghost_pos = newPos
    for ghost in newGhostStates: # distance to closest ghost from new pos
        if ghost_dist > util.manhattanDistance(newPos, ghost.getPosition()):
            ghost_pos = ghost.getPosition()
            ghost_dist = util.manhattanDistance(newPos, ghost.getPosition())
            if ghost_dist <= 1:
                return -10000 # if ghost is 1 step or less away from pacman

    food_pos = (float(food_pos[0]), food_pos[1])
    dist_food_ghost = util.manhattanDistance(food_pos, ghost_pos) # distance from closest food to closest ghost

    remainingFood = currentGameState.getNumFood()
    remainingPellets = len(currentGameState.getCapsules())
    
    # .001 added to avoid division by zero error
    return currentGameState.getScore() + 1/(food_dist+.001) -\
           1/(ghost_dist+.001) - 1/(dist_food_ghost+.001) +\
           1/(remainingFood+.001) + 1/(remainingPellets+.001)

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
