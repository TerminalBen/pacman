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


from pacman import GameState
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
        #dealing with food
        foodPos = newFood.asList() #list of pellets in board
        foodcount = len(foodPos) #number of pellets in board
        distance2food = float('inf') #bignumber here.
        for i in range(foodcount):
            distance = manhattanDistance(foodPos[i],newPos)+foodcount*100#use manhattan distance functin provided to calculate distance to next food
            if distance<distance2food:#compare thedistance to food to the stored distance
                distance2food = distance
        if foodcount == 0: #finish loop
            distance2food =0
        score = -distance2food #init score

        if action == "Stop":#we dont want to stop?(Time issues)
            score -=float('inf')
        print(score)

        #dealing with ghosts
        for i in range(len(newGhostStates)):#look for all of the ghost states in the game
            ghostPos = successorGameState.getGhostPosition(i+1)#gets their position
            if manhattanDistance(newPos,ghostPos)<=1:#compares the distance
                score -= float('inf') #runs away
        #return successorGameState.getScore()
        return score


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

    def getAction(self, gameState):
        """
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
        #format = [score,action]
        result = self.get_value(gameState,0,0)
        return result[1] #return action

    def get_value(self,gameState,index, depth):
        """Returns a value pair of[score,action] based on 1:terminaState 2:max_agent 3:min_agent


        Args:
            gameState ([type]): [description]
            index ([type]): [description]
            depth ([type]): [description]

        Returns:
            [type]: [description]
        """
        #return [score,action]
        if len(gameState.getLegalActions(index)) ==0 or depth ==self.depth:
            return[gameState.getScore(),""]
        #maxagent: Pacman has index =0
        if index ==0:
            return self.max_value(gameState,index,depth)
        #minagent ghost has index!=0
        else:
            return self.min_value(gameState,index,depth)
    
    def max_value(self,game_state,index,depth):
        """Returns the max of the pair [score,action]for maxagent

        Args:
            gameState ([type]): [description]
            index ([type]): [description]
            depth ([type]): [description]
        """
        legalMoves = game_state.getLegalActions(index)
        max_value = float('-inf')
        max_action =""
        for action in legalMoves:
            successor = game_state.generateSuccessor(index,action)
            successor_index = index+1
            successor_depth = depth

            #if agent is pacman update agent index and depth
            if successor_index == game_state.getNumAgents():
                successor_index =0
                successor_depth+=1
            
            currValue = self.get_value(successor,successor_index,successor_depth)[0]

            if currValue > max_value:
                max_value = currValue
                max_action = action
        return max_value,max_action

    def min_value(self,game_state,index,depth):
        """Returns the min of the pair [score,action]for minagent

        Args:
            gameState ([type]): [description]
            index ([type]): [description]
            depth ([type]): [description]
        """
        legalMoves = game_state.getLegalActions(index)
        min_value = float('inf')
        min_action =""
        for action in legalMoves:
            successor = game_state.generateSuccessor(index,action)
            successor_index = index+1
            successor_depth = depth

            #if agent is pacman update agent index and depth
            if successor_index == game_state.getNumAgents():
                successor_index =0
                successor_depth+=1
            
            currValue = self.get_value(successor,successor_index,successor_depth)[0]

            if currValue < min_value:
                min_value = currValue
                min_action = action
        return min_value,min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #init state : index=0,dept=0,alpha=-inf,beta=inf
        output = self.getBestOption(game_state,0,0,float('-inf'),float('inf'))

        return output[0]

    def getBestOption(self, game_state,index, depth,alpha,beta):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """    
        #terminal States
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "",game_state.getScore()
        #max agent--------------Pacman has index=0
        if index==0:
            return self.maxValue(game_state,index,depth,alpha,beta)
        #min agent--------------ghosts has index>0
        else:
            return self.minValue(game_state,index,depth,alpha,beta)


    def maxValue(self,game_state,index,depth,alpha,beta):
        #max value of [action,score] for maximizing agent using pruning

        legalMoves = game_state.getLegalActions(index) 
        maxValue = float('-inf')
        maxAction = ''

        for action in legalMoves:
            successor = game_state.generateSuccessor(index,action)
            successor_index = index+1
            successor_depth = depth

            #update the agent sucessor if it is pacman
            if successor_index == game_state.getNumAgents():
                successor_index =0
                successor_depth+=1

            #calculate the action-score of the successor
            current_action,current_value = self.getBestOption(successor,successor_index,successor_depth,alpha,beta)

            # Update max_value and max_action for maximizer agent
            if current_value > maxValue:
                maxValue = current_value
                maxAction = action   #double check this later

            #update alpha valuefor the current maximizer
            alpha = max(alpha,maxValue)

            #prunning: Return maxValue because next possilbe max values of the maximizer can be worse
            #can be worse for beta value of minimizer when coming back up
            if maxValue >beta:
                return maxAction,maxValue

        return maxAction,maxValue

    
    def minValue(self,game_state,index,depth,alpha,beta):
        #minValue of [action,score] of the minimizing agent using prunning
        legalMoves = game_state.getLegalActions(index) 
        minValue = float('inf')
        minAction = ''

        for action in legalMoves:
            successor = game_state.generateSuccessor(index,action)
            successor_index = index+1
            successor_depth = depth

            #update the agent sucessor if it is pacman
            if successor_index == game_state.getNumAgents():
                successor_index =0
                successor_depth+=1

            #calculate the action-score of the successor
            current_action,current_value = self.getBestOption(successor,successor_index,successor_depth,alpha,beta)

            # Update min_value and minaction for minimizer agent
            if current_value < minValue:
                minValue = current_value
                minAction = action   #double check this later

            #update beta valuefor the current minimizer
            beta = min(beta,minValue)

            #prunning: Return maxValue because next possilbe max values of the maximizer can be worse
            #can be worse for beta value of minimizer when coming back up
            if minValue < alpha:
                return minAction,minValue

        return minAction,minValue







class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
