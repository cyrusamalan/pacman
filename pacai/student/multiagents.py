import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
import sys


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***

        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        foodPositions = currentGameState.getFood().asList()
        newPosition = successorGameState.getPacmanPosition()

        # Compute Manhattan distances
        manhattan_distance = lambda goal: distance.manhattan(newPosition, goal)

        ghostDistances = [manhattan_distance(ghost) for ghost in ghostPositions]
        foodDistances = [manhattan_distance(food) for food in foodPositions]

        # estimate of closest ghost
        closestGhostDist = max(min(ghostDistances, default=float('inf')), 0.001)

        # Find closest food
        if foodPositions:
            closestFood = min(foodPositions, key=manhattan_distance)
            closestFoodDist = max(distance.maze(newPosition, closestFood, currentGameState), 0.001)
        else:
            closestFoodDist = 0.001

        # Return heuristic score
        return (1 - (1 / closestGhostDist)) + (1 / closestFoodDist)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):
        numAgents = gamestate.getNumAgents()
        
        def minimax(state, agent, depth, prevAction):
            if state.isOver() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state), prevAction

            legalActions = state.getLegalActions(agent)
            if not legalActions:  # return evaluation
                return self.getEvaluationFunction()(state), 'STOP'

            isMaximizing = (agent == 0)
            bestValue = float('-inf') if isMaximizing else float('inf')
            bestAction = 'STOP'

            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                nextAgent = (agent + 1) % numAgents
                nextDepth = depth + (nextAgent == 0)  # Increase depth
                value, _ = minimax(successor, nextAgent, nextDepth, action)

                if isMaximizing:  # choose max value
                    if value > bestValue:
                        bestValue, bestAction = value, action
                else:  # choose min value
                    if value < bestValue:
                        bestValue, bestAction = value, action

            return bestValue, bestAction

        _, action = minimax(gamestate, 0, 0, 'STOP')
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        evalfn = self.getEvaluationFunction()
        num_agents = gameState.getNumAgents()
        tree_depth = self.getTreeDepth()

        def max_value(state, agent_idx, depth, alpha, beta):
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            v = -(sys.maxsize - 1)
            if agent_idx == 0:
                actions = state.getLegalActions(agent_idx)
                actions.remove('Stop')
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = max(v, min_value(next_state, agent_idx + 1, depth, alpha, beta))
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
                return v

        def min_value(state, agent_idx, depth, alpha, beta):
            v = sys.maxsize
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            if agent_idx == (num_agents - 1):
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, max_value(next_state, 0, depth + 1, alpha, beta))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
                return v
            else:
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, min_value(next_state, agent_idx + 1, depth, alpha, beta))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
                return v

        pacman_moves = []
        actions = gameState.getLegalActions()
        actions.remove('Stop')
        ninf = -(sys.maxsize - 1)
        pinf = sys.maxsize
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            agent_idx = next_state.getLastAgentMoved()
            item = (action, min_value(next_state, agent_idx + 1, 0, ninf, pinf))
            pacman_moves.append(item)
        action_to_take = max(pacman_moves, key=lambda pacman_moves: pacman_moves[1])
        return action_to_take[0]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):
        numAgents = gamestate.getNumAgents()
        def expectimax(state, agent, depth):
            if state.isOver() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state), None

            legalActions = state.getLegalActions(agent)
            if not legalActions:  # No moves
                return self.getEvaluationFunction()(state), None

            if agent == 0:
                return max(
                    (expectimax(state.generateSuccessor(agent, action), (agent + 1) % numAgents, depth + (agent + 1 == numAgents))[0], action)
                    for action in legalActions
                )

            else:
                values = [expectimax(state.generateSuccessor(agent, action), (agent + 1) % numAgents, depth + (agent + 1 == numAgents))[0] 
                        for action in legalActions]
                return sum(values) / len(values), random.choice(legalActions)

        _, action = expectimax(gamestate, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    ghostPositions = [ghost.getPosition() for ghost in ghostStates]
    scaredTimes = [ghost.getScaredTimer() for ghost in ghostStates]

    score = currentGameState.getScore()

    # Compute distance
    if foodPositions:
        closestFoodDist = min(distance.manhattan(pacmanPosition, food) for food in foodPositions)
        foodScore = 1 / (closestFoodDist + 1)
    else:
        foodScore = 0  # No food left

    # Compute distance
    if ghostPositions:
        closestGhostDist = min(distance.manhattan(pacmanPosition, ghost) for ghost in ghostPositions)
        if closestGhostDist == 0:
            ghostScore = -float('inf')  
        else:
            ghostScore = -1 / closestGhostDist  
    else:
        ghostScore = 0 

    # chase scared ghosts
    scaredScore = 0
    for i in range(len(ghostStates)):
        if scaredTimes[i] > 0:
            scaredGhostDist = distance.manhattan(pacmanPosition, ghostPositions[i])
            if scaredGhostDist > 0:
                scaredScore += 5 / scaredGhostDist  # Encourage hunting vulnerable ghosts

    # remaining food pellets
    foodCountScore = -len(foodPositions)  # Fewer food pellets left = better

    return score + foodScore + ghostScore + scaredScore + foodCountScore

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)