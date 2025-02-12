import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

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
        food = currentGameState.getFood().asList()
        newPosition = successorGameState.getPacmanPosition()

        # Compute Manhattan distances
        manhattan = lambda goal: distance.manhattan(newPosition, goal)

        ghost_dist = [manhattan(ghost) for ghost in ghostPositions]
        # foodDistances = [manhattan_distance(food) for food in foodPositions]

        # estimate of closest ghost
        ghost = max(min(ghost_dist, default=float('inf')), 0.001)

        # Find closest food
        if food:
            closestFood = min(food, key=manhattan)
            food_dist = max(distance.maze(newPosition, closestFood, currentGameState), 0.001)
        else:
            food_dist = 0.001

        # Return score
        return (1 - (1 / ghost)) + (1 / food_dist)


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
        num_agents = gamestate.getNumAgents()

        def minimax(state, agent, depth):
            if state.isOver() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state), None

            leg_action = state.getLegalActions(agent)
            if not leg_action:
                return self.getEvaluationFunction()(state), None

            maxim = (agent == 0)

            if maxim:  # Maximize
                return max(
                    (
                        minimax(
                            state.generateSuccessor(agent, next),
                            (agent + 1) % num_agents,
                            depth + (agent + 1 == num_agents)
                        )[0], next
                    )
                    for next in leg_action
                )
            else:  # Minimize
                return min(
                    (
                        minimax(
                            state.generateSuccessor(agent, next),
                            (agent + 1) % num_agents,
                            depth + (agent + 1 == num_agents)
                        )[0], next
                    )
                    for next in leg_action
                )

        _, next = minimax(gamestate, 0, 0)
        return next

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
        num_agent = gameState.getNumAgents()

        def alphabeta(state, agent, depth, alpha, beta):
            if state.isOver() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state), None

            leg_action = state.getLegalActions(agent)
            if agent == 0 and 'Stop' in leg_action:  # Make sure it doesn't stop
                leg_action.remove('Stop')

            if not leg_action:
                return self.getEvaluationFunction()(state), None

            # Move ordering
            leg_action.sort(
                key=lambda action: self.getEvaluationFunction()(
                    state.generateSuccessor(agent, action)
                ),
                reverse=(agent == 0)  # Sort descending
            )

            if agent == 0:  # Maximizing
                value, bestAction = float('-inf'), None
                for action in leg_action:
                    successor = state.generateSuccessor(agent, action)
                    newValue, _ = alphabeta(
                        successor,
                        (agent + 1) % num_agent,
                        depth + (agent + 1 == num_agent),
                        alpha, beta
                    )
                    if newValue > value:
                        value, bestAction = newValue, action
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # Prune
                return value, bestAction

            else:  # Ghosts (Minimizing)
                value = float('inf')
                for next in leg_action:
                    successor = state.generateSuccessor(agent, next)
                    newValue, _ = alphabeta(
                        successor,
                        (agent + 1) % num_agent,
                        depth + (agent + 1 == num_agent),
                        alpha, beta
                    )
                    value = min(value, newValue)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # Prune
                return value, None

        _, next = alphabeta(gameState, 0, 0, float('-inf'), float('inf'))
        return next

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
        num_agents = gamestate.getNumAgents()

        def expectimax(state, agent, depth):
            if state.isOver() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state), None

            leg_action = state.getLegalActions(agent)
            if not leg_action:  # No moves
                return self.getEvaluationFunction()(state), None

            if agent == 0:
                return max(
                    (
                        expectimax(
                            state.generateSuccessor(agent, next),
                            (agent + 1) % num_agents,
                            depth + (agent + 1 == num_agents)
                        )[0], next
                    )
                    for next in leg_action
                )
              
            else:
                values = [
                    expectimax(
                        state.generateSuccessor(agent, next),
                        (agent + 1) % num_agents,
                        depth + (agent + 1 == num_agents)
                    )[0]
                    for next in leg_action
                ]
                return sum(values) / len(values), random.choice(leg_action)

        _, next = expectimax(gamestate, 0, 0)
        return next

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
        closestFoodDist = min(distance.manhattan(pacmanPosition, food)for food in foodPositions)
        foodScore = 1 / (closestFoodDist + 1)
    else:
        foodScore = 0  # No food left

    # Compute distance
    if ghostPositions:
        closestGhostDist = min(
            distance.manhattan(pacmanPosition, ghost)
            for ghost in ghostPositions
        )
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
                scaredScore += 5 / scaredGhostDist

    # remaining food pellets
    foodCountScore = -len(foodPositions)  # Fewer food pellets

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