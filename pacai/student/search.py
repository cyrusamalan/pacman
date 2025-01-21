"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def reconstruct_path(parent):
    """Reconstruct the path from goal to start using parent relationships."""
    path = []
    while parent:
        curr, parent, event = parent
        path.append(event)
        parent = parent
    return path

# WORKING

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    top = Stack()
    visited = set()
    context = []

    # Initialize the stack 
    start_state = problem.startingState()
    top.push((start_state, None, None))
    visited.add(start_state)

    while not top.isEmpty():
        curr, parent_comp, prev = top.pop()

        # reconstruct the path
        if problem.isGoal(curr):
            context = reconstruct_path((curr, parent_comp, prev))
            context.reverse()
            break

        # Process successors
        for successor, event, cost in problem.successorStates(curr):
            if successor not in visited:
                top.push((successor, (curr, parent_comp, prev), event))
                visited.add(successor)

    if context and context[0] is None:
        context.pop(0)

    return context

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # initialize structures
    queue = []  
    visited = [] 
    path = []

    # Enqueue the start state
    queue.append((problem.startingState(), None, None))  
    visited.append(problem.startingState())  

    while queue:
        curr, parent_comp, prev = queue.pop(0)  

        # If the goal is reached, reconstruct the path
        if problem.isGoal(curr):
            path.append(prev)
            while parent_comp:
                curr, parent, next = parent_comp
                path.append(next)
                parent_comp = parent
            break

        # Process successor states
        for successor, next, cost in problem.successorStates(curr):
            if successor not in visited:  
                queue.append((successor, (curr, parent_comp, prev), next)) 
                visited.append(successor)  

    # Reverse the path
    path.reverse()

    if path and path[0] is None:
        path.pop(0)

    return path

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # initialize structures
    p_queue = PriorityQueue()
    visited = set()
    context = []

    # Initialize the priority queue 
    start = problem.startingState()
    p_queue.push((start, None, None, 0), 0)
    visited.add(start)

    while not p_queue.isEmpty():
        current, parent_comp, prev, curr_cost = p_queue.pop()

        # reconstruct the path
        if problem.isGoal(current):
            context = reconstruct_path((current, parent_comp, prev))
            context.reverse()
            break

        # Process successors
        for successor, next, step_cost in problem.successorStates(current):
            if successor not in visited:
                total_cost = curr_cost + step_cost
                p_queue.push((successor, (current, parent_comp, prev), next, total_cost), total_cost)
                visited.add(successor)

    if context and context[0] is None:
        context.pop(0)

    return context

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # initialize structures
    p_queue = PriorityQueue()
    visited = set()
    events = []

    # Initialize the priority queue
    start_state = problem.startingState()
    p_queue.push((start_state, None, None, 0), heuristic(start_state, problem))
    visited.add(start_state)

    while not p_queue.isEmpty():
        current, parent_comp, prev, curr_cost = p_queue.pop()

        # reconstruct the path
        if problem.isGoal(current):
            events = reconstruct_path((current, parent_comp, prev))
            events.reverse()
            break

        # Process successors
        for successor, next, step_cost in problem.successorStates(current):
            if successor not in visited:
                total_cost = curr_cost + step_cost
                priority = total_cost + heuristic(successor, problem)
                p_queue.push((successor, (current, parent_comp, prev), next, total_cost), priority)
                visited.add(successor)

    if events and events[0] is None:
        events.pop(0)

    return events
