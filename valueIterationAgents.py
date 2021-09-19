# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # using 's' for states and 'a' for actions
        for i in range(self.iterations):
            # instructions said use a counter, so we'll use a counter
            vCounter = util.Counter()
            for s in self.mdp.getStates():
                # some unreasonably low score
                maxFound = -1000000
                # find the maximum value from all possible actions
                for a in self.mdp.getPossibleActions(s):
                    q = self.computeQValueFromValues(s, a)
                    maxFound = max(maxFound, q)
                    vCounter[s] = maxFound
            # assign the max found values to our self.values
            self.values = vCounter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        runningTotal = 0
        for transState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, transState)
            value = self.getValue(transState)
            runningTotal += prob * ((self.discount * value) + reward)
        return runningTotal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # first, check if there are any possible actions
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        else:
            maxFound = -1000000
            foundAction = None
            for a in actions:
                temp = maxFound
                maxFound = max(maxFound, self.computeQValueFromValues(state, a))
                # if maxFound changed, then update the bestIndex
                if maxFound > temp:
                    foundAction = a
            return foundAction
                

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        s = self.mdp.getStates()
        count = len(s)
        for i in range(self.iterations):
            state = s[i % count]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(s)
                # find maximum value from all q values
                maxFound = float("-inf")
                for a in actions:
                    maxFound = max(maxFound, self.computeQValueFromValues(state, a))
                self.values[state] = maxFound

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pQueue = util.PriorityQueue()
        preds = {}
        states = self.mdp.getStates()
        
        # based on the actions of every state,
        # add them to preds in appropriate fashion
        for s in states:
            if not self.mdp.isTerminal(s):
                for a in self.mdp.getPossibleActions(s):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                        if nextState in preds:
                            preds[nextState].add(s)
                        else:
                            preds[nextState] = {s}
        
        # now, for each state find te Q values and add them to pQueue
        for s in states:
            if not self.mdp.isTerminal(s):
                maxFound = -1000000
                #find max q value
                for a in self.mdp.getPossibleActions(s):
                    maxFound = max(maxFound, self.computeQValueFromValues(s, a))
                # now, find negative abs value of difference
                diff = abs(maxFound - self.values[s])
                pQueue.update(s, -diff)
        
        for i in range(self.iterations):
            if pQueue.isEmpty():
                break
            s = pQueue.pop()
            if not self.mdp.isTerminal(s):
                maxFound = -1000000
                for a in self.mdp.getPossibleActions(s):
                    maxFound = max(maxFound, self.computeQValueFromValues(s, a))
                self.values[s] = maxFound
            
            for p in preds[s]:
                if not self.mdp.isTerminal(p):
                    maxFound = -1000000
                    for a in self.mdp.getPossibleActions(p):
                        maxFound = max(maxFound, self.computeQValueFromValues(p, a))
                    # now, find negative abs value of difference
                    diff = abs(maxFound - self.values[p])
                    if diff > self.theta:
                        pQueue.update(p, -diff)
        
        
        
        
        
        
        
        
        