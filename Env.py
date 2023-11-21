# Import routines
import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

""" 
Assumptions:

1. The taxis are electric cars. It can run for 30 days non-stop, i.e., 24*30 hrs. Then it needs to
recharge itself. If the cab driver is completing his trip at that time, he’ll finish that trip and
then stop for recharging. So, the terminal state is independent of the number of rides
covered in a month, it is achieved as soon as the cab driver crosses 24*30 hours.

2. There are only 5 locations in the city where the cab can operate.

3. All decisions are made at hourly intervals. We won’t consider minutes and seconds for this
project. So for example, the cab driver gets requests at 1:00 pm, then at 2:00 pm, then at
3:00 pm and so on. So, he can decide to pick among the requests only at these times. A
request cannot come at (say) 2.30 pm.

4. The time taken to travel from one place to another is considered in integer hours (only) and
is dependent on the traffic. Also, the traffic is dependent on the hour of the day and the
day of the week."""

class CabDriver():
    

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        """Action Space:There’ll never be requests of the sort where pickup and drop locations are the same. So, the action
           space A will be: (𝑚 − 1) ∗ 𝑚 + 1 for m locations. Each action will be a tuple of size 2. Action space 
           is defined as below:
               • pick up and drop locations (𝑝, 𝑞) where p and q both take a value between 1 and m;
               • (0, 0) tuple that represents ’no-ride’ action."""
        
        """The state space is defined by the driver’s current location along with the time components (hour of
           the day and the day of the week). A state is defined by three variables:
           𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘 𝑤ℎ𝑒𝑟𝑒 𝑖 = 1 … 𝑚; 𝑗 = 0 … . 𝑡 − 1; 𝑘 = 0 … . . 𝑑 − 1
           Where 𝑋𝑖 represents a driver’s current location, 𝑇𝑗 represents time component (more specifically
           hour of the day), 𝐷𝑘 represents the day of the week
               • Number of locations: m = 5
               • Number of hours: t = 24
               • Number of days: d = 7"""
        
        self.action_space = list(permutations([i for i in range(m)], 2)) + [(0,0)] ## All permutaions of the actions and no action
        
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)] ##
        self.state_init = random.choice(self.state_space)
        self.reset()


    ## Encoding state & action for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for x in range (m+t+d)]  # initialize vector state
        state_encod[state[0]] = 1  # set the location value into vector
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1 # set time value into vector
        return state_encod

    def action_encod_arch1(self, action):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for x in range (m+m)]  # initialize the vector state
        if action[0] != 0 and action[1] != 0:
            state_encod[action[0]] = 1     # set pickup location:
            state_encod[m + action[1]] = 1     # set drop location
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])

        # Update index for [0, 0]

        possible_actions_index.append(self.action_space.index((0,0)))

        return possible_actions_index,actions


    def reward_func(self, state, action, Time_matrix):
       
        """Takes in state, action and Time-matrix and returns the reward"""
        
        """the reward function will be 
        (revenue earned from pickup point 𝑝 to drop point 𝑞) 
        - (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) 
        - (Cost of battery used in moving fromcurrent point 𝑖 to pick-up point 𝑝). 
        
        Mathematically,
        𝑅(𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘) = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝)) 𝑎 = (𝑝, 𝑞) 
                                          - 𝐶𝑓 𝑎 = (0,0)
        Where,
        𝑋𝑖 represents a driver’s current location, 
        𝑇𝑗 represents time component (more specifically hour of the day), 
        𝐷𝑘 represents the day of the week, 
        𝑝 represents the pickup location and 
        𝑞 represents the drop location."""
        
        # We need to find the next state and then calculate reward for the next state
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        revenue_time = ride_time
        idle_time = wait_time + transit_time
        reward = (R * revenue_time) - (C * (revenue_time + idle_time))

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        # Find current state of driver
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]

        # reward depends of time, lets initialize
        total_time = 0 # initiating with 0
        wait_time = 0 # initiating with 0
        ride_time = 0 # initiating with 0
        transit_time = 0 # initiating with 0

        # next state depends on possible actions taken by cab driver
        # There are 3 cases for the same
        # 1. Cab driver refuse a ride i.e. action (0,0)
        if (pickup_loc) == 0 and (drop_loc == 0):
            wait_time = 1
            next_loc = curr_loc

        # 2. Driver wants to have pickup and is at same location
        elif pickup_loc == curr_loc:
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_loc = drop_loc

        # 3. Driver pickup & current location is same
        else:
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            updated_time, updated_day = self.get_updated_day_time(curr_time, curr_day, transit_time)
            ride_time =  Time_matrix[pickup_loc][drop_loc][updated_time][updated_day]
            next_loc  = drop_loc
            curr_time = updated_time
            curr_day = updated_day

        total_time = ride_time + wait_time
        updated_time, updated_day = self.get_updated_day_time(curr_time, curr_day, total_time)
        next_state = [next_loc, updated_time, updated_day]

        return next_state, wait_time, transit_time, ride_time

    def get_updated_day_time(self, time, day, duration):
        '''
        Takes the current day, time and time duration and returns updated day and time based on the time duration of travel
        '''
        if time + int(duration) < 24:     # The day is not changed
            updated_time = time + int(duration)
            updated_day = day
        else:
            updated_time = (time + int(duration)) % 24
            days_passed = (time + int(duration)) // 24
            updated_day = (day + days_passed ) % 7
        return updated_time, updated_day


    """Reset state, action space and initial values"""
    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init


c = CabDriver()