'''
****************************************************************
*	Name: "Optimal path study on Gym FrozenLake-v0 environment"
****************************************************************
    Description
    In this project, FrozenLake-v0 environment from Gym OpenAI is utilized.
	The objective is to determine on the minimum list of actions that guide
	an agent from the starting grid at the top left corner to the goal grid
	(Frisbee) at the bottom right corner. Different  static and dynamic
	functions are proposed and their Performance are compared.

    The "main" function (at the very end) is the first function that is run
    in this program. It starts with running the Frozen Lake environment and
    followed by calling the following four functions.

        (1) Manual_Inspection.

        (2) Deﬁne_a_Function.

        (3) Extended_List.

        (4) Piecewise_Function.

    IMPORTANT TO RUN THE FUNCTIONS:
    Above four functions have been commented inside the "main" function and
    are ready to be uncommented and run individually.
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

'''
[ MAP:
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
'''


def func(env):
    '''
    It gets a render map from environment and return a list of tiles from 'S' to 'G'
    '''
    mapXY = env.render(mode='ansi')
    tileMap = []
    for i in range(len(mapXY)):
        row = mapXY[i][0]
        for j in range(len(row)):
            if row[j] == 'S' or row[j] == 'H' or row[j] == 'F' or row[j] == 'G':
                tileMap.append(row[j])
    return tileMap


def determine_nextPosition(position, map):
    '''
    For each current position, the next optimal position is obtained
    '''
    if position % 4 == 3:
        if map[position + 4] == 'H':
            newPosition = position - 1
        else:
            newPosition = position + 4
    elif 11 < position and position < 15:
        newPosition = position + 1
    else:
        if map[position + 4] == 'H':
            newPosition = position + 1
        else:
            newPosition = position + 4
    return newPosition


def directionToNextTile(currentPosition, newPosition):
    '''
    To get direction of a move using the current and new position
    '''
    if (newPosition - currentPosition) == -1:
        direction = 0
    if (newPosition - currentPosition) == 4:
        direction = 1
    if (newPosition - currentPosition) == 1:
        direction = 2
    return direction


def Plot(successCounter_Goal, failureCounter_Hole, failureCounter_StepsFinished, plotTitle):
    '''
    To plot a bar graph
    '''
    objects = ('Success', 'Failure')
    y_pos = np.arange(len(objects))
    performance = [successCounter_Goal, failureCounter_Hole + failureCounter_StepsFinished]
    plt.bar(y_pos, performance, align='center', alpha=0.5, color=['red', 'blue'])
    plt.xticks(y_pos, objects)
    plt.ylabel('Frequency of failures or successes')
    plt.title(plotTitle)
    plt.show()


def Manual_Inspection(env, optPathDirection):
    '''
    Inputs:
            env: 'FrozenLake-v0' environment
            optPathDirection: [1, 1, 2, 2, 1, 2]
            Note: The list includes an optimal direction path from "S" to "G".
    Functionality:
            Over 1000 runs, the optPathDirection steps are given to the environment and successes/failures are counted.
    Output:
            The success and failure rates are described and plotted with a bar graph.
            Note (1): Each episode is constraint to 6 moves which is the number of steps in the optimal path directions.
            Note (2): Success is achieved when the goal tile is reached.
            Note (3): Failures are due to either stepping into a hole or reaching to the maximum 6 steps limit.
    Conclusion:
        Result indicates a very low level of success rate (around 0.5%). For a comparison with the "Define a Function"
        method, please refer to the conclusion section at the beginning of that method.
    '''
    num_episodes = 1000
    successCounter_Goal = 0
    failureCounter_Hole = 0
    failureCounter_StepsFinished = 0
    for i_episode in range(num_episodes):
        # print('-----Episod ', i_episode, '---------')
        start = env.reset()
        for step in optPathDirection:
            observation, reward, done, info = env.step(step)
            ''' ---- Definition of the returned parameters  from the environment: ----
            "env.render()" shows location of the agent on the map
            "observation" is the tile number on the map (0 to 15)
            "reward" is 1.0 if observation locates at "G"; otherwise is 0.0
            "done" is "True" if observation locates at either "H" or "G"
            "info" provides probability level to slip
            '''
            if (done == True) and (reward == 1.0):
                successCounter_Goal += 1
                break
            if (done == True) and (reward == 0.0):
                failureCounter_Hole += 1
                break
        if (done == False): failureCounter_StepsFinished += 1
    # Print result and plot
    print('----- Results for "Manual Inspection" -----')
    print('Out of "', num_episodes, '" episodes:')
    print('Number of success to reach the goal is: ', successCounter_Goal)
    print('Number of trapping into the hole is: ', failureCounter_Hole)
    print('Number of unsuccessful attempts (episode steps finished before reaching the goal) is: ',
          failureCounter_StepsFinished)
    SuccessPercent = 100 * successCounter_Goal / (
                successCounter_Goal + failureCounter_Hole + failureCounter_StepsFinished)
    HoleFailurePercent = 100 * failureCounter_Hole / (
                successCounter_Goal + failureCounter_Hole + failureCounter_StepsFinished)
    UnfinishedFailurePercent = 100 * failureCounter_StepsFinished / (
                successCounter_Goal + failureCounter_Hole + failureCounter_StepsFinished)
    TotFailurePercent = HoleFailurePercent + UnfinishedFailurePercent
    print('The percentage of the successful times ended up with the frisbee = ', SuccessPercent, '%')
    print('The percentage of the times not ended up with the frisbee = ', TotFailurePercent, '%')
    print('Out of that failure rate,', HoleFailurePercent, '% stepped into a hole and', UnfinishedFailurePercent,
          '% reached to the 6 steps limit of the Manual search.')
    plotTitle = 'Manual Inspection: Success/Failure numbers'
    Plot(successCounter_Goal, failureCounter_Hole, failureCounter_StepsFinished, plotTitle)


def Deﬁne_a_Function(env):
    '''
    Inputs:
            env: 'FrozenLake-v0' environment
    Functionality:
            The goal grid is aimed over 1000 runs using a "best action" for each position to go to the next grid.
            To obtain the best action, "determine_nextPosition" function is used.
            Moving logic behind the "best action":
            Select the first move direction below. If it reaches to a hole or out-of-boundary location, go to the next:
                (1) go down
                (2) go right
                (3) go left
                (4) go top
            Successes/failures are counted.
    Output:
            The success and failure rates are described and plotted with a bar graph.
    Conclusion:
        Result indicates a low level of success rate (around 5%), but still 10 times more than the "Manual Inspection".
        The reason is because at each step the "Find a Function" method is able to provide the Frozen Lake environment
        with an optimal direction to the next grid, whereas in the "Manual Inspection" the steps through the initial
        minimal path is followed although the observed position has been slipped from where it was supposed to go.
    '''
    # Create a list from the map
    map = func(env)
    num_episodes = 1000
    successCounter_Goal = 0
    failureCounter_Hole = 0
    for i_episode in range(num_episodes):
        # print('-----Episod ', i_episode +1, '---------')
        start = env.reset()
        done = False
        step = 1
        while not done:
            observation, reward, done, info = env.step(step)
            if (done == True) and (reward == 1.0):
                successCounter_Goal += 1
                break
            if (done == True) and (reward == 0.0):
                failureCounter_Hole += 1
                break
            newPosition = determine_nextPosition(observation, map)
            # Calculate te direction based on the old and new grid positions
            step = directionToNextTile(observation, newPosition)
    print('----- Results for "Define a Function" -----')
    print('Out of "', num_episodes, '" episodes:')
    print('Number of success to reach the goal is: ', successCounter_Goal)
    print('Number of trapping into the hole is: ', failureCounter_Hole)
    SuccessPercent = 100 * successCounter_Goal / (successCounter_Goal + failureCounter_Hole)
    print('Percentage of success is = ', SuccessPercent)
    plotTitle = 'Define a Function: Success/Failure numbers'
    Plot(successCounter_Goal, failureCounter_Hole, 0, plotTitle)


def Extended_List(env, h_or_k_function_indicator):
    '''
    Inputs:
            env: 'FrozenLake-v0' environment
            h_or_k_function_indicator: is either True or False
             True is used for a complete "Extended List" run including printing and plotting
             False bypass the printing and plotting to provide the Piecewise_Function with only a minimalDirection list.
    Functionality:
            For each grid from 0 to 14, the minimal path is found. For that, the determine_nextPosition function used.
            For the minimal path moving logic, please refer to "Action" section from "Define a Function" method.
    Output:
            The minimal list of actions for each position are printed.
    '''
    map = func(env)
    minimalPath = [[] for i in range(15)]
    for tile in range(15):
        newPosition = tile
        minimalPath[tile].append(tile)
        while newPosition < 15:
            newPosition = determine_nextPosition(newPosition, map)
            if newPosition != 15: minimalPath[tile].append(newPosition)
        minimalPath[tile].append(15)
    minimalDirection = [[] for i in range(15)]
    for i in range(15):
        for j in range(len(minimalPath[i]) - 1):
            minimalDirection[i].append(directionToNextTile(minimalPath[i][j], minimalPath[i][j + 1]))
    # If below indicator is True, the h function minimal list is printed
    if h_or_k_function_indicator == True:
        direction = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'TOP'}
        print()
        print('----- "Extended List", minimal list of actions for each position -----')
        print()
        for i in range(15):
            print('h(', i, ') = ', end="")
            for j in range(len(minimalDirection[i])):
                print(direction[minimalDirection[i][j]], end=", ")
            print()
    # else the the h function minimal list is returned to be used in the Piecewise_Function
    else:
        return minimalDirection


def Piecewise_Function(env):
    '''
    Inputs:
            env: 'FrozenLake-v0' environment
    Functionality:
            The goal grid is aimed over 11000 times (1000 runs and 11 alpha). Alpha is {0, 0.1, ..., 1}.
            At each step of the communication with the Frozen Lake environment, alpha as a probability level determines
            either f ("Define a Function") or h ("Extended List") to be used. Selection process is as below.
                (1) A random generator runs and a provides a number in a [0, 1] range.
                (2) If the generated number is less than alpha, f is used. Otherwise, first action of h is used.
                Note: Practically, f and first action of h are the same!
            Successes/failures for 11 alpha are counted.
    Output:
            (1) The success and failure rates are described and plotted with bar graphs for each 11 alphas.
            (2) The success rates for all 11 alphas are described and plotted with a bar graph.
            (3) - Doing an investigation over different number of runs, the following statistical results are described:
                  For 10 runs:   average success rate is 4.54% with 6.56 standard deviation (Std)
                  For 100 runs:  average success rate is 3.91% with 1.44 Std
                  For 1000 runs: average success rate is 4.56% with 0.41 Std
                - Statistical results using normal distribution for 1000 runs:
                  68.27% (1 std) of the times, the success rate is 4.56% (+or- 0.41)
                  95.45% (2 stds) of the times, the success rate is 4.56% (+or- 0.82)
                  99.73% (3 stds) of the times, the success rate is 4.56% (+or- 1.23)
    Conclusion:
            (1) More number of runs, smaller Std, i.e. more reliable result (assuming Std as a reliability indicator).
            (2) The consistent and reliable success rate of 4.56 indicates that the "Piecewise Function" is not
            sensitive to the level of alpha. In other words, performance of f and h functions are the same.
            This is based on this fact that from the h function only its first action is used and practically both
            f and first action of h are the same as they both are in the direction of the minimal path to the goal.
    '''
    # From "Extended_List", a list with 15 element including the minimal list of actions for each position is imported.
    h_Funtion_Minimal_List = Extended_List(env, False)
    map = func(env)
    successRateList = []
    for i in range(11):
        alpha = i * 0.1
        num_episodes = 1000
        successCounter_Goal = 0
        failureCounter_Hole = 0
        f_function_counter = 0
        h_function_counter = 0
        for i_episode in range(num_episodes):
            # print('-----Episod ', i_episode +1, '---------')
            start = env.reset()
            done = False
            # First step of move from "S" tile for both f and h functions are the same toward "Down"
            step = 1
            while not done:
                observation, reward, done, info = env.step(step)
                if (done == True) and (reward == 1.0):
                    successCounter_Goal += 1
                    break
                if (done == True) and (reward == 0.0):
                    failureCounter_Hole += 1
                    break
                # Based on alpha, decide whether f or h functions are used
                if random.random() <= alpha:
                    # choose the next step using F function's logic
                    newPosition = determine_nextPosition(observation, map)
                    step = directionToNextTile(observation, newPosition)
                    f_function_counter += 1
                    # choose the next step using first element of the h function's minimal list
                else:
                    step = h_Funtion_Minimal_List[observation][0]
                    h_function_counter += 1
        tot = f_function_counter + h_function_counter
        print('----- Results for "Piecewise Function" for alpha = ', alpha, ' -----')
        print('F function used ', f_function_counter, ' times ( %', 100 * f_function_counter / tot, ')')
        print('H function used ', h_function_counter, ' times ( %', 100 * h_function_counter / tot, ')')
        print('Out of "', num_episodes, '" episodes:')
        print('Number of success to reach the goal is: ', successCounter_Goal)
        print('Number of trapping into the hole is: ', failureCounter_Hole)
        SuccessPercent = 100 * successCounter_Goal / (successCounter_Goal + failureCounter_Hole)
        print('Percentage of success is = ', SuccessPercent)
        plotTitle = 'Piecewise_Function (alpha = ' + str(alpha) + '): Success/Failure numbers'
        Plot(successCounter_Goal, failureCounter_Hole, 0, plotTitle)
        successRateList.append(SuccessPercent)
    # to obtain average and standard deviation of success rates starting from each tile
    print('----- "Piecewise Function" Statistical Results for alpha {0, 0.1, 0.2, . . . , 1} over 1000 runs -----')
    print(successRateList)
    alphaIntervals = []
    for i in range(11):
        alphaIntervals.append(i * 0.1)
    X = np.arange(11)
    plt.bar(alphaIntervals, successRateList, color='r', width=0.03)
    plt.xlabel('Alpha probability Level')
    plt.ylabel('Successful percentage over 1000 run')
    plt.title('"Success Rate - Alpha" Diagram')
    plt.show()
    successRateAverage = np.average(successRateList)
    successRateSTD = np.std(successRateList)
    print(' The Average Success Rate is :', successRateAverage)
    print(' The Standard Deviation of the Success Rate is :', successRateSTD)


def main():
    '''
    The "main" function is the first function that is run in this program.
    It starts with running the Frozen Lake environment and followed by calling four functions as four demanded
    sections of the assignment. They have been commented and ready to be uncommented and run individually.
    '''
    env = gym.make('FrozenLake-v0')
    # Uncomment each of the following calling methods to run the respective section of the assignment
    # Manual_Inspection(env, [1, 1, 2, 2, 1, 2])
    #Deﬁne_a_Function(env)
    # Extended_List(env, True)
    Piecewise_Function(env)

    '''
    This is the starting point of the program.
    '''


if __name__ == '__main__':
    main()



