import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        # What heapq.heappush does:
            # Takes the entry tuple and inserts it into the self.Heap list while maintaining the heap property. This property ensures that the element with the highest priority (lowest numerical value in the first element of the tuple) is always at the beginning of the list.
            # To achieve this, heapq.heappush performs operations like comparisons and swaps within the list to ensure the correct order based on priorities.
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        # What heapq.heappop does:
            # Removes the element: It extracts the element with the smallest value (highest priority in this case) from the beginning of the self.Heap list.
            # Maintains the heap property: After removing the element, heapq.heappop rearranges the remaining elements in the list to uphold the min-heap property. This ensures the element with the next smallest value becomes the new root (at index 0).
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    # Input: gameState: trạng thái khởi đầu của màn chơi
    # Output: lời giải của màn chơi

    beginBox = PosOfBoxes(gameState)    # vị trí khởi đầu của box
    beginPlayer = PosOfPlayer(gameState)    # vị trí khởi đầu của player

    # trạng thái bài toán = ( (vị trí agent / player), (vị trí các box) )
    startState = (beginPlayer, beginBox)    # trạng thái khởi đầu

    # hàng đợi chứa các node sắp đuọc bung ra
    # mỗi node chứa các trạng thái theo thứ tự tạo thành một lời giải / solution
    # mỗi node là một list chứa các list con, mỗi list con chứa 1 trạng thái
    frontier = collections.deque([[startState]])

    exploredSet = set() # set chứa trạng thái đã duyệt qua
    actions = [[0]] # list chứa list các action của các solution
    temp = []   # return list

    while frontier:
        # lấy node bên phải frontier / node mới nhất trong frontier / node sâu nhất
        node = frontier.pop()
        
        # lấy list action ứng với node vừa lấy
        node_action = actions.pop()

        # tt đang duyệt là end state
        if isEndState(node[-1][-1]):
            temp += node_action[1:] # list action -> temp
            break
        # end_if

        # tt đang duyệt chưa đc duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            # duyệt qua từng legal action của tt hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # tt mới sau khi thực hiện action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                
                # tt mới có thể không dẫn tới solution
                if isFailed(newPosBox):
                    continue    # bỏ qua
                # end_if

                # thêm tt mới vào node rồi thêm node vào bên phải frontier
                frontier.append(node + [(newPosPlayer, newPosBox)])

                # thêm action mới thực hiện vào list action rồi thêm vào list list actions
                actions.append(node_action + [action[-1]])
            # end_for
        # end_if
    # end_while
    
    return temp, len(exploredSet)+1
# end_def

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    # Input: gameState: trạng thái khởi đầu của màn chơi
    # Output: lời giải của màn chơi

    beginBox = PosOfBoxes(gameState)    # vị trí khởi đầu của box
    beginPlayer = PosOfPlayer(gameState)    # vị trí khởi đầu của player

    # trạng thái bài toán = ( (vị trí agent / player), (vị trí các box) )
    startState = (beginPlayer, beginBox)    # trạng thái khởi đầu

    # hàng đợi chứa các node sắp đuọc bung ra
    # mỗi node chứa các trạng thái theo thứ tự tạo thành một lời giải / solution
    # mỗi node là một list chứa các list con, mỗi list con chứa 1 trạng thái
    frontier = collections.deque([[startState]])

    exploredSet = set() # set chứa trạng thái đã duyệt qua
    actions = collections.deque([[0]]) # list chứa list các action của các solution
    temp = []   # return list

    ### CODING FROM HERE ###
    while frontier:
        # lấy node bên trái frontier / node cũ nhất trong frontier / node tiếp theo trên level hiện tại
        node = frontier.popleft()
        
        # lấy list action ứng với node vừa lấy
        node_action = actions.popleft()

        # tt đang duyệt là end state
        if isEndState(node[-1][-1]):
            temp += node_action[1:] # list action -> temp
            break
        # end_if

        # tt đang duyệt chưa đc duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            # duyệt qua từng legal action của tt hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # tt mới sau khi thực hiện action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                
                # tt mới có thể không dẫn tới solution
                if isFailed(newPosBox):
                    continue    # bỏ qua
                # end_if

                # thêm tt mới vào node rồi thêm node vào bên phải frontier
                frontier.append(node + [(newPosPlayer, newPosBox)])

                # thêm action mới thực hiện vào list action rồi thêm vào list list actions
                actions.append(node_action + [action[-1]])
            # end_for
        # end_if
    # end_while
    
    return temp, len(exploredSet)+1
# end_def

def cost(actions):
    """A cost function"""
    # cost = số action không di chuyển box
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    # Input: gameState: trạng thái khởi đầu của màn chơi
    # Output: lời giải của màn chơi

    beginBox = PosOfBoxes(gameState)    # vị trí khởi đầu của box
    beginPlayer = PosOfPlayer(gameState)    # vị trí khởi đầu của player

    # trạng thái bài toán = ( (vị trí agent / player), (vị trí các box) )
    startState = (beginPlayer, beginBox)    # trạng thái khởi đầu
    
    # hàng đợi chứa các node sắp đuọc bung ra
    # mỗi node chứa các trạng thái theo thứ tự tạo thành một lời giải / solution
    # mỗi node là một list chứa các list con, mỗi list con chứa 1 trạng thái
    frontier = PriorityQueue()
    frontier.push([startState], 0)

    exploredSet = set() # set chứa trạng thái đã duyệt qua

    # list chứa list các action của các solution
    actions = PriorityQueue()
    actions.push([0], 0)

    temp = []   # return list
    ### CODING FROM HERE ###
    while len(frontier.Heap) > 0:
        # lấy node có cost thấp nhất frontier
        node = frontier.pop()
        
        # lấy list action ứng với node vừa lấy
        node_action = actions.pop()

        # tt đang duyệt là end state
        if isEndState(node[-1][-1]):
            temp += node_action[1:] # list action -> temp
            break
        # end_if

        # tt đang duyệt chưa đc duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            # duyệt qua từng legal action của tt hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # tt mới sau khi thực hiện action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                
                # tt mới có thể không dẫn tới solution
                if isFailed(newPosBox):
                    continue    # bỏ qua
                # end_if

                new_node = node + [(newPosPlayer, newPosBox)]   # thêm tt mới vào node
                new_action = node_action + [action[-1]] # thêm action mới thực hiện vào list action
                
                # tính cost
                new_cost = cost(new_action[1:])

                # thêm node vào bên phải frontier
                frontier.push(new_node, new_cost)

                # thêm vào list list actions
                actions.push(new_action, new_cost)
            # end_for
        # end_if
    # end_while
    
    return temp, len(exploredSet)+1
# end_def

def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    # input:
        # posPlayer: vị trí player
        # posBox: tuple vị trí các box
    # output:
        # distance: distance between the else boxes and the else goals
    
    distance = 0

    # This set stores the boxes that are already on their goals
    completes = set(posGoals) & set(posBox)

    # This list holds the positions of remaining "else boxes" (elements in posBox that are not in completes)
    sortposBox = list(set(posBox).difference(completes))

    # This list holds the goal positions for the remaining "else boxes" (elements in posGoals that are not in completes)
    sortposGoals = list(set(posGoals).difference(completes))

    # The code iterates through sortposBox (same length as sortposGoals)
    for i in range(len(sortposBox)):
        # For each box and its corresponding goal at the same index:
            # The absolute difference between x-coordinates is calculated.
            # The absolute difference between y-coordinates is calculated.
            # Both differences are added to the distance variable.
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    # end_for
    
    return distance
# end_def

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    # Input: gameState: trạng thái khởi đầu của màn chơi
    # Output: lời giải của màn chơi

    # start =  time.time()
    beginBox = PosOfBoxes(gameState)    # vị trí khởi đầu của box
    beginPlayer = PosOfPlayer(gameState)    # vị trí khởi đầu của player

    temp = []   # return list

    # trạng thái bài toán = ( (vị trí agent / player), (vị trí các box) )
    start_state = (beginPlayer, beginBox)   # trạng thái khởi đầu

    # hàng đợi chứa các node sắp đuọc bung ra
    # mỗi node chứa các trạng thái theo thứ tự tạo thành một lời giải / solution
    # mỗi node là một list chứa các list con, mỗi list con chứa 1 trạng thái
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox))

    exploredSet = set() # set chứa trạng thái đã duyệt qua

    # list chứa list các action của các solution
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1]))

    while len(frontier.Heap) > 0:
        # lấy node có cost thấp nhất frontier
        node = frontier.pop()

        # lấy list action ứng với node vừa lấy
        node_action = actions.pop()

        # tt đang duyệt là end state
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        # end_if

        ### CONTINUE YOUR CODE FROM HERE

        # tt đang duyệt chưa đc duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            # duyệt qua từng legal action của tt hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # tt mới sau khi thực hiện action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                
                # tt mới có thể không dẫn tới solution
                if isFailed(newPosBox):
                    continue    # bỏ qua
                # end_if

                new_node = node + [(newPosPlayer, newPosBox)]   # thêm tt mới vào node
                new_action = node_action + [action[-1]] # thêm action mới thực hiện vào list action
                
                # tính sum = cost + heuristic
                sum_cost_heuristic = cost(new_action[1:]) + heuristic(newPosPlayer, newPosBox)

                # thêm node vào bên phải frontier
                frontier.push(new_node, sum_cost_heuristic)

                # thêm vào list list actions
                actions.push(new_action, sum_cost_heuristic)
            # end_for
        # end_if
    # enf_while
    
    # end =  time.time()

    return temp, len(exploredSet)+1
# end_def

def heuristic2(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between each box and the nearest goal"""
    # input:
        # posPlayer: vị trí player
        # posBox: tuple vị trí các box
    # output:
        # distance: overall distance between each box and the nearest goal
                    # tổng các khoảng cách giữa từng box và goal gần nhất
    
    distance = 0

    # This set stores the boxes that are already on their goals
    completes = set(posGoals) & set(posBox)

    # This list holds the positions of remaining "else boxes" (elements in posBox that are not in completes)
    sortposBox = list(set(posBox).difference(completes))

    # This list holds the goal positions for the remaining "else boxes" (elements in posGoals that are not in completes)
    sortposGoals = list(set(posGoals).difference(completes))

    # The code iterates through sortposBox (same length as sortposGoals)
    for i in range(len(sortposBox)):
        dis = -1
        for j in range(len(sortposGoals)):
            if dis == -1:
                # tính khoảng cách giữa box và goal theo 2 trục x, y
                dis = (abs(sortposBox[i][0] - sortposGoals[j][0])) + (abs(sortposBox[i][1] - sortposGoals[j][1]))
            else:
                dis = min(dis, (abs(sortposBox[i][0] - sortposGoals[j][0])) + (abs(sortposBox[i][1] - sortposGoals[j][1])) )
            # end_if
        # end_for
            
        distance += dis
    # end_for
    
    return distance
# end_def

def aStarSearch2(gameState):
    """Implement aStarSearch approach using heuristic2"""
    # Input: gameState: trạng thái khởi đầu của màn chơi
    # Output: lời giải của màn chơi

    # start =  time.time()
    beginBox = PosOfBoxes(gameState)    # vị trí khởi đầu của box
    beginPlayer = PosOfPlayer(gameState)    # vị trí khởi đầu của player

    temp = []   # return list

    # trạng thái bài toán = ( (vị trí agent / player), (vị trí các box) )
    start_state = (beginPlayer, beginBox)   # trạng thái khởi đầu

    # hàng đợi chứa các node sắp đuọc bung ra
    # mỗi node chứa các trạng thái theo thứ tự tạo thành một lời giải / solution
    # mỗi node là một list chứa các list con, mỗi list con chứa 1 trạng thái
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic2(beginPlayer, beginBox))

    exploredSet = set() # set chứa trạng thái đã duyệt qua

    # list chứa list các action của các solution
    actions = PriorityQueue()
    actions.push([0], heuristic2(beginPlayer, start_state[1]))

    while len(frontier.Heap) > 0:
        # lấy node có cost thấp nhất frontier
        node = frontier.pop()

        # lấy list action ứng với node vừa lấy
        node_action = actions.pop()

        # tt đang duyệt là end state
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        # end_if

        ### CONTINUE YOUR CODE FROM HERE

        # tt đang duyệt chưa đc duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])

            # duyệt qua từng legal action của tt hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):
                # tt mới sau khi thực hiện action
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                
                # tt mới có thể không dẫn tới solution
                if isFailed(newPosBox):
                    continue    # bỏ qua
                # end_if

                new_node = node + [(newPosPlayer, newPosBox)]   # thêm tt mới vào node
                new_action = node_action + [action[-1]] # thêm action mới thực hiện vào list action
                
                # tính sum = cost + heuristic2
                sum_cost_heuristic = cost(new_action[1:]) + heuristic2(newPosPlayer, newPosBox)

                # thêm node vào bên phải frontier
                frontier.push(new_node, sum_cost_heuristic)

                # thêm vào list list actions
                actions.push(new_action, sum_cost_heuristic)
            # end_for
        # end_if
    # enf_while
    
    # end =  time.time()

    return temp, len(exploredSet)+1
# end_def

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result, num_nodes = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result, num_nodes = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result, num_nodes = uniformCostSearch(gameState)
    elif method == 'astar': 
        result, num_nodes = aStarSearch(gameState)
    elif method == 'astar2': 
        result, num_nodes = aStarSearch2(gameState)       
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()

    # print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print('Runtime of %s (s): %.2f' %(method, time_end-time_start))
    print(result)

    # print(len(result))
    print(num_nodes, "nodes")
    # print(cost(result))
    
    return result
