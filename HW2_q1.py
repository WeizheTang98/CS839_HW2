import openai

# Your API Key (replace with your actual key)
openai.api_key = 'KeyAPI'


# Helper to call the GPT-4 API using the new syntax
def call_gpt(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant helping solve a puzzle."},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content']


# Thinker GPT-4: Generate all possible actions using the user's original question
def thinker_generate_moves(state):
    # Use the original question for generating actions
    prompt = f"""
    If I have four policemen and four thieves, now I need to pass the river. The thieves will flee if there is no policeman on the same side as the thieves. 
    The boat can take at most two people. In the current state:
    Left Bank: {state['left']['P']} Policemen, {state['left']['T']} Thieves
    Right Bank: {state['right']['P']} Policemen, {state['right']['T']} Thieves
    The boat is on the {'left' if state['boat'] == 'left' else 'right'} bank.

    What are the possible actions you think reasonably can take to successfully and safely move thieves/policeman across the 
    river from left to right? Please note that boat is on the {'left' if state['boat'] == 'left' else 'right'} bank now
    bank. Assume one police can control all the thieves (please Give me a list of tuples, the first entry in each tuple is the number of policeman to be moved 
    and second entry of the tuple is the number of thieves to be moved in python format I.E. [(entry1,entry2),(entry1,entry2) ...],) 
    and no need for other useless word and keep the list format strict like this without extra spaces/newline: [(entry1,entry2),(entry1,entry2) ...])
    """
    actions = call_gpt(prompt)
    actions = preprocess(actions)

    print(actions)

    return actions


# Critic GPT-4: Score the new states using the user's original question
def critic_score_state(new_state):
    # Use the original question for evaluating the state
    prompt = f"""
    Given the current state:
    Left Bank: {new_state['left']['P']} Policemen, {new_state['left']['T']} Thieves
    Right Bank: {new_state['right']['P']} Policemen, {new_state['right']['T']} Thieves
    The boat is on the {'left' if new_state['boat'] == 'left' else 'right'} bank.

    If the thieves will flee if there is no policeman present and we want to move all the police 
    and thieves from left to right, is this state safe and good for transporting? 
    Please score this state from 0 to 1 based on safety and goodness for transporting, where 1 is completely safe and 0 is 
    completely unsafe Just a number is fine don't use any word.
    """
    score = call_gpt(prompt)
    score = float(score)
    return score


# Function to apply a move and return the new state
def apply_move(state, move):
    """
    Apply the given move to the current state.

    Parameters:
    - state: A dictionary representing the current state of the game.
             - 'left': dict with counts of 'P' (policemen) and 'T' (thieves) on the left bank.
             - 'right': dict with counts of 'P' (policemen) and 'T' (thieves) on the right bank.
             - 'boat': str, either 'left' or 'right', indicating where the boat currently is.
    - move: A tuple (policemen, thieves) indicating how many policemen and thieves to move.

    Returns:
    - new_state: The updated state after the move.
    """

    # Extract the number of policemen and thieves to move
    policemen_to_move, thieves_to_move = move

    # Deep copy the current state to avoid mutating the original state
    new_state = {
        'left': state['left'].copy(),
        'right': state['right'].copy(),
        'boat': state['boat']
    }

    # Move from left to right
    if state['boat'] == 'left':
        # Check if move is valid (enough people on the left bank)
        if new_state['left']['P'] >= policemen_to_move and new_state['left']['T'] >= thieves_to_move:
            # Update the left bank
            new_state['left']['P'] -= policemen_to_move
            new_state['left']['T'] -= thieves_to_move
            # Update the right bank
            new_state['right']['P'] += policemen_to_move
            new_state['right']['T'] += thieves_to_move
            # Move the boat to the right bank
            new_state['boat'] = 'right'
        else:
            print("Invalid move: Not enough people on the left bank.")
            return None
            # raise ValueError("Invalid move: Not enough people on the left bank.")

    # Move from right to left
    elif state['boat'] == 'right':
        # Check if move is valid (enough people on the right bank)
        if new_state['right']['P'] >= policemen_to_move and new_state['right']['T'] >= thieves_to_move:
            # Update the right bank
            new_state['right']['P'] -= policemen_to_move
            new_state['right']['T'] -= thieves_to_move
            # Update the left bank
            new_state['left']['P'] += policemen_to_move
            new_state['left']['T'] += thieves_to_move
            # Move the boat to the left bank
            new_state['boat'] = 'left'
        else:
            print("Invalid move: Not enough people on the left bank.")
            return None
            # raise ValueError("Invalid move: Not enough people on the right bank.")

    return new_state


# Recursively explore the tree
# def tree_of_thought(state, path=[]):
#     # Base case: If everyone is on the right bank, solution found
#     if state['right'] == {'P': 4, 'T': 4}:
#         return path
#
#     # Thinker GPT-4 generates possible moves using the original question
#     possible_moves = thinker_generate_moves(state)
#
#     # Evaluate each move with Critic GPT-4 using the original question
#     best_move = None
#     best_score = 0
#     for move in possible_moves:
#         # Apply the move to get a new state
#         new_state = apply_move(state, move)
#
#         if new_state is None:
#             continue
#
#         # Critic GPT-4 scores the new state using the original question
#         score = critic_score_state(new_state)
#         score = float(score)  # Convert to float
#
#         if score > best_score:
#             best_score = score
#             best_move = move
#
#     # If no valid move is found, return None (no solution at this branch)
#     if best_move is None:
#         return None
#
#     # Recursively apply the best move
#     new_path = path + [best_move]
#     new_state = apply_move(state, best_move)
#     return tree_of_thought(new_state, new_path)
# Recursively explore the tree by exploring all possible branches (backtracking enabled)
# Recursively explore the tree until one valid solution is found
def tree_of_thought(state, path=[], threshold=0.2):
    """
    Recursive Tree of Thought implementation that explores the tree until one valid path is found.

    Parameters:
    - state: The current state of the puzzle.
    - path: The sequence of moves taken to reach the current state.
    - threshold: The safety score threshold, default is 0.9.

    Returns:
    - A valid solution (path) if found, otherwise None.
    """
    # Base case: If everyone is on the right bank, return the valid path
    if state['right'] == {'P': 4, 'T': 4}:
        return path

    # Thinker GPT-4 generates possible moves using the original question
    possible_moves = thinker_generate_moves(state)

    for move in possible_moves:
        # Apply the move to get a new state
        new_state = apply_move(state, move)

        # If the move is invalid (new_state is None), skip to the next move
        if new_state is None:
            continue  # Skip this invalid move

        # Critic GPT-4 scores the new state (float between 0 and 1)
        score = critic_score_state(new_state)

        # If the score is greater than or equal to the threshold, continue exploring this branch
        if score >= threshold:
            result = tree_of_thought(new_state, path + [move], threshold)
            if result is not None:
                return result  # Return the first valid solution found



    return None  # Return None if no valid solution is found at this branch



def preprocess(txt):
    # print(txt)
    idx_B = txt.index("[(")
    idx_E = txt.index(")]")

    info = txt[idx_B:idx_E+1]
    info = info.split(")")
    info = info[0:-1]
    ret = []
    for elem in info:
        e = list(elem)
        P = 0
        T = 0
        Find_p = False
        for s in e:
            try:
                s = int(s)
                if Find_p is False:
                    P = s
                    Find_p = True
                else:
                    T = s
            except:
                continue
        ret.append((P,T))
    return ret


# Initial state: 4 policemen and 4 thieves on the left bank, empty right bank
initial_state = {
    'left': {'P': 4, 'T': 4},
    'right': {'P': 0, 'T': 0},
    'boat': 'left'  # Boat starts on the left side
}

# print(critic_score_state(initial_state))


# Start the Tree of Thought exploration
solution = tree_of_thought(initial_state)

# Print the solution if found
if solution:
    for step in solution:
        print(step)
else:
    print("No solution found")
