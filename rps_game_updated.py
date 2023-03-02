"""
Python implementation of Rock-Paper-Scissors!

Rules:
Each player chooses a move (simultaneously) from the choices: rock, paper or scissors. 
If both players choose the same move, the round ends in a tie. 
Otherwise:
    Rock beats Scissors
    Scissors beats Paper
    Paper beats Rock.

To modify the GUI, please refer to:
Tkinter Documentation: https://docs.python.org/3/library/tk.html
"""

## Import required libraries
import random
import tkinter as tk
from tkinter import *
import numpy as np
from bayes_net_counting import *
import sys
# from bayes_net_counting import bayes_strategy
## Uncomment the following line once you have implemented naive_bayes_net as well as the naive_bayes_strategy function
# from naive_bayes_net import naive_bayes_strategy

bys_model = BYSModel()

def save_data(hm, cm):
    '''
    This function collects data 
    '''
    data.append([hm,cm])

def update_scores(winner):
    """
    returns updated total scores
    """
    global total_computer_score
    global total_human_score
    if (winner == 'human'):
        total_human_score += 1
        total_computer_score += 0
    if (winner == 'computer'):
        total_human_score += 0
        total_computer_score += 1
    if (winner == 'tie'):
        total_human_score += 0
        total_computer_score += 0
        
    return total_human_score, total_computer_score

def select_winner(computer_move, human_move):
    """
    return: winner of the round
    """
    if computer_move == human_move:
        return 'tie'
    if computer_move == 'rock':
        if human_move == 'paper':
            return 'human'
        return 'computer'
    elif computer_move == 'paper':
        if human_move == 'rock':
            return 'computer'
        return 'human'
    elif computer_move == 'scissors':
        if human_move == 'paper':
            return 'computer'
        return 'human'

def get_computer_move():
    """
    Using randint()m which is an inbuilt function of the random module in Python3, generate a number between (1,3)
    where, 1 - Rock, 2 - Paper, 3 - Scissors
    returns string representing what ai move (rock | paper | scissors)
    """
    move = random.randint(1, 3)
    if move == 1:
        return 'rock'
    elif move == 2:
        return 'paper'
    else:
        return 'scissors'

def get_ai_move(winner, human_move, variable):
    print("!!!")
    '''
    To Do: Implement the win-stay, lose-shift or the win-shift, lose-shift strategy
    '''
    if variable.get() == 'win_stay_lose_shift':
        '''
        Win Stay Lose Shift Strategy: If you win, play the move you just played, else play the move that will beat your opponent's previous move
        '''
        if winner == 'human':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'computer':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'tie':
            return get_computer_move()

    elif variable.get() == 'win_shift_lose_shift':
        '''
        Win Shift Lose Shift Strategy: If you win, play the move your opponent played during the previous move, else play the move that will beat your opponent's previous move
        '''
        if winner == 'human':
            if human_move == 'rock':
                return  'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'computer':
            if human_move == 'rock':
                return 'rock'
            elif human_move == 'paper':
                return 'paper'
            elif human_move == 'scissors':
                return 'scissors'

        elif winner == 'tie':
            return get_computer_move()
    elif variable.get() == 'bayes_net with win shift':
        if winner == 'human':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'computer':
            if human_move == 'rock':
                return 'rock'
            elif human_move == 'paper':
                return 'paper'
            elif human_move == 'scissors':
                return 'scissors'

        elif winner == 'tie':
            return get_computer_move()

    elif variable.get() == 'bayes_net with win stay':
        if winner == 'human':
            if human_move == 'rock':
                next_move = 'paper'
            elif human_move == 'paper':
                next_move = 'scissors'
            elif human_move == 'scissors':
                next_move = 'rock'
            return next_move
        elif winner == 'computer':
            if human_move == 'rock':
                next_move = 'paper'
            elif human_move == 'paper':
                next_move = 'scissors'
            elif human_move == 'scissors':
                next_move = 'rock'
            return next_move
        elif winner == 'tie':
            return get_computer_move()
    elif variable.get() == 'naive_bayes_net with win shift':
        if winner == 'human':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'
        elif winner == 'computer':
            if human_move == 'rock':
                return 'rock'
            elif human_move == 'paper':
                return 'paper'
            elif human_move == 'scissors':
                return 'scissors'

        elif winner == 'tie':
            return get_computer_move()
    elif variable.get() == 'naive_bayes_net with win stay':
        if winner == 'human':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'computer':
            if human_move == 'rock':
                return 'paper'
            elif human_move == 'paper':
                return 'scissors'
            elif human_move == 'scissors':
                return 'rock'

        elif winner == 'tie':
            return get_computer_move()



    

def get_bayes_net_human_move(human_move, computer_move, variable):
    '''
    To Do: Implement Bayesian Network that takes as input the previous round's moves and predicts the next move you should play
    The choice of Bayes network to use (V-DAG (Prediction|Human Move and Computer Move) or Naive Bayes (Inverted V-DAG) (Human Move|Prediction)x(Computer Move|Prediction))
    '''
    if variable.get() == 'bayes_net with win stay' or variable.get() == 'bayes_net with win shift':
        '''
        To Do: refer to assignment pdf for instructions. Please fill out the rest of the logic for this case and return 
        a move to play against according to the instructions.
        '''
        pred_move = bayes_function_vdag(human_move, computer_move)
        # if pred_move == 'rock':
        #     return get_computer_move()
        # elif pred_move == 'paper':
        #     return 'scissors'
        # else:
        #     return 'rock'
        return pred_move
    elif variable.get() == 'naive_bayes_net with win stay' or variable.get() == 'naive_bayes_net with win shift':
        '''
        To Do: refer to assignemnt pdf for instructions
        '''
        predicted_Y = bys_model.bayes_function_ivdag(human_move, computer_move, update=True)
        print("???")
        if predicted_Y == 'rock':
            return 'paper'
        elif predicted_Y == 'paper':
            return 'scissors'
        else:
            return 'rock'
        # return pred_move
def get_real_time_bayes_net_human_move():
    '''
    [BONUS]
    To Do: Build on your get_bayes_net_human_move() to keep updating your network based on data you collect as you are playing the game
    The choice of Bayes network to use (V-DAG (Prediction|Human Move and Computer Move) or Naive Bayes (Inverted V-DAG) (Human Move|Prediction)x(Computer Move|Prediction))
    '''

def get_human_move(human_move, tt, variable):
    """
    returns a valid move from the human (rock, paper, or scissors) and updates the scores and returns a winner
    """

    global continue_playing_button
    global last_round_human_move
    global last_round_winner
    global last_round_computer_move

    # Get computer move
    if variable.get() == 'random':
        computer_move = get_computer_move()
    elif variable.get() == 'win_stay_lose_shift':
        if count == 0: ## Round 1, choose random move for AI
            computer_move = get_computer_move()
        else:
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)
    elif variable.get() == 'win_shift_lose_shift':
        if count == 0: ## Round 1, choose random move for AI
            computer_move = get_computer_move()
        else:
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)
    elif variable.get() == 'bayes_net with win shift':
        if count == 0: ## Round 1, choose random move for Bayes Net
            human_move = get_computer_move()
            computer_move = get_computer_move()
        else:
            human_move = get_bayes_net_human_move(last_round_human_move, last_round_computer_move, variable)
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)
    elif variable.get() == 'bayes_net with win stay':
        if count == 0:  ## Round 1, choose random move for Bayes Net
            human_move = get_computer_move()
            computer_move = get_computer_move()
        else:
            human_move = get_bayes_net_human_move(last_round_human_move, last_round_computer_move, variable)
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)
        ## Change the following line to implement either of the AI strategy moves, you need to get this option as an input from the user or manually assign a strategy according to a combination. 
        # If you choose to manually enter a strategy to use, change get_ai_move() accordingly
    elif variable.get() == 'naive_bayes_net with win shift':
        if count == 0:
            human_move = get_computer_move()
            computer_move = get_computer_move()
        else:
            human_move = get_bayes_net_human_move(last_round_human_move, last_round_computer_move, variable)
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)
    elif variable.get() == 'naive_bayes_net with win stay':
        if count == 0:
            human_move = get_computer_move()
            computer_move = get_computer_move()
        else:
            human_move = get_bayes_net_human_move(last_round_human_move, last_round_computer_move, variable)
            computer_move = get_ai_move(last_round_winner, last_round_human_move, variable)

    # Select the winner
    winner = select_winner(computer_move, human_move)

    last_round_human_move = human_move
    last_round_computer_move = computer_move
    last_round_winner = winner

    # Update the scores
    update_scores(winner)

    save_data(human_move, computer_move)

    # Print round summary
    HM_label=Label(Window, foreground='black',background='white', text='Human move was {}'.format(human_move))
    HM_label.place(x = 240,y = 300) 

    CM_label=Label(Window, foreground='black',background='white', text='Computer move was {}'.format(computer_move))
    CM_label.place(x = 240,y = 340) 

    W_label=Label(Window, foreground='black',background='white', text='Winner is {}'.format(winner))
    W_label.place(x = 240,y = 380) 

    CSH_label=Label(Window, foreground='black',background='white', text='Current score for human: {}'.format(total_human_score))
    CSH_label.place(x = 240,y = 420) 

    CSC_label=Label(Window, foreground='black',background='white', text='Current score for computer: {}'.format(total_computer_score))
    CSC_label.place(x = 240,y = 460) 

    continue_playing_button=Button(Window, foreground='black',background='white',text='Continue Playing',command=lambda xx= tt: reset(xx, variable))
    continue_playing_button.place(x = 100, y = 500)

    labels.extend([HM_label,CM_label,W_label,CSH_label,CSC_label,continue_playing_button])
    return human_move, winner

def reset(tt, variables):
    '''
    This function is used to reset a round after it has been played as well as to terminate the game and display the game summary 
    once the input number of rounds have been played
    '''
    for label in labels:
        label.destroy()
    global count
    count += 1
    display_module(tt, variables)

    if count == int(tt.get()):
        for label in labels:
            label.destroy()

        for label in labels2:
            label.destroy()

        GS_label=Label(Window, foreground='black',background='white', text='Game Summary')
        GS_label.place(x = 240,y = 280) 

        # Print round summary
        TSH_label=Label(Window, foreground='black',background='white', text='Total score for Human: {}'.format(total_human_score))
        TSH_label.place(x = 240,y = 320) 

        TSC_label=Label(Window, foreground='black',background='white', text='Total score for Computer: {}'.format(total_computer_score))
        TSC_label.place(x = 240,y = 360) 

        labels2.extend([GS_label,TSH_label,TSC_label])


        if total_computer_score > total_human_score:
            computer_label=Label(Window, foreground='black',background='white', text='Computer Wins!')
            computer_label.place(x = 240,y = 400) 
            labels2.append(computer_label)

        elif total_computer_score < total_human_score:
            human_label=Label(Window, foreground='black',background='white', text='Human Wins!')
            human_label.place(x = 240,y = 400) 
            labels2.append(human_label)

        else:
            tie_label=Label(Window, foreground='black',background='white', text='Series ended in a tie')
            tie_label.place(x = 240,y = 400) 
            labels2.append(tie_label)

        ## Clicking "Reset Game" terminates the game and dumps all data collected in this game
        reset_button=Button(Window, foreground='black',background='white',text='Reset Game',command= lambda xx= "reset": reset_game(xx))
        reset_button.pack()
        reset_button.place(x = 240, y = 500)


def display_module(tt,variable):
    '''
    This function to asks you to select a move

    '''
    round_label=Label(Window, foreground='black',background='white', text='Round {}'.format(count+1))
    round_label.place(x = 40, y = 280)

    select_label=Label(Window, foreground='black',background='white', text='Enter your move (rock|paper|scissors):')
    select_label.place(x = 40, y = 240)

    if variable.get() == 'bayes_net with win stay':
        bayes_button=Button(Window, foreground='black',background='white',text='Get Bayes Move',command=lambda t = None: get_human_move(t,tt,variable))
        bayes_button.place(x = 330, y = 240)
        labels2.extend([round_label,select_label,bayes_button])
    elif variable.get() == 'bayes_net with win shift':
        bayes_button = Button(Window, foreground='black', background='white', text='Get Bayes Move',command=lambda t = None: get_human_move(t, tt, variable))
        bayes_button.place(x=330, y=240)
        labels2.extend([round_label, select_label, bayes_button])
    elif variable.get() == 'naive_bayes_net with win stay':
        bayes_button=Button(Window, foreground='black',background='white',text='Get Naive Bayes Move',command=lambda t = None: get_human_move(t,tt,variable))
        bayes_button.place(x = 330, y = 240)
        labels2.extend([round_label,select_label,bayes_button])
    elif variable.get() == 'naive_bayes_net with win shift':
        bayes_button=Button(Window, foreground='black',background='white',text='Get Naive Bayes Move',command=lambda t = None: get_human_move(t,tt,variable))
        bayes_button.place(x = 330, y = 240)
        labels2.extend([round_label,select_label,bayes_button])

    else:
        rock_button=Button(Window, foreground='black',background='white',text='Rock',command=lambda t= "rock": get_human_move(t,tt,variable))
        rock_button.place(x = 330, y = 240)

        paper_button=Button(Window, foreground='black',background='white',text='Paper',command=lambda t= "paper": get_human_move(t,tt,variable))
        paper_button.place(x = 430, y = 240)

        scissors_button=Button(Window, foreground='black',background='white',text='Scissors',command= lambda t= "scissors": get_human_move(t,tt,variable))
        scissors_button.place(x = 530, y = 240)

        labels2.extend([round_label, select_label, paper_button, scissors_button, rock_button])


def reset_game(xx):
    '''
    This function is used to reset a game after all rounds have been played as well as saves previously collected moves
    Note that clicking on reset dumps all previously collected data onto a numpy file and resets the game
    You might want to modify this function such that you keep collecting data and the numpy dumping happens only when you click "Exit Program"
    '''
    if xx == "reset":
        for label in labels2:
            label.destroy()
    np.array(data).dump(open('data.npy', 'wb'))
    welcome()


def welcome():
    '''
    This welcome function asks you to enter the number of rounds you would like to play and enables you to start playing the game
    '''
    games_label=Label(Window, foreground='black',background='white', text='Enter the names of rounds you want to play:') 
    games_label.place(x = 40,y = 100)

    user_entry = Entry(Window, width = 5)
    user_entry.pack()
    user_entry.place(x = 330, y = 97) 

    global labels
    labels = []

    strats_label=Label(Window, foreground='black',background='white', text='Choose a strategy you wish to play against:') 
    strats_label.place(x = 40,y = 60)

    ## Select Strategy
    OPTIONS = [
                "win_shift_lose_shift",
                "win_stay_lose_shift",
                "bayes_net with win stay",
                "bayes_net with win shift",
                "naive_bayes_net with win stay",
                "naive_bayes_net with win shift",
                "random"
                ]
    variable = StringVar(Window)
    variable.set(OPTIONS[6]) # default value random
    strats = OptionMenu(Window, variable, *OPTIONS)
    strats.pack()
    strats.place(x = 310,y = 60) 

    labels.extend([strats, strats_label])

    start_game_button=Button(Window,text='Start Playing!',command= lambda t= user_entry: playgame(t, variable))
    start_game_button.pack()
    start_game_button.place(x = 330, y = 130)


def playgame(t, variable):
    '''
    This function controls the round logic, based on how many rounds you would like to play
    '''
    nums_label=Label(Window, foreground='black',background='white', text='You will play {} games against {} strategy'.format(t.get(), variable.get()))
    nums_label.place(x = 40,y = 180) 
    end_label=Label(Window, foreground='black',background='white', text='----------------------------------------------')
    end_label.place(x = 40,y = 200) 

    global total_computer_score
    global total_human_score
    global labels2
    global data
    global count
    global last_round_winner
    global last_round_human_move
    global last_round_computer_move
    count = 0
    data = []
    total_human_score = 0
    total_computer_score = 0
    last_round_winner = 0
    last_round_human_move = 0
    last_round_computer_move = 0
    labels2 = []
    labels2.extend([nums_label, end_label])

    # Call display function to select a move
    display_module(t, variable)


if __name__ == '__main__':
    ## Initialize a Tkinter GUI window
    Window = Tk()
    ## Set GUI window dimensions
    Window.geometry("650x550")
    ## Set GUI background color
    Window.configure(background='white')
    ## Sets a title for the GUI
    Window.title("Rock - Paper - Scissors")
    ## Welcome statement
    welcome_label=Label(Window, foreground='black',background='white', text='Welcome to Rock, Paper, Scissors! --Presented to you by 24787 TA/CA')
    welcome_label.place(x = 40,y = 60) 
    welcome_label.pack()
    ## Call the welcome function
    welcome()
    ## Clicking "Exit Program" terminates the game and exits the program
    exitButton=Button(Window,text='Exit program',command=Window.destroy).place(x = 352, y = 500)
    Window.mainloop()
    
