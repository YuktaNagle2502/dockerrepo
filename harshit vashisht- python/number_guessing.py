winning_number = 7
guessed_number = int(input("Guess a number between 1 and 100: "))
if (guessed_number == winning_number):
    print("You win!")
else:  # nested if else statement
    if (guessed_number < winning_number):
        print("Too low!")
    else:
        print("Too high!")