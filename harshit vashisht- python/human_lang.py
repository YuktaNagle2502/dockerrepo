number = input("Enter numbers: ")
i = 0
sum = 0
while (i < len(number)):
    sum += int(number[i])
    i += 1
print("The sum of the digits is: ", sum)