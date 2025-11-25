num = input("Enter numbers: ")
i = 0
sum = 0
for i in range(len(num)):
    sum += int(num[i])
    i += 1
print(f"The sum of numbers is: {sum}")