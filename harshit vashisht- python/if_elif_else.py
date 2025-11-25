age = int(input("Enter your age: "))
if age == 0 or age < 0:
    print("you cannot watch movie")
elif 0<age<3:
    print("The ticket price is free")
elif 3<=age<10:
    print("The ticket price is 150")
elif 10<=age<60:
    print("The ticket price is 250")
else:
    print("The ticket price is 200")