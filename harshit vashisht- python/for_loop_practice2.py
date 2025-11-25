user_name = input("Enter your name: ")
temp = ""
for i in range(0, len(user_name)):
    if user_name[i] not in temp:
        print(f"{user_name[i]}: {user_name.count(user_name[i])}")
        temp += user_name[i]

     
