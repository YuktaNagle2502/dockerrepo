user_name, single_char = input("Enter your name: ").split(",")
print(f"length of user name is: {len(user_name)}")
user_name= user_name.strip().lower()  # remove starting and ending spaces and convert to lower case
single_char= single_char.strip().lower()
print(f"count of single char: {user_name.count(single_char)}")