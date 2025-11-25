name,age = input("Enter your name and age: ").split(",")
name = name.lower()
age = int(age)
if name[0]=='a' and age >=10:
    print("you can watch coco movie")
else:
    print("you cannot watch coco movie")


