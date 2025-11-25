name = "     yukta       "
name1 = "     yuk   ta       "
dots  = "............."
print(name + dots)
print(name.strip() + dots) # remove the starting and ending spaces from the string
print(name.lstrip() + dots) # remove the starting spaces from the string
print(name.rstrip() + dots) # remove the starting spaces from the string
print(name1.replace(" ", "") + dots) # remove all the spaces from the string
print(name1.strip() +  dots) # strip method will not remove the spaces between the words
