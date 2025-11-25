# replace()
string = "she is beautiful and she is a good dancer"
print(string.replace("is","was", 2)) 

# find()
print(string.find("is")) # find first occurrence of "is"
print(string.find("is", 5)) # find second occurrence of "is" 

is_pos1 = string.find("is")
is_pos2 = string.find("is", is_pos1 + 1) 
print(is_pos2) # find second occurrence of "is"