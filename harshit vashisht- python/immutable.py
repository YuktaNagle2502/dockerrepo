# strings are immutable
string = "hello"
# string[0] = "H"  # TypeError: 'str' object does not support item assignment
new_string = string.replace("h", "H") 
print(new_string)  # Output: hello
