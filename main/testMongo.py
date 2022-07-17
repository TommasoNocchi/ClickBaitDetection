from MongoDBManager import  *
from CLI import *

cli = CLI()
mongoDBManager = MongoDBManager()
user = {}
user = mongoDBManager.retriveUser("simone")
list = user.get("personalTitle")
print("User:",user)
print("list: ", list)
for title in list:
    print("*")
    print(title.get("title"))
    print(title.get("prediction"))
    print("*")
"""
Test 1
start = ""

mongoDBManager.deleteUser("user0")
mongoDBManager.insertUser("user0", "user0")

while(start != "end"):
    start = cli.startWelcomeMenu()

    #Log-In
    if(start == "S"):
        logIn = []
        #user = {}
        logIn = cli.log()
        user = mongoDBManager.retriveUser(logIn[0])
        print(user)
        if(user == None):
            print("OK null")
        #password = user.get("password")
        if(password == logIn[1]):
            logged = True
            print(logged)

    #else if(start = "R"):
    
"""