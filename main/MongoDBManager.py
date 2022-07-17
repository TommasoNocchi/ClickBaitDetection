import datetime
from pymongo import MongoClient
"""
First Example - Connection
# pprint library is used to make the output look more pretty
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://localhost:27017")
db=client.DietManager
# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)
"""

class MongoDBManager:
    def __init__(self):
        self.client = MongoClient(port=27017)
        self.db = self.client.ClickBaitDetector

    def insertUser(self, username, password):
        user = {
            '_id' : username,
            'password' : password
        }
        result = self.db.users.insert_one(user)

    def deleteUser(self, username):
        user= {
            '_id' : username
        }
        result = self.db.users.delete_one(user)

    def retriveUser(self, username):
        user = self.db.users.find_one({'_id': username})
        return user

    def insertNewPrediction(self, username, title, prediction):
        self.db.users.update_one(
            {'_id': username},
            {"$addToSet":
                 {'personalTitle':
                      {
                        "title": title,
                        "prediction": prediction,
                        "timestamp" : datetime.datetime.now()
                      }
                 }
            }
        )
    def retriveTitlesList(self, username):
        user = self.retriveUser(username)
        if(user.get("personalTitle") != None):
            return user.get("personalTitle")
        return None


"""
from random import randint
#Step 1: Connect to MongoDB - Note: Change connection string as needed
client = MongoClient(port=27017)
db=client.business
#Step 2: Create sample data
names = ['Kitchen','Animal','State', 'Tastey', 'Big','City','Fish', 'Pizza','Goat', 'Salty','Sandwich','Lazy', 'Fun']
company_type = ['LLC','Inc','Company','Corporation']
company_cuisine = ['Pizza', 'Bar Food', 'Fast Food', 'Italian', 'Mexican', 'American', 'Sushi Bar', 'Vegetarian']
for x in range(1, 501):
    business = {
        'name' : names[randint(0, (len(names)-1))] + ' ' + names[randint(0, (len(names)-1))]  + ' ' + company_type[randint(0, (len(company_type)-1))],
        'rating' : randint(1, 5),
        'cuisine' : company_cuisine[randint(0, (len(company_cuisine)-1))]
    }
    #Step 3: Insert business object directly into MongoDB via insert_one
    result=db.reviews.insert_one(business)
    #Step 4: Print to the console the ObjectID of the new document
    print('Created {0} of 500 as {1}'.format(x,result.inserted_id))
#Step 5: Tell us that you are done
print('finished creating 500 business reviews')



        self.db.users.update(
            {'_id': username},
            {"$push":
                {"personalTitle":
                    {
                        "title": title,
                        "prediction": prediction
                    }
                }
            }
        )
"""