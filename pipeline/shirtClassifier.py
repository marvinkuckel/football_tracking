class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!

    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a current frame list
        # The task of the shirt classifier module is to identify the two teams based on their shirt color and to assign each player to one of the two teams

        # Note: You can access data["image"] and data["tracks"] to receive the current image as well as the current track list
        # You must return a dictionary with the given fields:
        #       "teamAColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team A
        #       "teamBColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team B
        #       "teamClasses"       A list with an integer class for each track according to the following mapping:
        #           0: Team not decided or not a player (e.g. ball, goal keeper, referee)
        #           1: Player belongs to team A
        #           2: Player belongs to team B
        
        #Accessing the image and tracks
        image = data["image"]
        tracks = data["tracks"]
        track_classes = data["trackClasses"]      # trackClasses player = 2 
        
        #only pic out the players from the detected tracks -> trackClasses
        
        #get the color for every player -> maybe in a list with the color of every player
        
        #kluster the colors (klassifier) - i guess i will use KMeans with 2 clusters - in the end i have the 2 different teamcolors
        
        #every player is put into a team (using the klassifiers result)
        
        #returning something like this:
        #return { "teamAColor": tupleA,
        #         "teamBColor": tupleB,
        #         "teamClasses": teamClasses }
        
        
        