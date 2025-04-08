from sklearn.cluster import KMeans
import numpy as np
import cv2

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
        players = [i for i, cls in enumerate(track_classes) if cls == 2]
        
        #in case only 1 team is detected - so there is no error
        if len(players) < 2:
            return {
                "teamAColor": (0, 0, 255),
                "teamBColor": (255, 0, 0),
                "teamClasses": [0] * len(track_classes)
            }
        
        #get the color for every traked player in the actual frame
        colors = []
        for player in players:
            #get the coordinates of the players shirts 
            #x and y is the center of the Box
            #w and h is the width and height of the box
            x, y, w, h = tracks[player]  #find the box for every player in players
            x1, y1 = int(x - w / 2), int(y - h / 2)  #the corner left top of the box
            x2, y2 = int(x + w / 2), int(y + h / 2)  #the corner right bottom of the box
            shirt_region = image[y1:y2, x1:x2]     #here i need cv2 
            #skipping/continueing when box is not in the picture
            if shirt_region.size == 0:
                continue
            #calculate the average color of the shirt region
            avg_color = shirt_region.mean(axis=(0, 1))  #BGR  -> output like this: [120, 50, 200]
            colors.append(avg_color)   #put the average color of the shirt into the list
            #colors should be a list of tuples with the average color of every player. 
            #for example there will be many blue and many red colors.
            #so i can use KMeans to cluster the colors into 2 clusters (teamA and teamB)
        
        #in case only 1 team/color is detected - so there is no error
        if len(colors) < 2:
            return {
                "teamAColor": (0, 0, 255),
                "teamBColor": (255, 0, 0),
                "teamClasses": [0] * len(track_classes)
            }

        
        #kluster the colors (klassifier) - i guess i will use KMeans with 2 clusters - in the end i have the 2 different teamcolors
        kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
        teamAColor, teamBColor = kmeans.cluster_centers_
        
        #every player is put into a team (using the klassifiers result)
        
        #returning something like this:
        #return { "teamAColor": tupleA,
        #         "teamBColor": tupleB,
        #         "teamClasses": teamClasses }
        
        
        