from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cluster_model(self, image, n_clusters):
        # Reshape image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, 
                        init="k-means++", # faster runs
                        n_init=1,
                        random_state=0)
        kmeans.fit(image_2d)
        
        return kmeans

    def get_player_color(self, frame, player_bbox): 
        image = frame[int(player_bbox[1]):int(player_bbox[3]), int(player_bbox[0]):int(player_bbox[2])]
        top_half = image[:int(image.shape[0]/2), :]

        # Get the average color of the top half of the player
        kmeans = self.get_cluster_model(top_half, 2)

        # Get cluster lables 
        labels = kmeans.labels_

        # Reshape labels
        cluster_labels = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Get player (non-corner) clusters
        corner_clusters = [cluster_labels[0,0], cluster_labels[0,-1], cluster_labels[-1,0], cluster_labels[-1,-1]]
        corner_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-corner_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color

    def assign_teams(self, frame, player_detections):
        # Assign teams to players
        player_colors = []
        for _,player_detection in player_detections.items():
            player_bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, player_bbox)
            player_colors.append(player_color)

        # Perform KMeans clustering on player colors
        kmeans = KMeans(n_clusters=2, 
                        init="k-means++", # faster runs
                        n_init=1,
                        random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id): 
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team+=1 # we want 1 and 2

        self.player_team_dict[player_id] = team

        return team
    


    