import numpy as np

class PlayerBallAssigner():
    def __init__(self,max_player_distance=70):
        self.max_player_distance = max_player_distance

    def assign_ball_to_player(self, player_info,ball_bbox):
        ball_position = ((ball_bbox[0]+ball_bbox[2])//2,(ball_bbox[1]+ball_bbox[3])//2)
        # ball_position = (int(ball_position[0]),int(ball_position[1]))    
        min_dist = 1e9
        closest_player_id = -1
        for player_id,player in player_info.items():
            try:
                player_position = ((player['bbox'][0]+player['bbox'][2])//2,(player['bbox'][1]+player['bbox'][3])//2)
            # player_position = (int(player_position[0]),int(player_position[1]))
            except:
                print(player['bbox'])
                raise(ValueError)

            distance = ((player_position[0]-ball_position[0])**2 + (player_position[1]-ball_position[1])**2)**0.5
            # print(distance)
            if distance<min_dist and distance<self.max_player_distance and min_dist-distance>10:
                min_dist = distance
                closest_player_id = player_id
            return closest_player_id