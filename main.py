from utils import read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistance

import os
from datetime import datetime
import cv2

import pathlib
pathlib.Path().resolve()

def main():
    # Read video
    # print(os.path.exists('bundesliga_cv_project/input_video/A1606b0e6_0 (18).mp4'))
    print('Loading video...')
    time1 = datetime.now()
    input_video_name = 'C35bd9041_0 (57)'
    video_frames = read_video(f'bundesliga_cv_project/input_video/{input_video_name}.mp4')
    time2 = datetime.now()
    print("Video Loaded in ", time2-time1)

    # Initialize tracker
    print('Initializing tracker...')
    tracker = Tracker('bundesliga_cv_project/models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=f'bundesliga_cv_project/stubs/track_stubs_{input_video_name}.pkl')
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    time3 = datetime.now()
    print("Tracker initialized in ", time3-time2)

    # Get object positions
    print('Adding object positions to tracks...')
    tracker.add_object_positions_to_tracks(tracks)
    time4 = datetime.now()
    print("Object positions added in ", time4-time3)

    # Camera movement estimation
    print('Estimating camera movement...')
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path=f'bundesliga_cv_project/stubs/camera_movement_stubs_{input_video_name}.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement)
    time5 = datetime.now()
    print("Camera movement estimated in ", time5-time4)

    # View transformation
    print('Transforming view...')
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_tracks(tracks)
    time6 = datetime.now()
    print("View transformed in ", time6-time5)

    # Initialize assigner with first frame
    print('Assigning teams...')
    team_assigner = TeamAssigner()
    team_assigner.assign_teams(video_frames[0], tracks['players'][0])

    # Assign teams to players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], 
                track['bbox'], 
                player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    time7 = datetime.now()
    print("Teams assigned in ", time7-time6)

    
    # Assign speed and distance estimator
    print('Estimating speed and distance...')
    speed_distance_estimator = SpeedDistance()
    speed_distance_estimator.add_speed_dist_to_tracks(tracks)
    time8 = datetime.now()
    print("Speed and distance estimated in ", time8-time7)

    # Assign Ball position
    print('Assigning ball to player...')
    # player_ball_assigner = PlayerBallAssigner()
    # for frame_num, player_track in enumerate(tracks['players']):
    #     ball_bbox = tracks['ball'][frame_num].get(1,{}).get('bbox',[])
    #     player_id = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
    #     if player_id != -1:
    #         tracks['players'][frame_num][player_id]['has_ball'] = True
    time9 = datetime.now()
    print("Ball assigned to player in ", time9-time8)



    # Save cropped image of a player
    # for track_id,player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # crop bbox from frame
    #     cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save cropped frame
    #     cv2.imwrite(f'bundesliga_cv_project/output_images/player_{track_id}.jpg', cropped_frame)

    # Draw output
    print('Drawing output...')
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement)
    ## Draw speed and distance
    output_video_frames = speed_distance_estimator.draw_speed_distance(output_video_frames, tracks)
    time10 = datetime.now()
    print("Output drawn in ", time10-time9)

    # Save video
    print('Saving video...')
    output_file_name = f"bundesliga_cv_project/output_videos/output-video.avi"
    while os.path.exists(output_file_name):
        output_file_name = output_file_name[:-4]
        output_file_name+='-new.avi'

    # print(os.path.exists("bundesliga_cv_project/output_videos"))
    save_video(output_video_frames, output_file_name)
    time11 = datetime.now()
    print(f"Video saved as {output_file_name} in ", time11-time10)

if __name__=='__main__': 
    main()