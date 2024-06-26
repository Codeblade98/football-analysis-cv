from ultralytics import YOLO
import os
import supervision as sv
import pickle
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def interpolate_ball_positions(self,ball_positions):
        # print(ball_positions)
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        # bp = [x for x in ball_positions if len(x.keys())>0]
        # print(df_ball_positions)
        # Interpolate missing ball positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        # df_ball_positions = df_ball_positions.ffill()

        # print(df_ball_positions)

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        # print(ball_positions)

        return ball_positions  

    def detect_frames(self,frames):
        # frames=frames[:40]
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detection
            # print(detection[-1])

        return detections
    
    def get_object_tracks(self,frames,read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        tracks = {
            'players': [], #{frame_num: {track_id: {"bbox": [x1,y1,x2,y2]}} 
            'ball': [],
            'referees': []
        }
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} # reverse key->value mapping

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeepers to players
            # print("Class names\n=============================")
            # print(cls_names)
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if class_id == cls_names_inv['goalkeeper']:
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}
                
                elif class_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        # print(detection_with_tracks)
        return tracks

    def add_object_positions_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= (int(bbox[0]+bbox[2])//2, int(bbox[1]+bbox[3])//2)
                    else:
                        position = (int(bbox[0])+int(bbox[2])//2, int(bbox[3]))
                    tracks[object][frame_num][track_id]['position'] = position

    def draw_ellipse(self,frame,pt1,pt2,color,thickness,track_id=None):
        x1,y1 = pt1
        x2,y2 = pt2
        center = ((x1+x2)//2, (y1+y2)//2)
        axes = ((x2-x1)//2, (y2-y1)//2)
        cv2.ellipse(frame, 
                    center=(int(center[0]),int(y2)),
                    axes=(int(axes[0]+8),int(0.5*axes[0])+1), 
                    angle=0.0, 
                    startAngle=-30, 
                    endAngle=300, 
                    color=color, 
                    thickness=thickness)
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(center[0]) - rectangle_width//2        
        x2_rect = int(center[0]) + rectangle_width//2

        y1_rect = int(y2) - rectangle_height//2 + 15
        y2_rect = int(y2) + rectangle_height//2 + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect),int(y1_rect)), (int(x2_rect),int(y2_rect)),
                           color, 
                           cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id<10:
                x1_text += 5
            if track_id>99:
                x1_text -=12

            cv2.putText(frame, 
                        str(track_id), 
                        (int(x1_text), int(y2)+20),
                          cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, 
                            (255-color[0],255-color[1],255-color[2]), 
                            2)
        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x1 = int(bbox[0])
        x2 = int(bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[3])
        x,_ = ((x1+x2)//2, (y1+y2)//2)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames,tracks):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()
            ## to prevent changes in the original frame

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            ## Draw players
            for track_id, player in player_dict.items():
                x1,y1,x2,y2 = player['bbox']
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, (x1,y1), (x2,y2), color, 2,track_id=track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player['bbox'], (255,0,0))
            ## Draw referees
            for track_id, referee in referee_dict.items():
                x1,y1,x2,y2 = referee['bbox']
                frame = self.draw_ellipse(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            
            ## Draw ball
            for track_id, ball in ball_dict.items():
                x1,y1,x2,y2 = ball['bbox']
                frame = self.draw_traingle(frame, ball['bbox'], (0,255,0))
                # player = ball.get('player_id',None)
                
            output_video_frames.append(frame)
        return output_video_frames
 