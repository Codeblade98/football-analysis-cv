import cv2 

class SpeedDistance():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_dist_to_tracks(self,tracks):
        total_distance={}
        for obj, obj_tracks in tracks.items():
            if obj in ['referees','ball','referee']:
                continue
            num_frames = len(obj_tracks)
            for frame_num in range(0,num_frames,self.frame_window):
                last_frame = min(frame_num+self.frame_window,num_frames-1)
                # just to not go out of bounds

                for track_id, track_info in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    start_position = obj_tracks[frame_num][track_id].get('position_transformed',None)
                    end_position = obj_tracks[last_frame][track_id].get('position_transformed',None)

                    if start_position is None or end_position is None:
                        continue

                    distnce_covered = ((start_position[0][0]-end_position[0][0])**2+(start_position[0][1]-end_position[0][1])**2)**0.5
                    time = (last_frame-frame_num)/self.frame_rate
                    speed_metres_per_sec = distnce_covered/time
                    speed_kmph = speed_metres_per_sec*3.6

                    if obj not in total_distance:
                        total_distance[obj]={}

                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id]=0

                    total_distance[obj][track_id]+=distnce_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        tracks[obj][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]
    
    def draw_speed_distance(self,frames,tracks):
        output_frames = []
        for frame_num,frame in enumerate(frames):
            frame = frame.copy()
            for obj, obj_tracks in tracks.items():
                if object in ['referees','ball','referee']:
                    continue
                for track_id, track_info in obj_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed',None)
                        distance = track_info.get('distance',None)
                        if speed is None or distance is None:
                            continue
                        bbox = track_info['bbox']
                        cv2.putText(frame, 
                                    f"{speed:.2f} kmph", 
                                    (int(bbox[0]+bbox[2])//2,int(bbox[3])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (0,0,0), 
                                    2)
                        # cv2.putText(frame, 
                        #             f"{distance:.2f} m", 
                        #             (int(bbox[0]+bbox[2])//2,int(bbox[1]+20)), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 
                        #             1, 
                        #             (0,0,0), 
                        #             2)
            output_frames.append(frame)
        return output_frames



    