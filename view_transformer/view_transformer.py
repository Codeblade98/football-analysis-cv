import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 35

        self.pixel_vertices = np.array([
            [0,994],
            [1918,865],
            [503,246],
            [1126,238]
        ]) # trapezoid

        self.target_vertices = np.array([
            [0,0],
            [0,court_length],
            [court_width,0],
            [court_width,court_length]
        ]) # birds-eye rectangle

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[0]))
        # is_in_court = cv2.pointPolygonTest(self.pixel_vertices, p, False)>=0
        reshaped = point.reshape(1, -1, 2).astype(np.float32)
        point_transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer)
        return point_transformed.reshape(-1,2)
        # return None

    def add_transform_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    # print(position_transformed)
                    if position_transformed is not None:
                        tracks[object][frame_num][track_id]['position_transformed'] = position_transformed

