from collections import OrderedDict
import threading

from centroid_tracker import CentroidTracker

class AppState:
    def __init__ (self):
        self.tracker = CentroidTracker(
            max_disappeared=20,
            tight_distance_ratio=0.15,
            max_distance_ratio=0.35,
            hit_streak_required=3,
            velocity_decay=0.5,
            edge_margin=20,
            next_id_counter=1,
            max_color_distance=90.0,
            iou_suppresion_thresh=0.6
        )
        self.target_id = None
        self.drone_state = "MANUAL"
        
        self.frame = None
        self.last_frame_time = 0.0
        self.tracked = OrderedDict()
        self.hover_lost_time = None
        self.hold_start_time = None
        
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.target_lock = threading.Lock() 
        self.tracker_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.airborne = False