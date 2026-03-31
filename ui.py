import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

COLOR_CONFIRMED = (0, 220, 0) # Green bounding box for confirmed detections
COLOR_TENTATIVE = (180, 180, 0) # Yellow bounding box for tentative detection trackss
COLOR_TARGET = (0, 80, 255) # Red bounding box for selected target
COLOR_HUD = (240, 240, 240)

def draw_overlay(frame, tracked, target_id, state, battery):
    if target_id is None:
        target_id = 'none'
    
    for object_id, data in tracked.items():
        bbox = data.get('bbox')
        label = data.get('label')
        conf = data.get('confidence', 0.0)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        
        is_target = (object_id == target_id)
        
        color = COLOR_TARGET if is_target else COLOR_CONFIRMED
        
        thickness = 3 if is_target else 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        tag = f"ID:{object_id} {label} {conf:.2f}"
        
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1 + 2, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, tag, (x1 + 2, y1 - 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        if is_target:
            cx, cy = data['centroid']
            cv2.drawMarker(
                frame, (cx, cy), COLOR_TARGET, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA
            )
        
        hud_lines = [
            f"STATE: {state}"
            f" TARGET: {target_id}"
            f" BATTERY: {battery}"
            " T = takeoff | L = land | C = clear | Q = quit"
        ]
        
        for i, line in enumerate(hud_lines):
            cv2.putText(
                frame, line, (10, 25 + i * 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 1, cv2.LINE_AA
            )
        
    return frame

def make_mouse_callback(app_state):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with app_state.tracker_lock:
            tracked = dict(app_state.tracked)
        with app_state.target_lock:
            for object_id, data in tracked.items():
                bbox = data.get('bbox')
                if bbox and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    app_state.target_id = object_id
                    with app_state.state_lock:
                        app_state.drone_state = "TRACKING"
                    logging.info(f"Target acquired: ID {object_id} ({data.get('label')})")
                    return
            app_state.target_id = None
            with app_state.state_lock:
                app_state.drone_state = "MANUAL"
            logging.info("Target cleared.")
            return
    return on_mouse
                
        
        
        
        
        
        