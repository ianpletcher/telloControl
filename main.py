from djitellopy import Tello
import threading
import cv2
import time
import sys
from ultralytics import YOLO
import numpy as np
import logging

from app_state import AppState
from ui import make_mouse_callback, draw_overlay
from yolov8_inference import run_yolo_inference
from ctrl import run_control_loop

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Video frame config:
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
FPS_TARGET = 30

# YOLO model
YOLO_MODEL = 'yolov8s.pt'

# OpenCV window name
WINDOW_NAME = 'Tello POC - Click to Track'



def main():
    app_state = AppState()
    
    logging.info("Loading YOLOv8 model...")
    model = YOLO(YOLO_MODEL)
    logging.info(f"{YOLO_MODEL} loaded successfully!")
    
    logging.info("Connecting to Tello...")
    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    logging.info(f"Connected, Battery: {battery}%")
    
    if battery < 20:
        logging.warning(f"Low battery!")
        
    tello.streamon()
    frame_read = tello.get_frame_read()
    
    app_state.airborne = False
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(app_state))
    
    control_thread = threading.Thread(
        target=run_control_loop,args=(tello, app_state), daemon=True
    )
    
    control_thread.start()
    
    logging.info(f"Running. T=takeoff, L=land, Q=Quit. Click box to track")
    
    try:
        while True:
            raw = frame_read.frame
            if raw is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))
            
            detections = run_yolo_inference(model, frame, app_state)
            
            current_target = None
            
            with app_state.tracker_lock:
                if app_state.target_id:
                    current_target = app_state.target_id
                    tracked_objects = app_state.tracker.update_target(
                        detections, current_target, FRAME_WIDTH, FRAME_HEIGHT
                    )
                else:
                    tracked_objects = app_state.tracker.update_all_detections(
                        detections, FRAME_WIDTH, FRAME_HEIGHT
                    )
            
            state = app_state.drone_state
            
            display = draw_overlay(frame.copy(), app_state.tracked, current_target, state, battery)
            
            with app_state.frame_lock:
                app_state.frame = frame
                
            cv2.imshow(WINDOW_NAME, display)
            
            if int(time.time()) % 30 == 0:
                try:
                    battery = tello.get_battery()
                except Exception:
                    logging.warning(f"Couldn't refresh battery. Last known battery: {battery}")
                    pass
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('t') or key == ord('T'):
                if not app_state.airborne:
                    logging.info("Takeoff.")
                    tello.takeoff()
                    app_state.airborne = True
                
            elif key == ord('l') or key == ord('L'):
                logging.info("Landing.")
                tello.land()
                app_state.airborne = False
                with app_state.state_lock:
                    app_state.drone_state = "MANUAL"
                with app_state.target_lock:
                    app_state.target_id = None
                    
            elif key == ord('c') or key == ord('C'):
                with app_state.target_lock:
                    app_state.target_id = None
                with app_state.state_lock:
                    app_state.drone_state = "MANUAL"
                logging.info("Target cleared.")
                
            elif key == ord('q') or key == 27:
                logging.info("Quitting...")
                break
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        
    finally:
        app_state.stop_event.set()
        control_thread.join(timeout=2)

        if app_state.airborne:
            logging.info("Landing before exit...")
            try:
                tello.land()
            except Exception as e:
                logging.error(f"Land error: {e}")

        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        logging.info("Done.")
        sys.exit(0)

if __name__ == "__main__":
    main()
                
                
                
    
    
    

    
    

        
    
    
    