from djitellopy import Tello
import threading
import cv2
import time
import sys
from ultralytics import YOLO
from collections import OrderedDict
import logging

from app_state import AppState
from ui import make_mouse_callback, draw_overlay
from yolov8_inference import run_inference_loop
from ctrl import run_control_loop


logging.basicConfig(filename = 'app.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Video frame config:
FRAME_WIDTH = 960 # Tello default camera width
FRAME_HEIGHT = 720 # Tello default camera height
FPS_TARGET = 30

# YOLO model
YOLO_MODEL = 'yolov8s.pt'

# OpenCV window name
WINDOW_NAME = 'Tello POC'

def main():
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
    
    time.sleep(2)  # give stream time to initialize
    
    frame_read = tello.get_frame_read()
    
    logging.info("Waiting for first frame...")
    timeout = time.time() + 10.0
    while frame_read.frame is None:
        if time.time() > timeout:
            logging.error("Timed out waiting for first frame. Check Tello WiFi connection.")
            tello.streamoff()
            tello.end()
            sys.exit(1)
        time.sleep(0.1)
    logging.info("First frame received.")
    
    app_state = AppState()
    
    app_state.airborne = False
    
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(app_state))
    
    control_thread = threading.Thread(
        target=run_control_loop, args=(tello, app_state), daemon=True
    )
    
    inference_thread = threading.Thread(
        target=run_inference_loop, args=(model, frame_read, app_state, FRAME_WIDTH, FRAME_HEIGHT),
        daemon=True
    )
    
    control_thread.start()
    inference_thread.start()
    
    while app_state.frame is None:
        time.sleep(0.1)
    
    logging.info(f"Running. T=takeoff, L=land, Q=Quit. Click box to track")
    
    
    try:
        while True:
            with app_state.frame_lock:
                frame = app_state.frame
            with app_state.state_lock:
                state = app_state.drone_state
                
            with app_state.target_lock:
                current_target = app_state.target_id
                
            with app_state.tracker_lock:
                tracked_snapshot = OrderedDict(app_state.tracked)
            
            display = draw_overlay(frame.copy(), tracked_snapshot, current_target, state, battery)
            
            if display is None:
                logging.warning("Overlay skipped due to error.")
                display = frame.copy()
                
            if time.time() - app_state.last_frame_time > 3.0:
                logging.warning("Video stream stalled...")
                
            cv2.imshow(WINDOW_NAME, display)
            
            # Refresh battery occasionally
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
        logging.info("Stopping main loop...")
        
        control_thread.join(timeout=2)
        inference_thread.join(timeout=2)

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