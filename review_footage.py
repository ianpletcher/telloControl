import cv2
import logging

videos = []

def main():
    print("---Video Player---")
    choice = int(input("""Choose a video to review:
                    -1: Most recent.
                    Index: Video at index.
                   """))
    
    review_footage(videos[choice])

def append_to_video_list(video):
    videos.append(video)

def review_footage(frames):
    try:
        for frame in frames:
            cv2.imshow('Review Footage', frame)
        
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                logging.info("Quitting video player...")
                break
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        cv2.destroyWindow('Review Footage')

if __name__ == "__main__":
    main()