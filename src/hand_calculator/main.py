"""
Main entry point for the hand gesture calculator.
Handles video capture, gesture processing, and UI rendering.
"""

import cv2
import time
import argparse
from .hands import HandTracker
from .gestures import GestureRecognizer
from .evaluator import ExpressionEvaluator
from .overlay import UIOverlay

def main():
    parser = argparse.ArgumentParser(description='Hand Gesture Calculator')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    args = parser.parse_args()
    
    # Initialize components
    hand_tracker = HandTracker()
    gesture_recognizer = GestureRecognizer()
    evaluator = ExpressionEvaluator()
    ui_overlay = UIOverlay()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Hand Gesture Calculator Started!")
    print("Controls:")
    print("- Show finger gestures for numbers (1-9)")
    print("- Use specific gestures for +, -, *, /, =, C")
    print("- Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process hands
            landmarks = hand_tracker.process_frame(frame)
            
            current_gesture = None
            if landmarks:
                # Get gesture from first hand detected
                current_gesture = gesture_recognizer.recognize_gesture(landmarks[0])
            
            # Update gesture state and get token
            token = gesture_recognizer.update_gesture_state(current_gesture)
            
            # Process token if stable
            if token:
                if token == "=":
                    result = evaluator.evaluate_expression()
                    print(f"Result: {result}")
                elif token == "C":
                    evaluator.clear_expression()
                    print("Cleared")
                elif token == "⌫":
                    evaluator.backspace()
                else:
                    evaluator.add_token(token)
                    print(f"Added: {token}")
            
            # Draw UI overlay
            ui_overlay.draw_overlay(
                frame, 
                evaluator.get_current_expression(),
                evaluator.get_last_result(),
                current_gesture,
                gesture_recognizer.get_stability_info()
            )
            
            # Draw hand landmarks
            if landmarks:
                hand_tracker.draw_landmarks(frame, landmarks[0])
            
            # Display frame
            cv2.imshow('Hand Gesture Calculator', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
