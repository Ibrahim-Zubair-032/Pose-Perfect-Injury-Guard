import cv2
import math
import pandas as pd
import mediapipe as mp
import pickle
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without feature names")

# Initialize counters for foot placement evaluations
correct_foot_placement_count = 0
too_tight_foot_placement_count = 0
too_wide_foot_placement_count = 0

# Initialize counters for knee placement evaluations
correct_knee_placement_count = 0
too_tight_knee_placement_count = 0
too_wide_knee_placement_count = 0

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the logistic regression model for squat classification
with open("D:\\Desktop\\Squat FYP Model\\Train Model\\logistic_regression_model.pkl", "rb") as f:
    count_model = pickle.load(f)

# List of important landmarks (using the official MediaPipe PoseLandmark enum names)
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# Create headers for the CSV or DataFrame (x, y, z, visibility for each landmark)
headers = ["label"]  # Label column
for lm in IMPORTANT_LMS:
    headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Function to extract keypoints from mediapipe results
def extract_important_keypoints(results) -> list:
    landmarks = results.pose_landmarks.landmark  # Get the list of landmarks
    keypoints = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        keypoints.extend([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return keypoints  # Return the flattened list of x, y, z, v for each landmark

# Function to rescale a frame to a percentage of its original size
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)  # Rescale width
    height = int(frame.shape[0] * percent / 100)  # Rescale height
    dim = (width, height)  # New dimensions
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  # Resize the frame

def calculate_distance(pointX, pointY) -> float:
    x1, y1 = pointX
    x2, y2 = pointY
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
def generate_squat_report():
    report_content = (
        "Final Squat Feedback Report:\n\n"
        "Foot Placement Feedback:\n"
        f" - Correct: {correct_foot_placement_count} times\n"
        f" - Too Tight: {too_tight_foot_placement_count} times\n"
        f" - Too Wide: {too_wide_foot_placement_count} times\n\n"
        "Knee Placement Feedback:\n"
        f" - Correct: {correct_knee_placement_count} times\n"
        f" - Too Tight: {too_tight_knee_placement_count} times\n"
        f" - Too Wide: {too_wide_knee_placement_count} times\n"
    )

def generate_injury_estimation_report():
    # Round the counter values before calculating percentages
    correct_foot_placement_count_rounded = round(correct_foot_placement_count)
    too_tight_foot_placement_count_rounded = round(too_tight_foot_placement_count)
    too_wide_foot_placement_count_rounded = round(too_wide_foot_placement_count)
    
    correct_knee_placement_count_rounded = round(correct_knee_placement_count)
    too_tight_knee_placement_count_rounded = round(too_tight_knee_placement_count)
    too_wide_knee_placement_count_rounded = round(too_wide_knee_placement_count)

    # Foot Placement percentages
    total_foot_evaluations = correct_foot_placement_count_rounded + too_tight_foot_placement_count_rounded + too_wide_foot_placement_count_rounded
    total_knee_evaluations = correct_knee_placement_count_rounded + too_tight_knee_placement_count_rounded + too_wide_knee_placement_count_rounded

    # Calculate percentages for foot placement
    correct_foot_percentage = (correct_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    too_tight_foot_percentage = (too_tight_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    too_wide_foot_percentage = (too_wide_foot_placement_count_rounded / total_foot_evaluations) * 100 if total_foot_evaluations > 0 else 0
    
    # Calculate percentages for knee placement
    correct_knee_percentage = (correct_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0
    too_tight_knee_percentage = (too_tight_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0
    too_wide_knee_percentage = (too_wide_knee_placement_count_rounded / total_knee_evaluations) * 100 if total_knee_evaluations > 0 else 0

    # Start building the report content
    report_content = "Squat Injury Estimation Report\n"
    report_content += "=" * 50 + "\n"

    # FOOT PLACEMENT FEEDBACK
    report_content += "FOOT PLACEMENT FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    
    if correct_foot_percentage > 0:
        report_content += f"CORRECT: {round(correct_foot_percentage)}%\n"
        report_content += "✔ Excellent stability and balance\n"
        report_content += "✔ Efficient push-off for squats\n\n"
    
    if too_tight_foot_percentage > 0:
        report_content += f"TOO TIGHT: {round(too_tight_foot_percentage)}%\n"
        report_content += "❗ Risk of ankle strain due to tight stance\n"
        report_content += "❗ Limited stability and balance\n"
        report_content += "✔ Widen your stance slightly\n\n"
    
    if too_wide_foot_percentage > 0:
        report_content += f"TOO WIDE: {round(too_wide_foot_percentage)}%\n"
        report_content += "❗ Potential hip discomfort due to wide stance\n"
        report_content += "❗ Less efficient push-off for squats\n"
        report_content += "✔ Narrow your stance\n\n"

    # KNEE PLACEMENT FEEDBACK
    report_content += "KNEE PLACEMENT FEEDBACK\n"
    report_content += "-" * 50 + "\n"
    
    if correct_knee_percentage > 0:
        report_content += f"CORRECT: {round(correct_knee_percentage)}%\n"
        report_content += "✔ Excellent knee alignment\n"
        report_content += "✔ Reduced injury risk\n"
        report_content += "✔ Maintain proper knee tracking\n\n"
    
    if too_tight_knee_percentage > 0:
        report_content += f"TOO TIGHT: {round(too_tight_knee_percentage)}%\n"
        report_content += "❗ Increased risk of knee strain\n"
        report_content += "❗ Poor knee tracking\n"
        report_content += "✔ Focus on pushing your knees outward\n\n"
    
    if too_wide_knee_percentage > 0:
        report_content += f"TOO WIDE: {round(too_wide_knee_percentage)}%\n"
        report_content += "❗ Potential knee valgus (knees collapsing inward)\n"
        report_content += "❗ Possible ligament strain\n"
        report_content += "✔ Keep knees aligned with toes\n\n"

    # Save the report to a text file with UTF-8 encoding
    with open("squat_depth_feedback_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report_content)

    print("Squat Injury Estimation Report generated successfully!")

def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    '''
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis
    
    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    '''
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    # * Calculate shoulder width
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)

    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2
    
    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        print("Cannot see foot")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)

    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("up")
    middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("down")

    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    
    return analyzed_results

# Video path
VIDEO_DEMO_PATH = r"D:\Desktop\Squat FYP Model\8435486-uhd_2160_4096_25fps.mp4"

# VIDEO_DEMO_PATH = r"D:\Desktop\Squat FYP Model\6875569-hd_1080_1920_25fps.mp4"

#VIDEO_DEMO_PATH = r":\Desktop\Squat FYP Model\4265287-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(VIDEO_DEMO_PATH)

# Counter vars
counter = 0
current_stage = ""
PREDICTION_PROB_THRESHOLD = 0.6

# Error vars
VISIBILITY_THRESHOLD = 0.6
FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down": [0.7, 1.1],
}

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break
        
        # Reduce size of a frame
        image = rescale_frame(image, 20)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        if not results.pose_landmarks:
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            # * Model prediction for SQUAT counter
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=headers[1:])

            # Make prediction and its probability
            predicted_class = count_model.predict(X)[0]
            predicted_class = "down" if predicted_class == 0 else "up"
            prediction_probabilities = count_model.predict_proba(X)[0]
            prediction_probability = round(prediction_probabilities[prediction_probabilities.argmax()], 2)
            # Evaluate model prediction
            if predicted_class == "down" and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                current_stage = "down"
            elif current_stage == "down" and predicted_class == "up" and prediction_probability >= PREDICTION_PROB_THRESHOLD: 
                current_stage = "up"
                # print("Counter statement Executed")
                # print(counter)
                counter += 1

            # Analyze squat pose
            analyzed_results = analyze_foot_knee_placement(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)

            foot_placement_evaluation = analyzed_results["foot_placement"]
            knee_placement_evaluation = analyzed_results["knee_placement"]
            

            # * Evaluate FOOT PLACEMENT error
            if foot_placement_evaluation == -1:
                foot_placement = " "
            elif foot_placement_evaluation == 0:
                foot_placement = "Correct"
                correct_foot_placement_count += 1
            elif foot_placement_evaluation == 1:
                foot_placement = "Too tight"
                too_tight_foot_placement_count += 1
            else:  # foot_placement_evaluation == 2
                foot_placement = "Too wide"
                too_wide_foot_placement_count += 1
            
            # * Evaluate KNEE PLACEMENT error
            if knee_placement_evaluation == -1:
                knee_placement = " "
            elif knee_placement_evaluation == 0:
                knee_placement = "Correct"
                correct_knee_placement_count += 1
            elif knee_placement_evaluation == 1:
                knee_placement = "Too tight"
                too_tight_knee_placement_count += 1
            else:  # knee_placement_evaluation == 2
                knee_placement = "Too wide"
                too_wide_knee_placement_count += 1

            
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

            # Display class
            cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'{str(counter)}' , (25, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "STAGE", (105, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f'{str(predicted_class)}' , (110, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
            # Display Foot and Shoulder width ratio
            # Display foot placement and counters
            cv2.putText(image, "FOOT", (210, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, foot_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # cv2.putText(image, f"Correct: {correct_foot_placement_count}", (195, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, f"Too Tight: {too_tight_foot_placement_count}", (195, 85), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(image, f"Too Wide: {too_wide_foot_placement_count}", (195, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Display knee placement and counters
            cv2.putText(image, "KNEE", (350, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, knee_placement, (335, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # cv2.putText(image, f"Correct: {correct_knee_placement_count}", (335, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, f"Too Tight: {too_tight_knee_placement_count}", (335, 85), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(image, f"Too Wide: {too_wide_knee_placement_count}", (335, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Error: {e}")
        
        cv2.imshow("CV2", image)
        
        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Call the function to generate the report
    generate_squat_report()
    # Generate the report
    generate_injury_estimation_report()
    for i in range (1, 5):
        cv2.waitKey(1)
    #     key = cv2.waitKey(0)  # Wait indefinitely for a key press
    
    #     if key == 13:  # If Enter key is pressed, move to the next frame
    #         continue
    #     else:  # If Escape key is pressed, exit the loop
    #         break
    
    # # Release the capture and close windows
    # cap.release()
    # cv2.destroyAllWindows()
    # #Call the function to generate the report
    # generate_squat_report()
    # #Generate the report
    # generate_injury_estimation_report()
    # # Add a small wait after closing the window to ensure the program ends properly
    # for i in range(1, 5):
    #     cv2.waitKey(1)