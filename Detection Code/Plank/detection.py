import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
correct = 0
low_back = 0
high_back = 0
# Reconstruct the input structure
# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Setup some important functions
def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

    return np.array(data).flatten().tolist()

def generate_plank_injury_estimation_report(correct_counter, low_back_counter, high_back_counter):
    # Round the counter values for plank positions
    correct_counter_rounded = round(correct_counter)
    low_back_counter_rounded = round(low_back_counter)
    high_back_counter_rounded = round(high_back_counter)

    # Calculate total evaluations
    total_evaluations = correct_counter_rounded + low_back_counter_rounded + high_back_counter_rounded
    
    # Calculate percentages for each stage
    correct_percentage = (correct_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0
    low_back_percentage = (low_back_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0
    high_back_percentage = (high_back_counter_rounded / total_evaluations) * 100 if total_evaluations > 0 else 0

    # Round percentages to whole numbers
    correct_percentage = round(correct_percentage)
    low_back_percentage = round(low_back_percentage)
    high_back_percentage = round(high_back_percentage)

    # Start building the report content
    report_content = "Plank Injury Estimation Report\n"
    report_content += "=" * 50 + "\n"
    
    # Correct Position Feedback
    if correct_counter_rounded > 0:
        report_content += f"CORRECT: {correct_percentage}%\n"
        report_content += "✔ Excellent core engagement\n"
        report_content += "✔ Reduced injury risk\n"
        report_content += "✔ Maintain a straight line from head to heels\n\n"
    
    # Low Back Feedback
    if low_back_counter_rounded > 0:
        report_content += f"LOW BACK: {low_back_percentage}%\n"
        report_content += "❗ Risk of lower back injury (compression)\n"
        report_content += "❗ Limited core activation\n"
        report_content += "✔ Lift your hips and align your back\n\n"
    
    # High Back Feedback
    if high_back_counter_rounded > 0:
        report_content += f"HIGH BACK: {high_back_percentage}%\n"
        report_content += "❗ Increased strain on the lower back\n"
        report_content += "❗ Risk of back strain\n"
        report_content += "✔ Engage your core and avoid excessive arching\n\n"

    # Save the report to a text file with UTF-8 encoding
    with open("plank_injury_estimation_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report_content)

    print("Plank Injury Estimation Report generated successfully!")

def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compared to its original frame
    '''
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# VIDEO_PATH1 = "../data/plank/plank_test.mov"
# VIDEO_PATH2 = "../data/plank/plank_test_1.mp4"
# VIDEO_PATH3 = "../data/plank/plank_test_2.mp4"
# VIDEO_PATH4 = "../data/plank/plank_test_3.mp4"
# VIDEO_PATH5 = "../data/plank/plank_test_4.mp4"
VIDEO_TEST = r"D:\Desktop\PlankFYPModel\Uploads\WhatsApp Video 2024-12-03 at 13.51.23_334c9901.mp4"

# 1. Make detection with Scikit-learn model
# Load model
with open("D:\Desktop\PlankFYPModel\Model\LR_model.pkl", "rb") as f:
    sklearn_model = pickle.load(f)

# Load input scaler
with open("D:\Desktop\PlankFYPModel\Model\input_scaler.pkl", "rb") as f2:
    input_scaler = pickle.load(f2)

# Transform prediction into class
def get_class(prediction: float) -> str:
    return {
        0: "C",
        1: "H",
        2: "L",
    }.get(prediction)

cap = cv2.VideoCapture(VIDEO_TEST)
current_stage = ""
prediction_probability_threshold = 0.6

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 20)
        # image = cv2.flip(image, 1)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler.transform(X))

            # Make prediction and its probability
            predicted_class = sklearn_model.predict(X)[0]
            predicted_class = get_class(predicted_class)
            prediction_probability = sklearn_model.predict_proba(X)[0]
            # print(predicted_class, prediction_probability)
            # Evaluate model prediction
            if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                current_stage = "Correct"
                correct +=1
            elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                current_stage = "Low back"
                low_back +=1
            elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                current_stage = "High back"
                high_back+=1
            else:
                current_stage = " "

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            # Display class
            cv2.putText(image, "Stage", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("CV2", image)

        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    generate_plank_injury_estimation_report(correct,low_back,high_back)
    for i in range(1, 5):
        cv2.waitKey(1)
