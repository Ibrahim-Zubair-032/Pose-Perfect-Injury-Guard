from flask import Flask , request, jsonify, send_from_directory, send_file 
import os
from plank import process_video as plank_video , generate_plank_injury_estimation_report as plank_report ,get_report as plank_report_call
from plank import (PROCESSED_VIDEOS_FOLDER as Plank_Videos ,REPORT_FOLDER as Plank_Report)
from bicep import analyze_bicepCurl as bicep_video ,get_report as bicep_report_call
from bicep import (PROCESSED_VIDEOS_FOLDER as Bicep_Videos,REPORT_FOLDER as Bicep_Report) 
from squat import(PROCESSED_VIDEOS_FOLDER as Squat_Videos)
from squat import analyze_squat as squat_video , get_report as squat_report_call
from lunges import analyze_lunge as lunges_video ,get_report as lunges_report_call
from lunges import(PROCESSED_VIDEOS_FOLDER as Lunges_Videos)
app=Flask(__name__)



@app.route('/plank-analyze', methods=['POST'])
def plankAnalyze():
    return plank_video()


@app.route('/plank-processed_videos/<path:filename>')
def plank_processed_video(filename):
    return send_from_directory(Plank_Videos, filename, mimetype='video/mp4')



@app.route('/plank-get-report', methods=['GET'])
def plank_report():
    return plank_report_call()


@app.route('/bicep-analyze', methods=['POST'])
def bicepAnalyze():
    return bicep_video()


@app.route('/bicep-processed_videos/<path:filename>')
def bicep_processed_video(filename):
    return send_from_directory(Bicep_Videos, filename, mimetype='video/mp4')
@app.route('/bicep-get-report', methods=['GET'])
def bicep_report():
    return bicep_report_call()



@app.route('/squat-analyze', methods=['POST'])
def squatAnalyze():
    return squat_video()

@app.route('/processed_videos/<path:filename>')
def squat_processed_video(filename):
    return send_from_directory(Squat_Videos, filename, mimetype='video/mp4')

@app.route('/squat-get-report', methods=['GET'])
def squat_report():
    return squat_report_call()

@app.route('/lunge-analyze', methods=['POST'])
def lungeAnalyze():
    return lunges_video()

@app.route('/processed_videos/<path:filename>')
def lunges_processed_video(filename):
    return send_from_directory(Lunges_Videos, filename, mimetype='video/mp4')

@app.route('/lunges-get-report', methods=['GET'])
def lunges_report():
    return lunges_report_call()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
