#!/bin/bash

# Run Python file that starts a Flask app and capture its process ID
python TestEmotionDetector.py &
flask_app_pid=$!

# Wait for the Flask app to start running
sleep 30

# Check if the Flask app is running properly
response=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/video_feed)
if [[ $response -eq 200 ]]; then
  echo "Flask app is running properly"
else
  echo "Flask app is not running properly"
  exit 1
fi

# Stop running the Flask app
curl -X POST http://127.0.0.1:5000/shutdown
kill $flask_app_pid
