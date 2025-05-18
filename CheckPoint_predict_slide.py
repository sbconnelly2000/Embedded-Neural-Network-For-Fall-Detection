import time
import numpy as np
from sense_hat import SenseHat
import tflite_runtime.interpreter as tflite

# Initialize the sliding window with initial data
window_size = 15              #CHANGE IF NEEDED

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="Check_Point.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize SenseHat
sense = SenseHat()

# Function to get gyro and accel data from SenseHat
def get_sensor_data():
    gyro = sense.get_gyroscope_raw()
    accel = sense.get_accelerometer_raw()
    return [gyro['x'], gyro['y'], gyro['z'], accel['x'], accel['y'], accel['z']]

data_window = [get_sensor_data() for _ in range(window_size)]

# Function to update the sliding window with new data
def update_data_window():
    new_data = get_sensor_data()
    data_window.pop(0)  # Remove the oldest data point
    data_window.append(new_data)  # Append the newest data point

# Function to perform inference
def perform_inference():
    input_data = np.array(data_window).reshape(1, window_size, 6)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to update SenseHat color based on prediction
def update_sensehat_color(prediction):
    # Define colors
    colors = {
        0: (0, 0, 255),    # Blue for sitting
        1: (255, 255, 0),  # Yellow for standing
        2: (0, 255, 0),    # Green for walking
        3: (255, 0, 0)     # Red for falling
    }
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    # Set the SenseHat color
    sense.clear(colors[predicted_class])

# Perform inference and print the result
while True:
    start_time = time.time()
    update_data_window()  # Update the sliding window with new data
    result = perform_inference()
    end_time = time.time()
    print("Inference result:", result)
    print("Inference time: ", end_time - start_time)
    update_sensehat_color(result)
