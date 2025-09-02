import socket
import json
import numpy as np
import queue
import threading
import time
import sys
import os

# Add the IMU_EKF folder to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'IMU_EKF'))

# Import your custom EKF modules
from EKF import ExtendedKalmanFilter
from MathLib import *
import math

UDP_IP = "127.0.0.1"
UDP_PORT = 5000

# Create queues for passing data between threads
data_queue = queue.Queue(maxsize=10)
filtered_queue = queue.Queue(maxsize=10)

def udp_receiver_thread():
    """Thread to receive UDP data from Unity"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Listening for IMU data on {UDP_IP}:{UDP_PORT}...")
    
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            imu_data = json.loads(data.decode('utf-8'))
            
            # Extract IMU data
            accel = imu_data['accel']  # [ax, ay, az]
            gyro = imu_data['gyro']    # [wx, wy, wz]
            orientation = imu_data['orientation']  # Unity quaternion [x, y, z, w]
            
            # Create measurement dictionary
            measurement = {
                'accel': np.array(accel),
                'gyro': np.array(gyro),
                'mag': np.array([0, 0, 0]),  # Unity doesn't provide magnetometer
                'timestamp': time.time()
            }
            
            # Add to queue (non-blocking)
            try:
                if data_queue.full():
                    data_queue.get_nowait()  # Remove oldest data
                data_queue.put_nowait(measurement)
            except queue.Empty:
                pass
                
        except Exception as e:
            print(f"Error receiving data: {e}")

def simple_madgwick_filter(accel, gyro, dt, beta=0.1):
    """Simple Madgwick-style AHRS filter for quaternion estimation"""
    # Static variables to maintain state between calls
    if not hasattr(simple_madgwick_filter, "q"):
        simple_madgwick_filter.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    
    q = simple_madgwick_filter.q
    
    # Normalize accelerometer measurement
    if np.linalg.norm(accel) == 0:
        return q
    
    accel = accel / np.linalg.norm(accel)
    
    # Gradient decent algorithm corrective step
    f = np.array([
        2*(q[1]*q[3] - q[0]*q[2]) - accel[0],
        2*(q[0]*q[1] + q[2]*q[3]) - accel[1],
        2*(0.5 - q[1]**2 - q[2]**2) - accel[2]
    ])
    
    J = np.array([
        [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
        [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
        [0, -4*q[1], -4*q[2], 0]
    ])
    
    step = J.T @ f
    step = step / np.linalg.norm(step)  # normalize step magnitude
    
    # Compute rate of change of quaternion
    qDot = 0.5 * quaternion_multiply(q, np.array([0, gyro[0], gyro[1], gyro[2]])) - beta * step
    
    # Integrate to yield quaternion
    q = q + qDot * dt
    simple_madgwick_filter.q = q / np.linalg.norm(q)  # normalize quaternion
    
    return simple_madgwick_filter.q

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def filter_thread():
    """Thread to process IMU data through a simple filter"""
    print("Filter thread started...")
    
    prev_time = None
    
    while True:
        try:
            # Get latest measurement
            measurement = None
            try:
                # Drain queue to get most recent data
                while not data_queue.empty():
                    measurement = data_queue.get_nowait()
            except queue.Empty:
                pass
            
            if measurement is not None:
                current_time = measurement['timestamp']
                
                if prev_time is not None:
                    dt = current_time - prev_time
                    
                    # Apply simple complementary filter for orientation
                    accel = measurement['accel']
                    gyro = measurement['gyro']
                    
                    # Use Madgwick filter for orientation estimation
                    filtered_quaternion = simple_madgwick_filter(accel, gyro, dt)
                    
                    # Simple low-pass filter for acceleration
                    alpha = 0.8  # Filter coefficient
                    if not hasattr(filter_thread, "filtered_accel"):
                        filter_thread.filtered_accel = accel.copy()
                    
                    filter_thread.filtered_accel = alpha * filter_thread.filtered_accel + (1 - alpha) * accel
                    
                    # Simple low-pass filter for gyroscope
                    if not hasattr(filter_thread, "filtered_gyro"):
                        filter_thread.filtered_gyro = gyro.copy()
                    
                    filter_thread.filtered_gyro = alpha * filter_thread.filtered_gyro + (1 - alpha) * gyro
                    
                    # Convert to Euler angles for easier interpretation
                    euler = quaternion_to_euler(filtered_quaternion)
                    
                    # Create filtered data structure
                    filtered_data = {
                        'quaternion': filtered_quaternion.tolist(),
                        'euler': euler,
                        'filtered_accel': filter_thread.filtered_accel.tolist(),
                        'filtered_gyro': filter_thread.filtered_gyro.tolist(),
                        'raw_accel': accel.tolist(),
                        'raw_gyro': gyro.tolist(),
                        'timestamp': current_time
                    }
                    
                    # Add to filtered queue
                    try:
                        if filtered_queue.full():
                            filtered_queue.get_nowait()
                        filtered_queue.put_nowait(filtered_data)
                    except queue.Empty:
                        pass
                
                prev_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
            
        except Exception as e:
            print(f"Error in filter thread: {e}")

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

def display_thread():
    """Thread to display filtered results"""
    print("Display thread started...")
    
    while True:
        try:
            filtered_data = None
            try:
                # Get latest filtered data
                while not filtered_queue.empty():
                    filtered_data = filtered_queue.get_nowait()
            except queue.Empty:
                pass
            
            if filtered_data is not None:
                quat = filtered_data['quaternion']
                euler = filtered_data['euler']
                filtered_accel = filtered_data['filtered_accel']
                filtered_gyro = filtered_data['filtered_gyro']
                raw_accel = filtered_data['raw_accel']
                raw_gyro = filtered_data['raw_gyro']
                
                print(f"\r║ Filtered Orientation (R/P/Y): [{euler[0]:6.1f}°, {euler[1]:6.1f}°, {euler[2]:6.1f}°] "
                      f"║ Filtered Accel: [{filtered_accel[0]:6.2f}, {filtered_accel[1]:6.2f}, {filtered_accel[2]:6.2f}] "
                      f"║ Filtered Gyro: [{filtered_gyro[0]:6.2f}, {filtered_gyro[1]:6.2f}, {filtered_gyro[2]:6.2f}] ║", end="")
                
            time.sleep(0.1)  # Display update rate
            
        except Exception as e:
            print(f"Error in display thread: {e}")

def udp_send_thread():
    """Thread to send UDP data to Unity (if needed)"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_ip = "127.0.0.1"
    target_port = 5001

    while True:
        try:
            filtered_data = None
            try:
                # Get latest filtered data
                while not filtered_queue.empty():
                    filtered_data = filtered_queue.get_nowait()
            except queue.Empty:
                pass

            if filtered_data is not None:
                # Send filtered data over UDP
                sock.sendto(json.dumps(filtered_data).encode(), (target_ip, target_port))

            time.sleep(0.1)  # UDP send rate

        except Exception as e:
            print(f"Error in UDP send thread: {e}")


if __name__ == "__main__":
    # Start threads
    receiver_thread = threading.Thread(target=udp_receiver_thread, daemon=True)
    filtering_thread = threading.Thread(target=filter_thread, daemon=True)
    display_thread_obj = threading.Thread(target=display_thread, daemon=True)
    udp_send_thread_obj = threading.Thread(target=udp_send_thread, daemon=True)
    
    receiver_thread.start()
    filtering_thread.start()
    display_thread_obj.start()
    udp_send_thread_obj.start()
    
    print("IMU filtering started. Press Ctrl+C to exit.")
    print("=" * 120)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")