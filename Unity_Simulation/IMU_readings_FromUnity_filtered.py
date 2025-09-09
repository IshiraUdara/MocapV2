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
try:
    from EKF import ExtendedKalmanFilter
    from MathLib import *
    print("Successfully imported EKF modules")
except ImportError as e:
    print(f"Error importing EKF modules: {e}")
    sys.exit(1)

import math

UDP_IP = "127.0.0.1"
UDP_PORT = 5002

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
            
            # Create measurement dictionary matching your EKF format
            measurement = {
                'accel': np.array(accel, dtype=np.float64),
                'gyro': np.array(gyro, dtype=np.float64),
                'mag': np.array([0, 0, 0], dtype=np.float64),  # Unity doesn't provide magnetometer
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

def ekf_filter_thread():
    """Thread to process IMU data through your custom EKF"""
    print("EKF filter thread started...")
    
    # Initialize your custom EKF with proper dimensions
    try:
        # Based on typical IMU EKF: 16 states, 6 measurements
        # State: [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
        # Measurements: [ax, ay, az, gx, gy, gz] (accel + gyro)
        ekf = ExtendedKalmanFilter(dim_x=16, dim_u=6)
        
        # Initialize state vector
        ekf.x = np.zeros(16)
        ekf.x[0] = 1.0  # Initialize quaternion w component
        
        # Initialize covariance matrix
        ekf.P = np.eye(16)
        ekf.P[0:4, 0:4] *= 0.1    # Quaternion uncertainty
        ekf.P[4:7, 4:7] *= 1.0    # Position uncertainty
        ekf.P[7:10, 7:10] *= 0.1  # Velocity uncertainty
        ekf.P[10:13, 10:13] *= 0.01  # Gyro bias uncertainty
        ekf.P[13:16, 13:16] *= 0.01  # Accel bias uncertainty
        
        # Process noise covariance
        ekf.Q = np.eye(16) * 0.001
        ekf.Q[0:4, 0:4] *= 0.0001  # Low process noise for quaternion
        ekf.Q[4:7, 4:7] *= 0.01    # Position process noise
        ekf.Q[7:10, 7:10] *= 0.01  # Velocity process noise
        ekf.Q[10:16, 10:16] *= 0.00001  # Very low bias process noise
        
        # Measurement noise covariance
        ekf.R = np.eye(6)
        ekf.R[0:3, 0:3] *= 0.1   # Accelerometer noise
        ekf.R[3:6, 3:6] *= 0.05  # Gyroscope noise
        
        print("Available EKF methods:", [method for method in dir(ekf) if not method.startswith('_')])
        print("EKF initialized successfully")
        print(f"State dimension: {ekf.x.shape}")
        print(f"P matrix shape: {ekf.P.shape}")
        print(f"Q matrix shape: {ekf.Q.shape}")
        print(f"R matrix shape: {ekf.R.shape}")
        
    except Exception as e:
        print(f"Error initializing EKF: {e}")
        return
    
    prev_time = None
    iteration_count = 0
    
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
                    iteration_count += 1
                    
                    # Prepare measurement data
                    accel = measurement['accel']
                    gyro = measurement['gyro']
                    
                    try:
                        # Create measurement vector [ax, ay, az, gx, gy, gz]
                        z = np.concatenate([accel, gyro])
                        
                        # Predict step
                        if hasattr(ekf, 'predict'):
                            try:
                                ekf.predict(dt=dt)
                            except TypeError:
                                try:
                                    ekf.predict(dt)
                                except:
                                    ekf.predict()
                        
                        # Update step
                        if hasattr(ekf, 'update'):
                            try:
                                ekf.update(z)
                            except Exception as update_error:
                                print(f"Update error: {update_error}")
                        
                        # Get filtered quaternion from state
                        filtered_quaternion = ekf.x[0:4].copy()
                        
                        # Normalize quaternion
                        norm = np.linalg.norm(filtered_quaternion)
                        if norm > 0:
                            filtered_quaternion = filtered_quaternion / norm
                        
                        # Get other state components
                        position = ekf.x[4:7].copy()
                        velocity = ekf.x[7:10].copy()
                        gyro_bias = ekf.x[10:13].copy()
                        accel_bias = ekf.x[13:16].copy()
                        
                        # Convert quaternion to Euler angles
                        euler = quaternion_to_euler(filtered_quaternion)
                        
                        # Create filtered data structure
                        filtered_data = {
                            'quaternion': filtered_quaternion.tolist(),
                            'euler': euler,
                            'position': position.tolist(),
                            'velocity': velocity.tolist(),
                            'gyro_bias': gyro_bias.tolist(),
                            'accel_bias': accel_bias.tolist(),
                            'raw_accel': accel.tolist(),
                            'raw_gyro': gyro.tolist(),
                            'timestamp': current_time,
                            'filter_type': 'EKF',
                            'iteration': iteration_count
                        }
                        
                        # Add to filtered queue
                        try:
                            if filtered_queue.full():
                                filtered_queue.get_nowait()
                            filtered_queue.put_nowait(filtered_data)
                        except queue.Empty:
                            pass
                        
                    except Exception as e:
                        print(f"\nError in EKF processing (iteration {iteration_count}): {e}")
                        # Continue with raw data when EKF fails
                        raw_data = {
                            'quaternion': [1, 0, 0, 0],
                            'euler': [0, 0, 0],
                            'position': [0, 0, 0],
                            'velocity': [0, 0, 0],
                            'gyro_bias': [0, 0, 0],
                            'accel_bias': [0, 0, 0],
                            'raw_accel': accel.tolist(),
                            'raw_gyro': gyro.tolist(),
                            'timestamp': current_time,
                            'filter_type': 'RAW-Error',
                            'iteration': iteration_count,
                            'error': str(e)
                        }
                        
                        try:
                            if filtered_queue.full():
                                filtered_queue.get_nowait()
                            filtered_queue.put_nowait(raw_data)
                        except queue.Empty:
                            pass
                
                prev_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
            
        except Exception as e:
            print(f"Error in EKF filter thread: {e}")

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
    if len(q) != 4:
        return [0, 0, 0]
    
    # Handle [w,x,y,z] format
    w, x, y, z = q[0], q[1], q[2], q[3]
    
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
                filter_type = filtered_data.get('filter_type', 'Unknown')
                euler = filtered_data.get('euler', [0, 0, 0])
                position = filtered_data.get('position', [0, 0, 0])
                gyro_bias = filtered_data.get('gyro_bias', [0, 0, 0])
                accel_bias = filtered_data.get('accel_bias', [0, 0, 0])
                raw_accel = filtered_data.get('raw_accel', [0, 0, 0])
                raw_gyro = filtered_data.get('raw_gyro', [0, 0, 0])
                iteration = filtered_data.get('iteration', 0)
                
                print(f"\r[{iteration:4d}] ║ {filter_type:10s} ║ Orientation (R/P/Y): [{euler[0]:6.1f}°, {euler[1]:6.1f}°, {euler[2]:6.1f}°] "
                      f"║ Position: [{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f}] "
                      f"║ Gyro Bias: [{gyro_bias[0]:6.3f}, {gyro_bias[1]:6.3f}, {gyro_bias[2]:6.3f}] ║", end="")
                
            time.sleep(0.05)  # Display update rate
            
        except Exception as e:
            print(f"Error in display thread: {e}")

if __name__ == "__main__":
    # Start threads - using EKF instead of simple filtering
    receiver_thread = threading.Thread(target=udp_receiver_thread, daemon=True)
    ekf_thread = threading.Thread(target=ekf_filter_thread, daemon=True)
    display_thread_obj = threading.Thread(target=display_thread, daemon=True)
    
    receiver_thread.start()
    ekf_thread.start()
    display_thread_obj.start()
    
    print("IMU EKF filtering started. Press Ctrl+C to exit.")
    print("=" * 150)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")