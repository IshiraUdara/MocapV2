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

# Receive raw IMU data from Unity
UDP_IP_RECEIVE = "127.0.0.1"
UDP_PORT_RECEIVE = 5003

# Send filtered IMU data back to Unity
UDP_IP_SEND = "127.0.0.1"
UDP_PORT_SEND = 5004  # Different port for sending filtered data back

# Create queues for passing data between threads
data_queue = queue.Queue(maxsize=10)
filtered_queue = queue.Queue(maxsize=10)

def udp_receiver_thread():
    """Thread to receive UDP data from Unity"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP_RECEIVE, UDP_PORT_RECEIVE))
    
    print(f"Listening for IMU data on {UDP_IP_RECEIVE}:{UDP_PORT_RECEIVE}...")
    
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

def udp_sender_thread():
    """Thread to send filtered IMU data back to Unity"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Sending filtered IMU data to {UDP_IP_SEND}:{UDP_PORT_SEND}...")
    
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
                # Prepare data for Unity C# script
                unity_data = {
                    "quaternion": filtered_data.get('quaternion', [1, 0, 0, 0]),
                    "euler": filtered_data.get('euler', [0, 0, 0]),
                    "filtered_accel": [
                        filtered_data.get('raw_accel', [0, 0, 0])[0] - filtered_data.get('accel_bias', [0, 0, 0])[0],
                        filtered_data.get('raw_accel', [0, 0, 0])[1] - filtered_data.get('accel_bias', [0, 0, 0])[1],
                        filtered_data.get('raw_accel', [0, 0, 0])[2] - filtered_data.get('accel_bias', [0, 0, 0])[2]
                    ],
                    "filtered_gyro": [
                        filtered_data.get('raw_gyro', [0, 0, 0])[0] - filtered_data.get('gyro_bias', [0, 0, 0])[0],
                        filtered_data.get('raw_gyro', [0, 0, 0])[1] - filtered_data.get('gyro_bias', [0, 0, 0])[1],
                        filtered_data.get('raw_gyro', [0, 0, 0])[2] - filtered_data.get('gyro_bias', [0, 0, 0])[2]
                    ],
                    "raw_accel": filtered_data.get('raw_accel', [0, 0, 0]),
                    "raw_gyro": filtered_data.get('raw_gyro', [0, 0, 0]),
                    "timestamp": filtered_data.get('timestamp', time.time())
                }
                
                # Convert to JSON and send
                json_data = json.dumps(unity_data)
                sock.sendto(json_data.encode('utf-8'), (UDP_IP_SEND, UDP_PORT_SEND))
                
            time.sleep(0.01)  # Send at ~100Hz
            
        except Exception as e:
            print(f"Error sending data: {e}")

def ekf_filter_thread():
    """Thread to process IMU data through your custom EKF"""
    print("EKF filter thread started...")
    
    # Initialize your custom EKF with proper dimensions
    try:
        # Your custom EKF expects dim_x and dim_u 
        # dim_u should be 9 for [ax, ay, az, wx, wy, wz, mx, my, mz] but we'll use 6 for accel+gyro only
        ekf = ExtendedKalmanFilter(dim_x=16, dim_u=6)  # 16 states, 6 measurements (accel + gyro)
        
        # Initialize state vector [qw, qx, qy, qz, px, py, pz, vx, vy, vz, gbx, gby, gbz, abx, aby, abz]
        ekf.x = np.zeros(16)
        ekf.x[0] = 1.0  # Initialize quaternion w component
        
        # Initialize matrices
        ekf.P = np.eye(16)
        ekf.P[0:4, 0:4] *= 0.01     # Quaternion uncertainty
        ekf.P[4:7, 4:7] *= 1.0      # Position uncertainty  
        ekf.P[7:10, 7:10] *= 0.1    # Velocity uncertainty
        ekf.P[10:13, 10:13] *= 0.01 # Gyro bias uncertainty
        ekf.P[13:16, 13:16] *= 0.01 # Accel bias uncertainty
        
        ekf.Q = np.eye(16) * 0.001
        ekf.Q[0:4, 0:4] *= 0.0001   # Low process noise for quaternion
        ekf.Q[10:16, 10:16] *= 0.00001  # Very low bias process noise
        
        ekf.R = np.eye(6)
        ekf.R[0:3, 0:3] *= 0.1      # Accelerometer noise
        ekf.R[3:6, 3:6] *= 0.05     # Gyroscope noise
        
        # Define the predict_x function that your EKF expects
        def predict_x_func(u):
            """Prediction function for state transition"""
            # This should update ekf.x using the state transition model
            # For now, use your existing f() function
            dt = 0.01  # Assume 100Hz for now, will be updated with real dt
            return ekf.f(ekf.x, dt, u)
        
        # Assign the predict_x function to your EKF
        ekf.predict_x = predict_x_func
        
        # Define measurement function Hx
        def Hx_func(x):
            """Measurement function - expected sensor readings given state"""
            # For IMU: return expected accelerometer and gyroscope readings
            # This is simplified - you may need a more sophisticated model
            h = np.zeros(6)
            
            # Expected accelerometer reading (gravity + linear acceleration)
            # For now, just return zero acceleration (hovering)
            h[0:3] = [0, 0, 9.81]  # Expected gravity in body frame
            
            # Expected gyroscope reading (angular velocity)
            h[3:6] = [0, 0, 0]  # Expected zero angular velocity
            
            return h
        
        # Define Jacobian of measurement function
        def HJacobian_func(x):
            """Jacobian of measurement function"""
            # For simplified model, return identity matrix scaled appropriately
            H = np.zeros((6, 16))
            
            # Accelerometer measurements depend on orientation (quaternion)
            H[0:3, 0:4] = 0.1  # Small influence from quaternion
            H[0:3, 13:16] = 1.0  # Direct influence from accel bias
            
            # Gyroscope measurements depend on angular velocity and bias
            H[3:6, 10:13] = 1.0  # Direct influence from gyro bias
            
            return H
        
        # Compute Jacobian of state transition function
        def compute_F_jacobian(x, dt):
            """Compute Jacobian of state transition function"""
            # For simplified model, start with identity matrix
            F = np.eye(16)
            
            # Add derivatives for position/velocity integration
            F[4:7, 7:10] = dt * np.eye(3)  # position depends on velocity
            
            return F
        
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
                        # Create control input vector (what your f() function expects)
                        u = np.concatenate([accel, gyro])  # [ax, ay, az, wx, wy, wz]
                        
                        # Update the dt in the predict_x function
                        def predict_x_with_dt(u_input):
                            return ekf.f(ekf.x, dt, u_input)
                        
                        ekf.predict_x = predict_x_with_dt
                        
                        # Update F matrix for this time step
                        ekf.F = compute_F_jacobian(ekf.x, dt)
                        
                        # Predict step
                        try:
                            ekf.predict(u)
                            print(f"Predicted state: {ekf.x}")
                        except Exception as predict_error:
                            print(f"Predict error: {predict_error}")
                            continue
                        
                        # Create measurement vector [ax, ay, az, gx, gy, gz]
                        z = u  # The measurements are the same as control inputs for IMU
                        
                        # Update step with required parameters
                        try:
                            ekf.update(z, HJacobian_func, Hx_func)
                            print(f"Updated state: {ekf.x}")
                        except Exception as update_error:
                            print(f"Update error: {update_error}")
                            continue
                        
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
    sender_thread = threading.Thread(target=udp_sender_thread, daemon=True)
    ekf_thread = threading.Thread(target=ekf_filter_thread, daemon=True)
    display_thread_obj = threading.Thread(target=display_thread, daemon=True)
    
    receiver_thread.start()
    sender_thread.start()
    ekf_thread.start()
    display_thread_obj.start()
    
    print("IMU EKF filtering started. Press Ctrl+C to exit.")
    print(f"Receiving raw IMU data on port {UDP_PORT_RECEIVE}")
    print(f"Sending filtered IMU data to port {UDP_PORT_SEND}")
    print("=" * 150)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")