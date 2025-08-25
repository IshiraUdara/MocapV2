import socket
import json

UDP_IP = "127.0.0.1"  # Match Unity's IP
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for IMU data on {UDP_IP}:{UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024)  # buffer size
    imu_data = json.loads(data.decode('utf-8'))
    accel = imu_data['accel']
    gyro = imu_data['gyro']
    orientation = imu_data['orientation']

    print(f"Accel: {accel}, Gyro: {gyro}, Orientation: {orientation}")
