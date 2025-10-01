import numpy as np
import matplotlib.pyplot as plt
from robot import Robot

arlo = Robot()

id_list = []
landmarks = []
radius = 180

for i in range (len(ids)):
    landmark_id = ids[i][0]
    if landmark_id in id_list:
        continue
    
    map_x = tvecs[i][0][0]
    map_y = tvecs[i][0][2]
    
    
    landmarks.append((landmark_id, map_x, map_y))
    id_list.append(landmark_id)
    
def in_collision (path, landmarks, robot_radius=150):
    for p in path:
        x, y = p
        
        for landmark in landmarks:
            id, map_x, map_y, radius = landmark
            distance = np.sqrt((x - map_x)**2 + (y - map_y)**2) #Euclidean distance
            if distance <= radius + robot_radius:
                return True
    return False
    

def follow_rrt_path(arlo,path):
    for i in range(1, len(path)):
        start = path[i-1]
        target = path [i]
        
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        arlo.rotate_robot(angle_deg)
        
        distance_m = np.sqrt(dx**2 + dy**2) / 1000 # to meter
        
        distance_travelled = 0
        
        while distance_travelled < distance_m:
            arlo.drive_forward_meter(distance_m, leftspeed= 64,  rightspeed = 67)
            distance_travelled += 0.2
            
            current_position = arlo.get_position()
            if in_collision (current_position, landmarks):
                print("Collision detected! Stopping.")
                arlo.stop()
                break
            

        

    
        
fig, ax = plt.subplots() 
obstacle_x = []
obstacle_y = []
for landmark in landmarks:
    id, map_x, map_y, radius = landmark
    circle = plt.Circle((map_x, map_y), radius, color='r', fill=False, linestyle='--', alpha=0.5)
    
    