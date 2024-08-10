import rosbag
import cv2
import numpy as np
import os
import tqdm
from scipy import signal

import matplotlib.pyplot as plt

np.random.seed(42)

bag_dir = 'bag/val_day1'
train_path = 'drone_pointnav_it/val'
train_ratio = 0.9

os.makedirs(train_path, exist_ok=True)
filename_list = os.listdir(bag_dir)
total_dur = 0

    
def get_relative_position(current_state, next_state, drone_head_at_x=True):
    '''
    current state = [x0,y0,z0,yaw0] in the global frame
    next_state state = [x1,y1,z1,yaw1] in the global frame
    '''
    x0, y0, z0, yaw0 = current_state[0], current_state[1], current_state[2], current_state[3]
    x1, y1, z1, yaw1 = next_state[0], next_state[1], next_state[2], next_state[3]

    xdot, ydot, zdot, yawdot = x1-x0, y1-y0, z1-z0, yaw1-yaw0

    # If the difference is greater than pi, adjust by subtracting 2*pi
    # If the difference is less than -pi, adjust by adding 2*pi
    if yawdot > np.pi:
        yawdot -= 2 * np.pi
    elif yawdot < -np.pi:
        yawdot += 2 * np.pi

    # Method 1: Differentiate (May only work if the sampling frequency is high and angular velocity is low)
    # xdot_p = xdot*np.cos(yaw0) + ydot*np.sin(yaw0) - x0*np.sin(yaw0)*yawdot + y0*np.cos(yaw0)*yawdot
    # ydot_p = -xdot*np.sin(yaw0) + ydot*np.cos(yaw0) -x0*np.cos(yaw0)*yawdot - y0*np.sin(yaw0)*yawdot  
    
    # Method 2: Transform x1 y1 to the current drone coordinate
    # Waypoint (~velocity) = (x1_p - x0_p, y1_p-y0_p)

    xdot_p = (xdot)*np.cos(yaw0) + (ydot)*np.sin(yaw0)
    ydot_p = -(xdot)*np.sin(yaw0) + (ydot)*np.cos(yaw0)

    # Sanity check
    # if np.sqrt((x1-x0)**2 + (y1-y0)**2) > 0.03:
    #     print(f"x0,y0, x1,y1, theta,sin,cos = {x0} {y0} {x1} {y1} {yaw0*(180/np.pi)} {np.sin(yaw0)} {np.cos(yaw0)}")
    #     print(f"xdot_p, ydot_p {xdot_p} {ydot_p}")
    #     print("----------------")

    if drone_head_at_x:
        '''
        # If the drone forward direction is initialized at x axis 
        # we may want to change x_p to represent right(+) and left(-)
        # and y_p to represent forward (+) and backward (-)
        '''
        ydot_pp = xdot_p
        xdot_pp = -ydot_p
        return xdot_pp, ydot_pp, zdot, yawdot
    else:
        '''
        # If the drone forward direction is initilizaed at y axis
        # We can use what we calculate from the linear tranformation
        '''
        return xdot_p, ydot_p, zdot, yawdot


def create_episode(path, state_array, obs_array, action_array, wandb_log):
    episode = []
    episode_length = len(obs_array)
    assert len(state_array) == episode_length
    for step in range(episode_length):
        terminal_step = episode_length - 1
        goal_array = state_array[terminal_step] - state_array[step]
        goal_array = goal_array[:2] # Goal is only (x, y) position
        is_terminal = False
        if step == terminal_step:
            action = np.zeros_like(action_array[step-1])
            is_terminal = True
        else:
            action = action_array[step]
        episode.append({
            'image': obs_array[step],
            'state': state_array[step],
            'action': action,
            'goal': goal_array,
            'is_terminal' : is_terminal
        })
    np.save(path, episode)
    if wandb_log:
        import wandb

        global_coord = []
        actions = []

        for step in tqdm.tqdm(range(0, episode_length-1)):
            global_coord.append(
                    episode[step]['state']
            )
            actions.append(
                episode[step]['action']
            )

        ep_name = os.path.basename(path)
        wandb.init(
            # set the wandb project where this run will be logged
            project="check_drone_manchester_val",
            name=f"{ep_name}"
        )
        img_strip = np.concatenate(np.array(obs_array[::3]), axis=1)
        ACTION_DIM_LABELS = ['dx_local', 'dy_local', 'dz', 'dyaw']
        GLOBAL_DIM_LABELS = ['x_global', 'y_global', 'z_global', 'yaw_global']
        figure_layout = [
            ['image'] * len(ACTION_DIM_LABELS),
            ACTION_DIM_LABELS,
            GLOBAL_DIM_LABELS,
        ]
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['figure.constrained_layout.use'] = True
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        global_coord = np.array(global_coord).squeeze()
        actions = np.array(actions).squeeze()
        for dim, label in enumerate(ACTION_DIM_LABELS):
            try:
                axs[label].plot(actions[:, dim])
                axs[label].set_title(label)
                axs[label].set_xlabel('Time in one episode')
            except:
                pass
        for dim, label in enumerate(GLOBAL_DIM_LABELS):
            try:
                axs[label].plot(global_coord[:, dim])
                axs[label].set_title(label)
                axs[label].set_xlabel('Time in one episode')
            except:
                pass

        axs['image'].imshow(img_strip)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        plt.legend()

        wandb.log({f"chart {ep_name}": plt})
        vids = np.array(obs_array) # (T, H, W, C)
        vids = vids.transpose(0, 3, 1, 2) # (T, C, H, W)
        wandb.log({f"video {ep_name}": wandb.Video(vids, fps=10)})
        wandb.finish()

for fn in tqdm.tqdm(filename_list):
    # Path to the ROS bag file
    bag_file = os.path.join(bag_dir, fn)

    # Topic name of the compressed image
    topic = '/data_out'
    x_arr = []
    y_arr = []
    z_arr = []
    yaw_arr = []
    image_arr = []
    dpose = []
    # Open the bag file
    down_sample_factor = 6 # Downsample from 30Hz to 5Hz
    step = 0
    x_prev = 0
    x_curr = 0
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            header = msg.header
            x_curr = msg.x_curr
            y_curr = msg.y_curr
            z_curr = msg.z_curr
            yaw_curr = msg.yaw_curr
            image_data = np.frombuffer(msg.image.data, np.uint8)
            cv_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR) # BGR image
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            if step % down_sample_factor == 0:
                image_arr.append(cv_image_rgb)
                total_dur+=1

            # if yaw_curr < 0:
            #     yaw_curr += 2*np.pi
            
            # New method: LPF + Downsamplinng
            x_arr.append(x_curr)
            y_arr.append(y_curr)    
            z_arr.append(z_curr)
            yaw_arr.append(yaw_curr)

            step+=1

        dx,dy,dz,dyaw = [], [], [], []
        state_array_temp = np.array([x_arr, y_arr, z_arr, yaw_arr]) # (4, T)
        state_array_temp = np.transpose(state_array_temp, (1,0)) # (T, 4)
        for t in range(len(state_array_temp)-1):
            dx_t, dy_t, dz_t, dyaw_t = get_relative_position(state_array_temp[t], state_array_temp[t+1])
            dx.append(dx_t)
            dy.append(dy_t)
            dz.append(dz_t)
            dyaw.append(dyaw_t)
        

        x_arr = signal.decimate(x_arr, down_sample_factor)
        y_arr = signal.decimate(y_arr, down_sample_factor)
        z_arr = signal.decimate(z_arr, down_sample_factor)
        yaw_arr = signal.decimate(yaw_arr, down_sample_factor)

        state_array = np.array([x_arr, y_arr, z_arr, yaw_arr]) # (4, T)
        state_array = np.transpose(state_array, (1,0)) # (T, 4)

        # Process action
        dx = signal.decimate(dx, down_sample_factor) # (T-1, 1)
        dy = signal.decimate(dy, down_sample_factor) # (T-1, 1)
        dz = signal.decimate(dz, down_sample_factor) # (T-1, 1)
        dyaw = signal.decimate(dyaw, down_sample_factor) # (T-1, 1)

        action_array = np.array([dx, dy, dz, dyaw]) # (4, T-1)
        action_array = np.transpose(action_array, (1,0)) # (T-1, 4)


        obs_array = np.array(image_arr)
        if len(state_array) != len(obs_array):
            print(len(state_array), len(obs_array))

        fn_base = fn.replace(".bag", "")
        path = os.path.join(train_path, f"{fn_base}.npy")

        create_episode(path, state_array, obs_array, action_array, wandb_log=True)

print(f"TOTAL FRAMES: {total_dur}")