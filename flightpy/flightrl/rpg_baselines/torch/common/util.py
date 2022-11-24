from tkinter.tix import Tree
import numpy as np
import pandas as pd
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
import statistics
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.patches as patches
import matplotlib.collections as mc

columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "Tilt",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "ax",
    "ay",
    "az",
    "mot1",
    "mot2",
    "mot3",
    "mot4",
    "thrust1",
    "thrust2",
    "thrust3",
    "thrust4",
    "act1",
    "act2",
    "act3",
    "act4",
]


def traj_rollout(env, policy, max_ep_length = 1000):
    traj_df = pd.DataFrame(columns=columns)
    # max_ep_length = 1000
    obs = env.reset(random=False)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    for _ in range(max_ep_length):
        act, _ = policy.predict(obs, deterministic=True)
        act = np.array(act, dtype=np.float64)
        #
        obs, rew, done, info = env.step(act)

        episode_id[done] += 1

        state = env.getQuadState()
        action = env.getQuadAct()

        # reshape vector
        done = done[:, np.newaxis]
        rew = rew[:, np.newaxis]

        # stack all the data
        data = np.hstack((episode_id, done, rew, state, action))
        data_frame = pd.DataFrame(data=data, columns=columns)

        # append trajectory
        traj_df = pd.concat([traj_df, data_frame], axis=0, ignore_index=True)
    return traj_df


def plot3d_traj(ax3d, pos, vel):
    sc = ax3d.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=np.linalg.norm(vel, axis=1),
        cmap="jet",
        s=1,
        alpha=0.5,
    )
    ax3d.view_init(elev=40, azim=50)
    #
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    # ax3d.set_zticks([])

    #
    # ax3d.get_proj = lambda: np.dot(
    # Axes3D.get_proj(ax3d), np.diag([1.0, 1.0, 1.0, 1.0]))
    # zmin, zmax = ax3d.get_zlim()
    # xmin, xmax = ax3d.get_xlim()
    # ymin, ymax = ax3d.get_ylim()
    # x_f = 1
    # y_f = (ymax - ymin) / (xmax - xmin)
    # z_f = (zmax - zmin) / (xmax - xmin)
    # ax3d.set_box_aspect((x_f, y_f * 2, z_f * 2))

def set_collision_point(hydrus_theta: list, hydrus_l: float) -> tuple:
    theta = [np.deg2rad(i) for i in hydrus_theta] # from deg to rad
    bR2 = np.array([0.0,0.0])
    bC2 = np.array([-hydrus_l/2,0.0])
    bC2ToR1 = np.array([-hydrus_l/2*np.cos(theta[0]),hydrus_l/2*np.sin(theta[0])])
    bR1 = bC2+bC2ToR1
    bC1 = bC2+bC2ToR1*2
    bC3 = np.array([hydrus_l/2,0.0])
    bC3ToR3 = np.array([hydrus_l/2*np.cos(theta[1]),hydrus_l/2*np.sin(theta[1])])
    bR3 = bC3+bC3ToR3
    bC4 = bC3+bC3ToR3*2
    bC4ToR4 = np.array([hydrus_l/2*np.cos(theta[1]+theta[2]),hydrus_l/2*np.sin(theta[1]+theta[2])])
    bR4 = bC4+bC4ToR4
    bC5 = bC4+bC4ToR4*2
    CG = (bR1+bR2+bR3+bR4)/4

    print("CG: ",CG)

    C_list = [bC1-CG,bC2-CG,bC3-CG,bC4-CG,bC5-CG]
    R_list = [bR1-CG,bR2-CG,bR3-CG,bR4-CG]

    C_list = [pos.reshape(2, 1) for pos in C_list]
    R_list = [pos.reshape(2, 1) for pos in R_list]

    return C_list,R_list


def test_policy(env, model, render=False):
    max_ep_length = env.max_episode_steps
    num_rollouts = 100
    frame_id = 0
    final_x_list = []
    ave_vel_list = []
    act_diff_sum = np.zeros(3)
    # print(act_diff_sum.shape)
    act = np.zeros(3)
    past_act = np.zeros(3)
    step_num = 0
    tilt = 0

    # parameters to show hydrus when they're done
    hydrus_theta = [0,60,0]
    hydrus_l = 0.6
    hydrus_r = 0.2

    C_list, R_list = set_collision_point(hydrus_theta,hydrus_l)

    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        final_x = 0
        final_t = 0
        lstm_states = None

        figure, axes = plt.subplots()

        if render:
            x_list=[]
            y_list=[]
            vel_list=[]
            yaw_list=[]
            path = os.environ["FLIGHTMARE_PATH"]+"/flightpy/configs/vision/HYDRUS_tree_2/environment_"+"490"+"/"
            
            with open(path+'static_obstacles.csv') as f:
                reader = csv.reader(f)
                for row in reader:
                    x = float(row[1])
                    y = float(row[2])
                    r = float(row[8]) #+body_size
                    draw_circle = plt.Circle((x, y), r)
                    axes.add_artist(draw_circle)
            plt.ylim([-5.0, 5.0])

        while not (done or (ep_len >= max_ep_length)):
            # print(obs)
            past_act = act
            act, lstm_states = model.predict(obs, state = lstm_states, deterministic=True)
            # https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html#sb3_contrib.ppo_recurrent.RecurrentPPO
            act = act.reshape(3)
            # print(act.shape)
            # print(past_act.shape)
            # print(act_diff_sum.shape)
            act_diff_sum += np.power(act - past_act , 2)
            # print(act)
            obs, rew, done, info = env.step(act)
            step_num += 1
            #
            env.render(ep_len)

            tilt += env.getQuadState()[0][8]


            # ======Gray Image=========
            # gray_img = np.reshape(
            #     env.getImage()[0], (env.img_height, env.img_width))
            # cv2.imshow("gray_img", gray_img)
            # cv2.waitKey(100)


            # ======RGB Image=========
            # img =env.getImage(rgb=True) 
            # rgb_img = np.reshape(
            #    img[0], (env.img_height, env.img_width, 3))
            # cv2.imshow("rgb_img", rgb_img)
            # os.makedirs("./images", exist_ok=True)
            # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img)
            # cv2.waitKey(100)


            # # ======Depth Image=========
            # depth_img = np.reshape(env.getDepthImage()[
            #                        0], (env.img_height, env.img_width))
            # os.makedirs("./depth", exist_ok=True)
            # cv2.imwrite("./depth/img_{0:05d}.png".format(frame_id), depth_img.astype(np.uint16))
            # cv2.imshow("depth", depth_img)
            # cv2.waitKey(100)

            # print(roll)
            # print(tilt)
            # print(yaw)


            #
            if render:
                x_list.append(env.getQuadState()[0][1])
                y_list.append(env.getQuadState()[0][2])
                vel_list.append(np.sqrt(env.getQuadState()[0][9]**2+env.getQuadState()[0][10]**2))
                quaternion = [env.getQuadState()[0][4],env.getQuadState()[0][5],
                env.getQuadState()[0][6],env.getQuadState()[0][7]]
                r = R.from_quat([quaternion[0], quaternion[1], 
                quaternion[2], quaternion[3]])
                euler = r.as_euler('zyx')
                # print("euler: ",euler)
                yaw_list.append(euler[0])
                

            if done:
                r = R.from_quat([quaternion[0], quaternion[1], 
                quaternion[2], quaternion[3]])
                euler = r.as_euler('zyx')
                print("euler: ",euler)
                final_yaw = euler[0]
                r = np.array([
                    [np.cos(final_yaw), -np.sin(final_yaw)],
                    [np.sin(final_yaw), np.cos(final_yaw)]
                ])


                if final_x == 0:
                    # reset the test, becuase the drone collide with object in the initial state
                    obs, done, ep_len = env.reset(), False, 0
                    print("reset the test, becuase the drone collide with object in the initial state")
                    print(env.getQuadState()[0][0])
                    continue
                final_x_list.append(final_x)
                # print("n_roll: ",n_roll)
                print("final x: {}".format(final_x))
                ave_vel_list.append(final_x/final_t)
                print("ave vel: {}".format(final_x/final_t))

                final_pos = np.array([
                    [final_x],
                    [final_y]
                    ])

                if render:
                    plt.xlim(final_x-5,final_x+5)
                    # plt.ylim(final_y-5,final_y+5)
                    plt.scatter(x_list,y_list,
                        c=vel_list,
                        cmap=cm.jet,
                        marker='.',lw=0)
                    # ax=plt.colorbar()
                    # ax.set_label('vel [m/s]')

                    edge_list = [final_pos + r @ pos for pos in C_list]
                    lines = [[(edge_list[i][0,0],edge_list[i][1,0]),(edge_list[i+1][0,0],edge_list[i+1][1,0])]
                    for i in range(len(edge_list)-1)]
                    lc = mc.LineCollection(lines, colors = "g",linewidths=2)
                    axes.add_collection(lc)

                    rotor_list = [final_pos + r @ pos for pos in R_list]
                    # axes = plt.axes()
                    for rotor in rotor_list:
                        c = patches.Circle(xy=(rotor[0,0], rotor[1,0]), radius=hydrus_r, fc="none", ec='r')
                        axes.add_patch(c)

                    plt.show()

                    plt.plot(yaw_list)
                    plt.show()
                    # https://villageofsound.hatenadiary.jp/entry/2015/09/13/010352
            else:
                final_t = env.getQuadState()[0][0]
                final_x = env.getQuadState()[0][1]
                final_y = env.getQuadState()[0][2]
                quaternion = [env.getQuadState()[0][4],env.getQuadState()[0][5],
                env.getQuadState()[0][6],env.getQuadState()[0][7]]

            ep_len += 1
            frame_id += 1


    print("average final x: {}".format(sum(final_x_list)/num_rollouts))
    print("standard deviation x: {}".format(statistics.pstdev(final_x_list)))
    plt.hist(final_x_list)
    plt.show()
    print("average vel: {}".format(sum(ave_vel_list)/num_rollouts))
    print("standard deviation vel: {}".format(statistics.pstdev(ave_vel_list)))

    print("action difference: {}".format(np.round(np.sqrt(act_diff_sum/step_num),decimals = 2)))
    print("tilt: {}".format(np.round(tilt/step_num,decimals = 2)))
    if render:
        env.disconnectUnity()
