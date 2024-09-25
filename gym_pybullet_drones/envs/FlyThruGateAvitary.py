import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class FlyThruGateAvitary(BaseRLAviary):
    """Single agent RL problem: fly through a gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int=240,
                 ctrl_freq: int=30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = 12
        super().__init__(drone_model=drone_model,
                         num_drones = 1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.collide = False


    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        super()._addObstacles()
        g1 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0, 1, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g2 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0.5, 3, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g3 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [-0.5, 5, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g4 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [-0.5, 6, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )

        self.GATE_IDs = np.array([g1, g2, g3, g4])

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        rpy = state[7:10]
        ang_vel=state[13:16]
        for i, key in enumerate(self.racing_setup.keys()):
            if self.passing_flag[i]:
                continue
            cur_dist = np.linalg.norm(np.array(self.racing_setup[key][0]) - state[:3])
            prev_dist = np.linalg.norm(np.array(self.racing_setup[key][0]) - self.prev_pos)
            break
        self.prev_pos = state[:3]
        on_way_reward = prev_dist - cur_dist - 0.001 * np.linalg.norm(ang_vel)
        passing = False
        for i, key in enumerate(self.racing_setup.keys()):
            if self.passing_flag[i]:
                continue
            if self.racing_setup[key][1][0] < state[0] < self.racing_setup[key][2][0] and \
                self.racing_setup[key][1][1] < state[1] < self.racing_setup[key][1][1] + 2*self.offset and \
                self.offset < state[2] < self.offset + self.h:
                passing = True
                self.passing_flag[i] = True
                break
        self.collide = False
        for i in range(4):
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0],
                       bodyB=self.GATE_IDs[i],
                       physicsClientId=self.CLIENT
                       )
            if len(contact_points) != 0:
                self.collide = True
                break
        if passing:
            print("passing")
            print(state[:3])
            print(self.passing_flag)
            reward = 10
        elif self.collide:
            print("collide")
            print(state[:3])
            reward = -10
        else:
            reward = on_way_reward

        return reward

    ################################################################################
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.w/2 + 0.5 + 2*self.offset or state[1] > 6 or state[1] < -2 * self.offset \
        or state[3] > self.h + 2 * self.offset# Truncate when the drone is too far away
        ):
            if self.passing_flag[0]:
                print('trucated')
                print(state[:3])
            return True
        # if abs(state[7]) > .78 or abs(state[8]) > .78: # Truncate when the drone is too tilted
        #     if self.passing_flag[0]:
        #         print('trucated by tilted')
        #         print(state[:3])
        #     return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if  self.collide==True: #self.passing_flag[2] == True or
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
