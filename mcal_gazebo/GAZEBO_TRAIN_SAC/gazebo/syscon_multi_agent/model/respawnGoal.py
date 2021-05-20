
import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self, init_goal_x, init_goal_y, model_name):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath + "/goal_box/model.sdf"
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.goal_position = Pose()

        self.goal_position.position.x = init_goal_x
        self.goal_position.position.y = init_goal_y
        self.modelName = model_name

        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

        self.is_goal_model = False

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            #print(model.name[i])
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")

        self.is_goal_model = True

    def deleteModel(self):

        if self.is_goal_model == True:

            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox(self.modelName)
            self.is_goal_model = False
