'''
We define the method of comunication of agents here.
'''

class message_pool:
    def __init__(self):
        self.msg = {}
        self.state = {"merge" : 0, "merge_pos" : 0, "merge_t" : 0}

    def get_all_msg(self):
        return self.msg

    def join(self, veh_id, message):
        self.msg[veh_id] = message

