from anytree import Node, RenderTree
import numpy as np
import random
import math
import copy



class Game:
    def __init__(self):
        self.action_space = [
            {'action_name': 'a'},
            {'action_name': 'b'},
            {'action_name': 'c'}
        ]
        self.stat = []
    
    def set_stat(self, stat):
        self.stat = copy.deepcopy(stat)
        
    def get_stat(self):
        return copy.deepcopy(self.stat)
    
    def next_stat(self, action_id):
        if self.is_illegal(action_id):
            raise ValueError('illegal action_id: %s' % action_id)
        if self.is_endgame():
            raise ValueError('This game is end!')
        self.stat.append(self.action_space[action_id]['action_name'])
        
    def is_illegal(self, action_id):
        return False
    
    def is_endgame(self):
        test = 3
        for action in self.action_space:
            if action['action_name'] in self.stat:
                test -= 1
        
        if test == 0:
            return True
        else:
            return False
        
    def get_value(self):
        if len(self.stat) <= 4:
            return 1
        else:
            return 0
        
        

class MCTS():
    """
    MCTS
    """
    def __init__(self, game, C=1, simulation_number=5):
        """
        init
        """
        self.game = game
        self.C = C
        
        self.simulation_number = simulation_number
        
        self.root = Node('root')
        self.init_node(self.root)
    
    def init_node(self, node):
        """
        init node
        """
        node.stat = self.game.get_stat()
        node.number = 0
        node.accum_value = 0        
        if self.game.is_endgame():
            node.is_endgame = True
        else:
            node.is_endgame = False

    def search(self, max_search):
        """
        perform mcts algorithm
        """
        for i in range(max_search):
            selected_node = self.selection()
            if selected_node:
                self.expansion(selected_node)
            else:
                print('reach all path')
                break
            
    def get_ucb(self, this_node, C):
        """
        get ucb
        """
        exploitation = this_node.accum_value / this_node.number
        exploration = (2*math.log(this_node.parent.number) / this_node.number)**0.5

        ucb = exploitation + C*exploration
        return ucb

    def select(self, node_candidates, C):
        """
        select max ucb node within candidate
        """
        max_ucb = -np.inf
        selected_node = None
        for node_candidate in node_candidates:
            ucb = self.get_ucb(node_candidate, C)

            if ucb > max_ucb:
                max_ucb = ucb
                selected_node = node_candidate
        return selected_node

    def selection(self):
        """
        selection procedure
        """
        stack_node_list = [self.root]
        node_candidates_dict = {
            self.root: [node_candidate for node_candidate in self.root.children]
        }        
        
        while stack_node_list:
            point_node = stack_node_list[-1]
            
            if point_node.is_endgame:
                del stack_node_list[-1]
                continue
            
            if not point_node.children:
                break
            
            node_candidates = node_candidates_dict[point_node]
            
            selected_node = self.select(node_candidates, self.C)
            if selected_node:
                node_candidates.remove(selected_node)
                
                stack_node_list.append(selected_node)
                node_candidates_dict.update({
                    selected_node: [node_candidate for node_candidate in selected_node.children]
                })
                
            else:
                del stack_node_list[-1]
        if stack_node_list:
            return stack_node_list[-1]
        else:
            return None
        
    def expansion(self, this_node):
        """
        expansion procedure
        """
        for idx, action in enumerate(self.game.action_space):
            
            stat = this_node.stat
            self.game.set_stat(stat)
            
            if self.game.is_illegal(idx):
                continue
            
            expand_node = Node('action%s' % idx, parent=this_node)
            expand_node.action_id = idx
            
            self.game.next_stat(idx)
            self.init_node(expand_node)
            
            self.simulation(expand_node)
    
    def take_action_id(self):
        """
        take an action with given policy
        can be replaced
        """
        return random.randint(0, len(self.game.action_space)-1)

    def simulation(self, this_node):
        """
        simulation procedure
        """
        estimate_value = 0.0

        stat = this_node.stat
        self.game.set_stat(stat)

        for step in range(self.simulation_number):
            if self.game.is_endgame():
                break

            action_id = self.take_action_id()
            if self.game.is_illegal(action_id):
                continue
            self.game.next_stat(action_id)

        if self.game.is_endgame():
            value = self.game.get_value()
        else:
            value = 0

        self.backpropagation(this_node, value)

    def backpropagation(self, this_node, value):
        """
        backpropagation procedure
        """
        this_node.number += 1
        this_node.accum_value += value
        while this_node.parent:
            this_node = this_node.parent

            this_node.number += 1
            this_node.accum_value += value
    
    def tree_represent(self):
        """
        print tree
        """
        for pre, fill, node in RenderTree(self.root):
            # print("%s%s%s" % (pre, node.name, node.number))        
            # print("%s%s" % (pre, node.number)) 
            print("%s%s:(%s,%s)" % (pre, node.name, node.accum_value, node.number)) 
    
    def get_solution(self):
        """
        get solution after perform mcts search
        """
        action_list = []
        point_node = self.root
        
        # tree search
        while point_node.children:
            node_candidates = [node_candidate for node_candidate in point_node.children]
            selected_node = self.select(node_candidates, 0)
            action_list.append(selected_node.action_id)
            point_node = selected_node
            
        # random search if game is not end
        self.game.set_stat(selected_node.stat)
        while True:
            if self.game.is_endgame():
                break
            while True:
                action_id = self.take_action_id()
                if not self.game.is_illegal(action_id):
                    break
                
            self.game.next_stat(action_id)
            action_list.append(action_id)

        return action_list
                
            
        
if __name__ == '__main__':    
    game = Game()
    mcts = MCTS(game)
    mcts.search(50)
    mcts.tree_represent()
    print(mcts.get_solution())
