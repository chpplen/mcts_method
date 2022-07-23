from anytree import Node, RenderTree
import numpy as np
from mcts.mcts import MCTS
import random




class MCTSSelfPlay(MCTS):
    def __init__(self, game, actor_critic_function, C=1):
        self.actor_critic_function = actor_critic_function
        
        self.game = game
        self.C = C
        
        self.root = Node('root')
        self.root.action_name = 'root'
        self.init_node(self.root)

    def init_node(self, node):
        """
        init node
        """
        node.stat = self.game.get_stat()
        node.stat_transform = self.game.get_stat_transform()
        node.number = 0
        node.accum_value = 0
        
        x_schedule, x_process, x_left_job = self.game.get_stat_transform()
        
        y_value = self.actor_critic_function.predict([
            np.array([x_schedule]), 
            np.array([x_process]), 
            np.array([x_left_job])
        ])
        

        
        
        value = y_value[0][0]
        action_distribution = y_value[0][1:]
        
        node.value = value
        node.action_distribution = action_distribution
        
        if self.game.is_endgame():
            node.is_endgame = True
        else:
            node.is_endgame = False
        
    def get_ucb(self, this_node):
        """
        get ucb
        """
        exploitation = this_node.accum_value / this_node.number if this_node.number > 0 else 0
        exploration = (this_node.parent.number)**0.5 / (1 + this_node.number)
        
        action_distribution = this_node.parent.action_distribution
        action_id = this_node.name
        
        action_probability = action_distribution[action_id]

        ucb = exploitation + self.C*action_probability*exploration
        return ucb
    
    def select(self, node_candidates):
        """
        select max ucb node within candidate
        """
        max_ucb = -np.inf
        selected_node = None
        for node_candidate in node_candidates:
            ucb = self.get_ucb(node_candidate)

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
#             print(stack_node_list)
            point_node = stack_node_list[-1]
            
            if point_node.is_endgame:
                del stack_node_list[-1]
                continue
            
            if not point_node.children:
                node_candidates = []
                for idx, action in enumerate(self.game.action_space):
                    stat = point_node.stat
                    self.game.set_stat(stat)

                    if self.game.is_illegal(idx):
                        continue
                    new_node = Node(idx, parent=point_node)
                    new_node.action_name = action['action_name']

                    self.game.next_stat(idx)
                    self.init_node(new_node)
                    
                    node_candidates.append(new_node)
                    selected_node = self.select(node_candidates)
                    stack_node_list.append(selected_node)
                break
            
            node_candidates = node_candidates_dict[point_node]
            
            selected_node = self.select(node_candidates)

            if selected_node:
                node_candidates.remove(selected_node)
                
                stack_node_list.append(selected_node)
                
                if selected_node.number == 0:
                    break
                
                node_candidates_dict.update({selected_node: [node_candidate for node_candidate in selected_node.children]})
                
            else:
                del stack_node_list[-1]
        if stack_node_list:
            return stack_node_list[-1]
        else:
            return None
        
    def take_action_id(self, node):
        action_distribution = node.action_distribution
        action_id = np.random.choice(np.arange(0, action_distribution.shape[0]), p=action_distribution)
        return action_id
    
    
    def expansion(self, this_node):
        """
        expansion procedure
        """
        for idx, action in enumerate(self.game.action_space):
            
            stat = this_node.stat
            self.game.set_stat(stat)
            
            if self.game.is_illegal(idx):
                continue
            
            expand_node = Node(action['action_name'], parent=this_node)
            expand_node.action_id = idx
            
            self.game.next_stat(idx)
            self.init_node(expand_node)
            
            self.backpropagation(expand_node, expand_node.value)
    
    def search(self, max_search):
        """
        perform mcts algorithm
        """
        for i in range(max_search):
            selected_node = self.selection()
#             print(selected_node)
            if selected_node:
                self.backpropagation(selected_node, selected_node.value)
            else:
                print('reach all path')
                break

    def get_samples(self, tou=1):
        samples_list = []
        point_node = self.root
        print("%s:(%s,%s)" % (
            point_node.name,
            point_node.accum_value,
            point_node.number
        )) 

        while True:
            is_break_flag = True
            action_id_list = np.arange(len(self.game.action_space))
            pi_tou = [0.0]*len(self.game.action_space)
            for child in point_node.children:
                number = child.number
                action_id = child.name
                
                pi_tou[action_id] = number**(1.0/tou)
                
                if number > 0:
                    is_break_flag = False
                    
            if is_break_flag:
                break
                
            pi_tou = np.array(pi_tou)/np.sum(pi_tou)
            
            action_id = np.random.choice(action_id_list, p=pi_tou)
            
            
            for child in point_node.children:
                if child.name == action_id:
                    selected_node = child
                    break

            samples_list.append([
                point_node.stat_transform,
                pi_tou
            ])
            point_node = selected_node

            
            print("%s:(%s,%s)%s" % (
                selected_node.name,
                selected_node.accum_value,
                selected_node.number,
                selected_node.action_name
            )) 
            
        self.game.set_stat(selected_node.stat)
        
        while True:
            
            if self.game.is_endgame():
                break
            while True:
                action_id = random.randint(0, len(self.game.action_space)-1)
                if not self.game.is_illegal(action_id):
                    break
                
            self.game.next_stat(action_id)
            
            samples_list.append([
                self.game.get_stat_transform(),
                np.array([1.0/len(self.game.action_space)]*len(self.game.action_space))
            ])
        
        
        final_value = self.game.get_value()

        for sample in samples_list:
            sample.append(final_value)
               
        print('is endgame:', self.game.is_endgame())
        # print(samples_list[-5])
            
        return self.game.is_endgame(), samples_list
                
                
                
if __name__ == '__main__':
#     actor_critic_function = AC()

#     mcts = MCTSSelfPlay(scheduling, actor_critic_function, C=0.1)
#     mcts.search(1)
#     mcts.tree_represent()
#     mcts.get_samples()
    pass

    
    
    
    