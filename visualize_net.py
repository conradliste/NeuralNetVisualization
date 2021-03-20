
import torch
import torch.nn as nn
import torch.nn.functional as F
from manim import *
from collections import OrderedDict
from extract_layers import extract_layers
import numpy as np
import time
import copy
import math
from more_geometry import BGrid

# Configuration
config.max_files_cached = 10

# Connects two linear layers with edges
def connect_lin_layers(src_layer, dest_layer):
    edges = VGroup()
    array = []
    # Add edge between each neuron in the src and dest layer
    for src_neuron in src_layer:
        for dest_neuron in dest_layer:
            # Do not add Latex objects
            if not isinstance(src_neuron, Tex) and not isinstance(dest_neuron, Tex):
                edge = create_edge(src_neuron, dest_neuron, buffer=src_layer.neuron_radius)
                array.append(edge)
    edges.add(*array)
    return edges 

# Creates an edge between two neurons
def create_edge(src, dest, show_arrow=False, buffer=0, edge_color=LIGHT_GREY, edge_width=0.2):
    return Line(
        src.get_center(),
        dest.get_center(),
        buff=buffer,
        stroke_color= edge_color,
        stroke_width = edge_width
    )


class nnLayer(VGroup):
    def __init__(self):
        super(nnLayer, self).__init__()
        self.neuron_radius = 0.155
        self.in_color =  ORANGE
        self.hid_color = WHITE
        self.out_color = BLUE
        self.neuron_stroke_width = 1
        self.neuron_dist = self.neuron_radius 
        self.neuron_opacity = 1
        self.edge_color = LIGHT_GREY
        self.edge_stroke_width = 2
        self.edge_prop_time = 1
        self.arrow_tip_size = 0.1
        self.layer_dist = self.neuron_radius * 2  * 7
        self.border_buffer = self.neuron_radius



class ConvVisual(nnLayer):
    def __init__(self, input_shape, output_channels, kernel_size, stride, padding=0, plain=False):
        super(ConvVisual, self).__init__()
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.base_shift = 0.05
        self.base_scale = 0.1
        self.plain = plain
        self.create_layer()

    def create_layer(self):
        depth = self.input_shape[0]
        height = self.input_shape[1]
        width = self.input_shape[2]
        if self.plain:
            for i in range(0, depth):
                s = Square(
                    fill_color=BLUE,
                    side_length=height,
                    stroke_width=1,
                    stroke_color=WHITE,
                    fill_opacity=1
                )
                s.scale(self.base_scale)
                s.shift(np.array((-(depth - 1 - i) * self.base_shift, (depth - 1 - i) * self.base_shift, 0.0)))
                self.add(s)
        else:
            for i in range(0, depth):
                b = BGrid(height, width, fill_color=BLUE, stroke_width=0.8)
                b.scale(self.base_scale)
                # Shift it up and to the left based on whichever feature map we're currently creating
                b.shift(np.array((-(depth - 1 - i) * self.base_shift, (depth - 1 - i) * self.base_shift, 0.0)))
                self.add(b)
        

class LinearVisual(nnLayer):
    def __init__(self, size, layer_type="input", cap_neurons=True):
        super(LinearVisual, self).__init__()
        # Calculate max neurons that should be shown
        if cap_neurons:
            self.max_neurons_shown = self.calc_max_neurons()
        else:
            self.max_neurons_shown = 2 ** 30
        # Create the input and output layers
        self.create_layer(size, layer_type)

    # Helper to calculate the max number of neurons
    def calc_max_neurons(self):
        height = config.frame_height - 2 * self.border_buffer
        width = config.frame_width - 2 * self.border_buffer
        # DEBUGGING MATH WORK
        # height >= num_neurons * 2 * neuron_radius + (num_neurons-1) * spacing
        # height >= (2 * neuron_radius + spacing) * num_neurons - spacing
        # num_neurons <= (height + spacing)/(2 * neuron_radius + spacing)
        return math.floor((height + self.neuron_dist)/(2 * self.neuron_radius + self.neuron_dist))
    
    # Helper to get stroke colors based on layer type
    def get_stroke_color(self, layer_type):
        if layer_type == 'input':
            return self.in_color
        if layer_type == 'hidden':
            return self.hid_color
        else:
            return self.out_color
             
    # Creates a single layer of radial neurons
    def create_layer(self, size, layer_type='input'):
        num_neurons = size
        # Cap the shwon layer size at the max neurons shown
        if size > self.max_neurons_shown:
            size = self.max_neurons_shown
        mid = size // 2
        neurons = []
        # Initalize neurons
        for i in range(size):
            # Make circle
            neuron = Circle(
                radius=self.neuron_radius, 
                stroke_color= self.get_stroke_color(layer_type), 
                stroke_width= self.neuron_stroke_width,
                stroke_opacity=1,
                fill_color = BLACK,
                fill_opacity = self.neuron_opacity
            )
            offset = np.array(np.array((0.0, (mid - i) * self.neuron_dist, 0.0)))
            # Shift the neurons based on neuron radius and dist from middle neurone
            if i != mid:
                if i < mid:
                    offset[1] += 2 * self.neuron_radius * (mid - i)
                else:
                    offset[1] -= 2 * self.neuron_radius * (i - mid)
            # Add
            neuron.shift(offset)
            neurons.append(neuron)

        # Add an ellispe and text containing true size if we had to cap the number of neurons shown
        if num_neurons > size:
            dots = Tex(' . \n\n \\vspace{-.4em} \n\n .')
            true_size = Tex("${}$".format(num_neurons))
            true_size.height = self.neuron_radius
            # Move the neurons so they do not cover the Latex
            VGroup(*neurons [:len(neurons) // 2]).next_to(dots, UP, self.neuron_radius)
            VGroup(*neurons [len(neurons) // 2:]).next_to(dots, DOWN, self.neuron_radius)

            self.add(true_size)
            self.add(dots)
            
        self.add(*neurons)
    


        
class NetVisual(nnLayer):
    def __init__(self, model, input_shape, device=torch.device("cuda:0")):
        super(NetVisual, self).__init__()
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.layers_dict = extract_layers(model, input_shape, device=device)
        self.visuals = []
        self.edge_group = VGroup()
        # Create all the layers
        for index, layer_key in enumerate(self.layers_dict):
            layer = self.layers_dict[layer_key]
            input_size = layer["input_shape"]
            output_size = layer["output_shape"]
            layer_name = layer_key.split("-")[0]
            print(layer_key)
            # Set layer type
            if index == 0:
                layer_type = "input"
            else:
                layer_type = "hidden"
            # Case for linear layer
            if layer_name == "Linear":
                self.visuals.append(LinearVisual(input_size[0], layer_type=layer_type))
                # Add an additional output layer if this is our final layer
                if index == len(self.layers_dict) - 1:
                    self.visuals.append(LinearVisual(output_size[0], layer_type="output"))
  
        temp = []
        # Create copies of each neuron and add to the vgroup
        for layer in self.visuals:
            for neuron in layer:
                if not isinstance(neuron, Tex):
                    temp.append(neuron)
        self.neurons = VGroup(*temp)
        
        temp = []
        # Connect all the layers
        self.visuals[0].to_edge(LEFT)
        if len(self.visuals) == 1:
            self.add(self.visuals[0]) 
        else:
            for i in range(1, len(self.visuals)):
                prev = self.visuals[i-1]
                cur = self.visuals[i]
                # Move the cur layer to right of prev layer
                cur.next_to(prev, RIGHT, self.layer_dist)
                if isinstance(prev, LinearVisual) and isinstance(cur, LinearVisual):
                    edges = connect_lin_layers(prev, cur)
                    temp.append(edges)
                    self.add(*[edges, prev, cur])
        self.edge_group.add(*temp)
    
    def update_layers_dict(self, new_inputs):
        self.layers_dict = extract_layers(self.model, self.input_shape, inputs=new_inputs, device=self.device)
        
class nnVisual(MovingCameraScene):
    
    def __init__(self, net_visual):
        super(nnVisual, self).__init__()
        self.net_visual = net_visual
        self.flash_threshold = 0
    
    # Visualizes the forward propagtation
    def forward_visual(self):
        anims = []
        # Visualize the first input layer
        keys = list(self.net_visual.layers_dict.keys())
        layer = self.net_visual.layers_dict[keys[0]]
        self.flash_layer(self.net_visual.visuals[0], layer["input"][0][0], self.net_visual.in_color)
        # Loop through each layer
        for index, layer_key in enumerate(self.net_visual.layers_dict):
            layer = self.net_visual.layers_dict[layer_key]
            # Grab weights and outputs
            weights = layer["weights"]
            outputs = layer["output"]
            layer_name = layer_key.split("-")[0]
            # Copy layer and edges for animation
            original_layer = self.net_visual.visuals[index+1]
            layer_copy = copy.deepcopy(original_layer)
            original_edges = self.net_visual.edge_group[index]
            edges_copy = copy.deepcopy(original_edges)
            # Flash the layer
            print("Num Edges: ", len(self.net_visual.edge_group[index]))
            print("Num Weights: ", len(weights[0]))
            # Forward animation for linear layer
            if layer_name == "Linear":
                weights = torch.flatten(weights)
                self.flash_layer(original_edges, weights, self.net_visual.edge_color)
                self.flash_layer(original_layer, outputs[0], self.net_visual.hid_color)
                self.play(Transform(edges_copy, original_edges))
                self.play(Transform(layer_copy, original_layer))

        # Unflash all layers
        neuron_copy = copy.deepcopy(self.net_visual.neurons)
        edges_copy = copy.deepcopy(self.net_visual.edge_group)
        self.unflash_layers()
        self.play(Transform(neuron_copy, self.net_visual.neurons))
        self.play(Transform(edges_copy, self.net_visual.edge_group))
        return anims
    
    # Unflashes layers
    def unflash_layers(self):
        for neuron in self.net_visual.neurons:
            neuron.set_fill(opacity=1, color=BLACK)
        for edge in self.net_visual.edge_group:
            edge.set_fill(opacity=1, color=self.net_visual.edge_color)
        return 

    # Flashes neurons or edges if they pass a threshold
    def flash_layer(self, layer, values, color, threshold=0):
        for index, mobject in enumerate(layer):
            value = abs(values[index].item())
            if value >= threshold and not isinstance(mobject, Tex):
                mobject.set_fill(color=color, opacity=value)
            
    # Visualizes backprop
    def backward_visual(self):
        pass

    # Visualizes dropout
    def dropout_visual(self, layer):
        pass
    
    def construct(self):
        self.camera.frame.move_to(self.net_visual.get_center())
        self.camera.frame.scale(1.1) 
        self.add(self.net_visual)
        
        # Animate
        self.forward_visual()
        #self.play(ShowCreation(square))
        #self.play(Transform(square, circle))
        #self.play(FadeOut(square))
 

