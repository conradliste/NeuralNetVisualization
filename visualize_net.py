
import torch
import torch.nn as nn
import torch.nn.functional as F
from manim import *
from collections import OrderedDict
from extract_layers import extract_layers
import numpy as np
import time
import math

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
            if not isinstance(src_neuron, MathTex) and not isinstance(dest_neuron, MathTex):
                edge = create_edge(src_neuron, dest_neuron)
                array.append(edge)
    edges.add(*array)
    return edges 

# Creates an edge between two neurons
def create_edge(src, dest, show_arrow=False, buffer=0, edge_color=LIGHT_GREY, edge_width=0.3):
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
        self.in_color =  WHITE
        self.hid_color = WHITE
        self.out_color = BLUE
        self.neuron_stroke_width = self.neuron_radius / 5.0
        self.neuron_dist = self.neuron_radius 
        self.neuron_opacity = 1
        self.edge_color = LIGHT_GREY
        self.edge_stroke_width = 2
        self.edge_prop_time = 1
        self.arrow_tip_size = 0.1
        self.layer_dist = self.neuron_radius * 2  * 4
        self.border_buffer = self.neuron_radius
        

class LinearVisual(nnLayer):
    def __init__(self, size, layer_type="input", cap_neurons=True):
        super(LinearVisual, self).__init__()
        # Calculate max neurons that should be shown
        if cap_neurons:
            self.max_neurons_shown = self.calc_max_neurons()
        else:
            self.max_neurons_shown = 2 ** 30
        # Create the input and output layers
        self.layer = self.create_layer(size, layer_type)
        # Add the edges, input layer, and output layer
        self.add(self.layer)

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
        layer = VGroup()
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
                fill_color = self.get_stroke_color(layer_type),
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
            true_size = MathTex("{}".format(num_neurons))
            true_size.height = self.neuron_radius
            # Move the neurons so they do not cover the Latex
            VGroup(*neurons [:len(neurons) // 2]).next_to(dots, UP, self.neuron_radius)
            VGroup(*neurons [len(neurons) // 2:]).next_to(dots, DOWN, self.neuron_radius)

            layer.add(true_size)
            layer.add(dots)
            
        layer.add(*neurons)
        return layer

        
class NetVisual(nnLayer):
    def __init__(self, model, input_shape, device=torch.device("cuda:0")):
        super(NetVisual, self).__init__()
        self.layers_dict = extract_layers(model, input_shape, device=device)
        self.net_visual = VGroup()
        self.visuals = []
        # Create all the layers
        for index, layer_key in enumerate(self.layers_dict):
            layer = self.layers_dict[layer_key]
            input_size = layer["input_shape"]
            output_size = layer["output_shape"]
            layer_name = layer_key.split("-")[0]
            print(layer_key)

            # Case for linear layer
            if layer_name == "Linear":
                self.visuals.append(LinearVisual(input_size[0]))
                # Add the output layer if this is our final layer
                if index == len(self.layers_dict) - 1:
                    self.visuals.append(LinearVisual(output_size[0], layer_type="output"))
            
        self.visuals[0].to_edge(LEFT)
        # Connect all the layers
        if len(self.visuals) == 1:
            self.net_visual.add(self.visuals[0]) 
        else:
            for i in range(1, len(self.visuals)):
                prev = self.visuals[i-1]
                cur = self.visuals[i]
                # Move the cur layer to right of prev layer
                cur.next_to(prev.layer, RIGHT, self.layer_dist)
                if isinstance(prev, LinearVisual) and isinstance(cur, LinearVisual):
                    edges = connect_lin_layers(prev.layer, cur.layer)
                    self.net_visual.add(*[edges, prev, cur])
        
class nnVisual(Scene):
    
    def __init__(self, net_visual):
        super(nnVisual, self).__init__()
        self.net_visual = net_visual
        #self.net_visual.scale(0.2)
    
    def animate_network(self):
        pass
        
    def construct(self):
        self.add(self.net_visual)
        #self.net_visual.center()
        #self.net_visual.scale(0.1)
        #self.play(ScaleInPlace(self.net_visual, 0.2))
        self.wait(1)


        
        # Animate
        #self.play(ShowCreation(square))
        #self.play(Transform(square, circle))
        #self.play(FadeOut(square))
 

