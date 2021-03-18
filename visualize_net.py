
import torch
import torch.nn as nn
import torch.nn.functional as F
from manim import *
from collections import OrderedDict
from extract_layers import extract_layers
import numpy as np

# Configuration

# Connects two linear layers with edges
def connect_lin_layers(src_layer, dest_layer):
    edges = VGroup()
    i = 0
    array = []
    for src_neuron in src_layer:
        for dest_neuron in dest_layer:
            edge = create_edge(src_neuron, dest_neuron)
            array.append(edge)
        i += 1
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
        self.neuron_radius = 0.5
        self.in_color =  WHITE
        self.hid_color = WHITE
        self.out_color = BLUE
        self.neuron_stroke_width = 2
        self.edge_color = LIGHT_GREY
        self.edge_stroke_width = 2
        self.edge_prop_time = 1
        self.arrow_tip_size = 0.1
        self.neuron_dist = 10
        self.neuron_opacity = 1
        

class LinearVisual(nnLayer):
    def __init__(self, input_size, output_size, in_type="input", out_type="output"):
        super(LinearVisual, self).__init__()
        self.in_layer = self.create_layer(input_size, in_type)
        self.out_layer = self.create_layer(output_size, out_type)
        self.out_layer.shift(np.array((15.0, 0.0, 0.0)))
        self.edges = connect_lin_layers(self.in_layer, self.out_layer)
        self.add(self.edges)
        self.add(self.in_layer)
        self.add(self.out_layer)
        
      
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
        mid = size // 2
        # Initalize neurons
        for i in range(size):
            # Make circle
            neuron = Circle(
                radius=self.neuron_radius, 
                stroke_color= self.get_stroke_color(layer_type), 
                stroke_width= self.neuron_stroke_width,
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
            # add
            neuron.shift(offset)
            layer.add(neuron)
        return layer

        
class NetVisual(nnLayer):
    def __init__(self, model, input_shape, device=torch.device("cuda:0")):
        super(NetVisual, self).__init__()
        self.layers_dict = extract_layers(model, input_shape, device=device)
        self.net_visual = VGroup()
        self.visuals = []
        print(len(self.layers_dict))
        for layer_key in self.layers_dict:
            layer = self.layers_dict[layer_key]
            input_size = layer["input_shape"]
            output_size = layer["output_shape"]
            layer_name = layer_key.split("-")[0]
            print(1)
            
            if layer_name == "Linear":
                self.visuals.append(LinearVisual(input_size[0], output_size[0]))
        
        if len(self.visuals) == 1:
            print("hello")
            self.net_visual.add(self.visuals[0]) 
        else:
            for i in range(1, len(self.visuals)):
                prev = self.visuals[i-1]
                cur = self.visuals[i]

                if isinstance(prev, LinearVisual) and isinstance(cur, LinearVisual):
                    edges = connect_lin_layers(prev.in_layer, cur.out_layer)
                    self.net_visual.add(*[edges, prev, cur])
        
class nnVisual(Scene):
    
    def __init__(self, net_visual):
        super(nnVisual, self).__init__()
        self.net_visual = net_visual
        self.net_visual.scale(0.2)
        
    def print_layers(self):
        pass
        
    def initialize_shapes(self):
        pass
    def animate_network(self):
        pass
        
    def construct(self):
        self.add(self.net_visual)
        self.wait(1)


        
        # Animate
        #self.play(ShowCreation(square))
        #self.play(Transform(square, circle))
        #self.play(FadeOut(square))
 

