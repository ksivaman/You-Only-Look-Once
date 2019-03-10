from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as Variable
import numpy as 

def parse_cfg(cfgfile):
   file = open(cfgfile, 'r')
   lines = [x for x in lines if len(x) > 0]
   lines = [x for x in lines if x[0] != '#']
   lines = [x.rstrip().lstrip() for x in lines]

   block = {}
   blocks = []

   for line in lines:
       if line[0] == "[":
           if len(block) != 0:
               blocks.append(block)
               block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters=int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), act)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            up = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{0}".format(index), up)

        elif(x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except
                end = 0