# -*- coding: utf-8 -*-
"""
    File Name : default
    Description : default
    Author : Yingli Gong
    Time : Mon Sep 27 10:30:12 2021
"""
def net2ho_edges(rfileName, wfileName):
    threshold = 1
    wfile = open(wfileName, 'w')
    with open(rfileName) as file:
        line = file.readline()
        while line:
            line = line.split(',')
            if int(line[2]) >= threshold:
                wfile.write(line[0] + ' ' + line[1] + '\n')
            line = file.readline()
    wfile.close()

def trace2edges(rfileName, wfileName):
    edges = []
    edges_num = {}
    count = 0
    wfile = open(wfileName, 'w')
    with open(rfileName) as file:
        line = file.readline()
#        print(line)
        while line: 
            count = count + 1          
            if count % 1000 == 0:
                print(count)
            line = line.split()
            length = len(line)
            for i in range(1, length-1):
                edge = line[i] + ' ' + line[i+1]
                if edge not in edges:
                    if edge not in edges_num.keys():
                        edges_num[edge] = 1
                    else:
                        edges_num[edge] = edges_num[edge] + 1
                    if edges_num[edge] > 5:
                        edges.append(edge)
                        wfile.write(edge + '\n')
            line = file.readline()
    wfile.close()

def click2trace(rfileName, wfileName):
    trace = []
    wfile = open(wfileName, 'w')
    count = 1
    with open(rfileName) as file:
        line = file.readline()
        while line:
            sline = line.split()
            if len(sline) < 3:
                line = file.readline()
                continue
            trace.append(line.strip())
            wfile.write(str(count) + ' ' + line)
            line = file.readline()
            count =  count + 1
    wfile.close()
    print(len(trace))
        
if __name__ == '__main__':
    #files:
#    './applications/traces-simulated-mesh-v100000-t100-mo4.csv'
#    './data/click-stream/kosarak.dat'
#    './data/network-traces-simulated.csv'
    
    raw_data = '../tmpdata/click-stream/kosarak.dat'
    traces_file = '../tmpdata/click-stream/traces.txt'
    edges_file = '../tmpdata/click-stream/edges.txt'
    
    network_file = '../tmpdata/click-stream/network.csv'
    rules_file = '../tmpdata/click-stream/rules.csv'
    ho_edges_file = '../tmpdata/click-stream/ho-edges.txt'
    
#    traces_file = './tmpdata/trace-10000/traces.txt'
#    edges_file = './tmpdata/trace-10000/edges.txt'
#    
#    network_file = './tmpdata/trace-10000/network.csv'
#    rules_file = './tmpdata/trace-10000/rules.csv'
#    ho_edges_file = './tmpdata/trace-10000/ho-edges.txt'
#    
#     print('click2trace')
#     click2trace(raw_data, traces_file)
    print('trace2edges')
    trace2edges(traces_file, edges_file)
    #
    # net2ho_edges(network_file, ho_edges_file)