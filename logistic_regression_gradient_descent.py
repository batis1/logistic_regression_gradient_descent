# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:10:29 2021

@author: mohammed batis - 18511160002
"""

cur_x = 20 # The algorithm starts at x=20
cur_y = 20 # The algorithm starts at y=20
rate = 0.1 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 4  # maximum number of iterations
iters = 0 #iteration counter
dfx  = lambda x: 2*x-20 #Gradient of our function
dfy  = lambda y:2*y-16 
fx = lambda x: pow(x - 10,2)
fy = lambda y: pow(y- 8,2)
while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    prev_y = cur_y #Store current y value in prev_y
    cur_x = cur_x - rate * dfx(prev_x) #Grad descent
    cur_y = cur_y - rate * dfy(prev_y)

    fxy = fx(prev_x) +  fy(prev_y) 
    previous_step_size = abs(cur_x - prev_x) #Change in x
    previous_step_size = abs(cur_y - prev_y) #Change in y
    iters = iters+1 #iteration count
    
    print("iteration",iters,"\nX value is",round(prev_x,2),"\nY value is",round(prev_y,2),"\ngradient=",round(dfx(prev_x),2),"î+",round(dfy(prev_y),2),"ĵ","\nf(x,y)=",round(fxy,2))
    print("The peak occurs at",cur_x)
    print("*"*20)
    
print("The local minimum occurs at", round(cur_x,2))
