import sys; args = sys.argv[1:]
file = open("x_gate.txt")
import math


def transfer(t_funct, input):
    if t_funct=='T1': 
        return input
    if t_funct=='T2':
        return input if input>0 else 0
    if t_funct=='T3': 
        return 1/(1+math.exp(-input))
    if t_funct=='T4':
        return -1 + 2/(1+math.exp(-input))
    return input 

def dot_product(input, weights):
    sum = 0
    for i in range(len(input)): 
        sum += input[i]*weights[i]
    return sum 

def evaluate(file, inputs, t_funct):
    wts = []
    for line in file: 
        nums = list(map(float, line.strip().split()))
        wts.append(nums)
    xvals = [inputs]
    for i, li in enumerate(wts[:-1]): 
        n = len(li)//len(xvals[i])
        xli = []
        for x in range(n): 
            s = dot_product(xvals[i], li[x*len(xvals[i]):(x+1)*len(xvals[i])])
            res = transfer(t_funct, s)
            xli.append(res)
        xvals.append(xli)
    ret = []
    for i, x in enumerate(wts[-1]):
        ret.append(x*xvals[-1][i])
    return ret

def main():
    inputs, t_funct, transfer_found = [], 'T1', False
    for arg in args[1:]:
        if not transfer_found:
            t_funct, transfer_found = arg, True
        else:
            inputs.append(float(arg))
    li = evaluate(file, inputs, t_funct)
    for x in li:
        print (x, end=' ')

if __name__ == '__main__': main()