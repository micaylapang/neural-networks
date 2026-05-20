import sys; args = sys.argv[1:]
file = open("x_gate.txt")
import math, random

def transfer(t_funct, input):
   if t_funct == 'T3': return [1 / (1 + math.e**-x) for x in input]
   elif t_funct == 'T4': return [-1+2/(1+math.e**-x) for x in input]
   elif t_funct == 'T2': return [x if x > 0 else 0 for x in input]
   else: return [x for x in input]

def dot_product(input, weights, stage):
   return [sum([input[x]*weights[x+s*len(input)] for x in range(len(input))]) for s in range(stage)]

def ff(ts, xv, weights, t_funct):
   for n in range(len(weights)-1): 
      x = dot_product(xv[n], weights[n], len(xv[n+1]))
      y = transfer(t_funct, x)
      xv[n+1] = y
   for n in range(len(xv[-1])):
      xv[-1][n] = xv[-2][n]*weights[-1][n]
   err = sum([(ts[i-len(xv[-1])] - xv[-1][i])**2 for i in range(len(xv[-1]))])/2
   return xv, err

def bp(ts,xv,weights,ev,ng):  
   for i in range(len(xv[-1])):  
      ev[-1][i]=ts[-len(xv[-1])+i]-xv[-1][i]
      ng[-1][i]=ev[-1][i]*xv[-2][i]
   for i in range(2):
      for e in range(len(ev[2-i])): 
         ev[2-i][e]=(sum(weights[2-i][e+j*len(ev[2-i])]*ev[3-i][j] for j in range(len(ev[3-i])))
               *xv[2-i][e]*(1-xv[2-i][e]))
      for n in range(len(ng[1-i])): 
         ng[1-i][n]=ev[2-i][n//len(xv[1-i])]*xv[1-i][n%len(xv[1-i])]
   return ev,ng

def update_weights(weights, ng, alpha):
   nw = [[*i] for i in weights] 
   for i in range(len(weights)): 
      for n in range(len(weights[i])): 
         nw[i][n] = ng[i][n]*alpha+weights[i][n]
   return nw

def main():
   t_funct = 'T3'
   training_set = [] 
   inp = 0 
   out = 0 
   for line in file: 
      parts = line.strip().split('=>')
      inputs, outputs = parts[0].strip(), parts[1].strip()
      input_vals = list(map(float, inputs.split()))
      output_vals = list(map(float, outputs.split()))
      arr = input_vals+output_vals
      training_set.append(arr)
      out = len(output_vals)
      inp = len(input_vals)
   layer_counts = [inp+1, out+1, out, out] 
   print ('Layer counts: ', layer_counts)
   x_vals = [[temp[0:len(temp)-1]] for temp in training_set] 
   for i in range(len(training_set)):
      for j in range(len(layer_counts)):
         if j == 0: x_vals[i][j].append(1.0)
         else: x_vals[i].append([0 for temp in range(layer_counts[j])])
   print(x_vals)
   weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
   weights.append([round(random.uniform(-2.0, 2.0), 2) for i in range(layer_counts[-1])])
   ev = [[[*i] for i in j] for j in x_vals]  
   ng = [[*i] for i in weights] 
   errors = [10]*len(training_set) 
   count = 1 
   alpha = 0.3
   
   for k in range(len(training_set)):
      x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
   error = sum(errors)
   while error>1: 
      weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])]  for i in range(len(layer_counts)-2)]
      weights.append([round(random.uniform(-2.0, 2.0), 2) for i in range(layer_counts[-1])])
      for k in range(len(training_set)):
         x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
      error = sum(errors)

   while error>0.01 and count<20000: 
      for k in range(len(training_set)):
         ev[k], ng = bp(training_set[k], x_vals[k], weights, ev[k], ng)
         weights = update_weights(weights, ng, alpha)
         x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
         count+=1
      error = sum(errors)
   print("Errors: ", errors)
   print ('Weights:')
   for w in weights: 
      print (w)

if __name__ == '__main__': main()