import random

from r0123456 import r0123456

a = r0123456()
b  = [([1,2,3],2),([1,3,4],54)]

matingPool = [a.selection(a.createInitialPopulation(a.distanceM("tour29.csv"))) for _ in range(5)]
matingPoolTours = [a_tuple[0] for a_tuple in matingPool]
random.shuffle(matingPoolTours)
[index1,index2] = random.sample(list(range(10)),2)
c= list(range(1,30))
print(c)
print(a.optimize("tour29.csv"))

 