import matplotlib.pyplot as plt 
import numpy as np
objects = ("Linear_reg", "Nonlinear_reg" ,"Logistic" , "Decision_Tree")
x = np.arange(len(objects))
y = [81.43 , 81.43 , 92.86 , 93.81]
plt.bar( x, y, align='center', alpha=0.5)

plt.xticks(x,objects)
plt.ylabel('Performance(accuracy)')
plt.title('Algorithms Performance')
plt.legend('%')
plt.show()