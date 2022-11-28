import numpy as np
list1 = [('abc', 'def')]
list1 = [list1[0][i] for i in range(2)]
list2 = [i for i in range(10)]
print(list1)
print(list2)
data = [i for i in range(10)]
data = np.array(data)
np.random.shuffle(data)
print(data)
y_p = np.sign(np.array([0.5, -0.2]))
print(y_p)