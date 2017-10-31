from random import *

eingabe = input()

for x in range (1,int(eingabe)):
	y=randint(1,20)
	print([int(i) for i in str(randint(1, 10**y))])
