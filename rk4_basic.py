def calc(t,y):
	output = 2 * t
	return output

t_i = 0.0
y_i = 0.0
h = 0.1

#RK4 Code
def rk4(t_i, y_i, h):
	rk1 = h * calc(t_i, y_i)
	rk2 = h * calc(t_i + (0.5*h), y_i + (0.5*rk1))
	rk3 = h * calc(t_i + (0.5*h), y_i + (0.5*rk2))
	rk4 = h * calc(t_i + h, y_i + rk3)
	return y_i + ((1/6)*(rk1+2*rk2+2*rk3+rk4))

for i in range(0, 1000):
	output = rk4(t_i, y_i, h)
	print(output) 
	t_i += h
	y_i = output

