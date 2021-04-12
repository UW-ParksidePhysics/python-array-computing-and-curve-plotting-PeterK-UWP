#5.1




from numpy import sqrt, exp, pi
from matplotlib.pyplot import plot, show, xlabel, ylabel

def h(x):
    return 1 / sqrt(2 * pi) * exp(-0.5 * x * x)

n = 41
dx = 8. / (n - 1)

xlist = [-4 + i * dx for i in range(n)]
hlist = [h(x) for x in xlist]

plot(xlist, hlist)
xlabel('x')
ylabel('h')
show()


#def h(x):
 # return 1/(sqrt(2*pi)) * e**-(0.5*x**2)

#n = 41
#dx = (4-(-4))/n
#xlist = [i*dx for i in range(n)]
#ylist = [h(x) for x in xlist]
#pairs = [[x, y] for x, y in zip(xlist, ylist)]

#show()
