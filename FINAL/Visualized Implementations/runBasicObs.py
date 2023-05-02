from prm import prm
from anc_prm import anc_prm

XDIM = 640
YDIM = 480
w = 100
w2 = 25
h = 150
h2 = 60
X0 = XDIM/5
Y0 = YDIM/3
X1 = 4*XDIM/5
Y1 = 2*YDIM/3
CENTER_SIZE = 20
Obs = {}

Obs[0] = [(XDIM/2, YDIM/7),(XDIM/2, YDIM/3)]
Obs[1] = [(XDIM/2, 6*YDIM/7),(XDIM/2, 2*YDIM/3)]

Obs[2] = [(XDIM/6, YDIM/2),(XDIM/3, YDIM/2)]
Obs[3] = [(5*XDIM/6, YDIM/2),(2*XDIM/3, YDIM/2)]
Obs[4] = [(3*XDIM/6, YDIM/2+20),(2*XDIM/3+24, YDIM/2+60)]
Obs[5] = [(XDIM/2+100, 6*YDIM/7+50),(XDIM/2+140, 2*YDIM/3+30)]

XY_START = (X0+w/2,Y0+3*h/4) # Start in the trap
XY_GOAL = (4*XDIM/5,5*YDIM/6)
XY_GOAL = (X1-w/2,Y1-3*h/4)

game = prm(Obs,XY_START,XY_GOAL, XDIM, YDIM)
#game = anc_prm(Obs, XY_START, XY_GOAL, XDIM, YDIM)
game.runGame()