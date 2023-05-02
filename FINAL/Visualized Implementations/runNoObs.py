from prm import prm
from anc_prm import anc_prm

XDIM = 640
YDIM = 480
Obs = {}

XY_START = (XDIM/3,YDIM/6)
XY_GOAL = (4*XDIM/5,5*YDIM/6)

#game = prm(Obs, XY_START, XY_GOAL, XDIM, YDIM)
game = anc_prm(Obs, XY_START, XY_GOAL, XDIM, YDIM)
game.runGame()