import numpy as np
import matplotlib.pyplot as plt
import timeit
import astropy.constants as c

#exercise 1

#a

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel

t0 = Time("2021-12-07 10:00")

with solar_system_ephemeris.set('jpl'):
	sun = get_body_barycentric_posvel('sun', t0)
	merc = get_body_barycentric_posvel('mercury', t0)
	venus = get_body_barycentric_posvel('venus', t0)
	earth = get_body_barycentric_posvel('earth', t0)
	mars = get_body_barycentric_posvel('mars', t0)
	jup = get_body_barycentric_posvel('jupiter', t0)
	sat = get_body_barycentric_posvel('saturn', t0)
	uran = get_body_barycentric_posvel('uranus', t0)
	nept = get_body_barycentric_posvel('neptune', t0)


#put positions and velocities into an array for convenience
solsyspos = np.array([sun[0], merc[0], venus[0], earth[0], mars[0], jup[0], sat[0], uran[0], nept[0]])
solsysvel = np.array([sun[1], merc[1], venus[1], earth[1], mars[1], jup[1], sat[1], uran[1], nept[1]])

solsysname = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])

for l in range(len(solsyspos)):
	posx = solsyspos[l].x.to_value(u.AU)
	posy = solsyspos[l].y.to_value(u.AU)
	
	plt.scatter(posx, posy, label=solsysname[l])

plt.title("Solar System positions 2021-12-07 10:00")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.legend()
plt.savefig("NUR4Q1solsysxy.png")
plt.close()

for k in range(len(solsyspos)):
	posz = solsyspos[k].z.to_value(u.AU)
	
	plt.scatter(posx, posz, label=solsysname[k])

plt.title("Solar System positions 2021-12-07 10:00")
plt.xlabel("x (AU)")
plt.ylabel("z (AU)")
plt.legend()
plt.savefig("NUR4Q1solsysxz.png")
plt.close()


#b & c
G = c.G.to_value()
Msun = c.M_sun.to_value()

#equations to calculate the acceleration
def grav(t, r):
    """calculate the acceleration due to gravity between a body and the Sun, assuming r is the x,y,z position in meter."""
    r_inv = 1/ (r[0]**2 + r[1]**2 + r[2]**2)**(3/2)
    accel = -G*Msun*r * r_inv


    return accel


def leapfrog(func,a,b,y0,z0,h):
    """leapfrog method for 2nd order differential equations that satisfy d^2y/dx^2 = func and dy/dx = z. Integrates from a until b with a stepsize of h and initial conditions y(0) = y0 and z(0) = z0."""
    #make the arrays, xs are the function values we will integrate over
    xs = np.arange(a,b+h,h)
    ys = np.zeros((len(xs),3))
    zs = np.zeros((len(xs),3))
    #set the initial conditions
    ys[0,:] = y0
    zs[0,:] = z0
    
    for i in range(len(xs)-1):
        if i == 0:
	#give the velocity its initial kick of 1/2 a step with the Euler method
            k1 = func(xs[i], ys[i,:])*h
	#this is z(1/2) = z0 + 1/2*k1
            zs[i+1,:] = zs[i,:] + 0.5*k1
        else:
	#calculate z(i+1/2) = z(i-1/2) + a_i*h
            zs[i+1,:] = zs[i,:] + func(xs[i], ys[i,:])*h
        #now calculate yi = y(i-1) + z(i+1/2)*h
        ys[i+1,:] = ys[i,:] + zs[i+1,:]*h

#return the xs, ys, and zs   
    return xs, ys, zs

def Euler(func,a,b,y0,z0,h):
    """Euler method for 2nd order differential equations that satisfy d^2y/dx^2 = func and dy/dx = z. Integrates from a until b with a stepsize of h and initial conditions y(0) = y0 and z(0) = z0."""
    #make the arrays, xs are the function values we will integrate over
    xs = np.arange(a,b+h,h)
    ys = np.zeros((len(xs),3))
    zs = np.zeros((len(xs),3))
    #set the initial conditions of the function (y) and the derivative (z)
    ys[0,:] = y0
    zs[0,:] = z0
    
    for i in range(len(xs)-1):
        zs[i+1,:] = zs[i,:] + func(xs[i], ys[i,:])*h
        ys[i+1,:] = ys[i,:] + zs[i,:]*h
        
    return xs, ys, zs


#get the value of a year and a day in secs so we can get everything in the right units
day = 3600 * 24
year = 365 * day

#time range and time steps
timestep = 0.5*day
timrang = 200*year

timerange = np.arange(0,timrang+timestep,timestep)
nrsteps = len(timerange)

#arrays to save the positions into (8 planets)
xposs = np.zeros((nrsteps, 8))
yposs = np.zeros((nrsteps, 8))
zposs = np.zeros((nrsteps, 8))

xposE = np.zeros((nrsteps, 8))
yposE = np.zeros((nrsteps, 8))
zposE = np.zeros((nrsteps, 8))

for i in range(1,len(solsyspos)):
#center the positions so that the sun is at [0,0,0] and we can accurately use the acceleration equation (in m so it works with calculating the acceleration)
	posx = (solsyspos[i].x - solsyspos[0].x).to_value(u.m)
	posy = (solsyspos[i].y - solsyspos[0].y).to_value(u.m)
	posz = (solsyspos[i].z - solsyspos[0].z).to_value(u.m)

#velocities in m/s so the acceleration calculation works alright
	velx = solsysvel[i].x.to_value(u.m/u.s)
	vely = solsysvel[i].y.to_value(u.m/u.s)
	velz = solsysvel[i].z.to_value(u.m/u.s)

	#dimensionless so we can put it in the leapfrog method
	rpos = np.array([posx,posy,posz])
	vel = np.array([velx,vely,velz])
	
	t_end, r_end, v_end = leapfrog(grav, 0, timrang, rpos, vel, timestep)
	t_eul, r_eul, v_eul = Euler(grav, 0, timrang, rpos, vel, timestep)

	xposs[:,i-1], yposs[:,i-1], zposs[:,i-1] = (r_end[:,0] * u.m).to_value(u.AU), (r_end[:,1] * u.m).to_value(u.AU), (r_end[:,2] * u.m).to_value(u.AU)
	
	xposE[:,i-1], yposE[:,i-1], zposE[:,i-1] = (r_eul[:,0] * u.m).to_value(u.AU), (r_eul[:,1] * u.m).to_value(u.AU), (r_eul[:,2] * u.m).to_value(u.AU)
	

figure = plt.figure(figsize=(12,12))
for i in range(1,len(solsyspos)):
	plt.plot(xposs[:,i-1], yposs[:,i-1], label=solsysname[i], zorder=(10-i))
	
plt.title("Solar System orbits from 2021-12-07 10:00 (t=0) until 2221-12-07 10:00 (t=200), Leapfrog")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.legend()
plt.savefig("NUR4Q1solsysxyorbits.png")
plt.close()

figure = plt.figure(figsize=(15,10))
for i in range(1,len(solsyspos)):
	plt.plot(timerange/year, zposs[:,i-1], label=solsysname[i], zorder=(10-i))
	
plt.title("Solar System z position evolution from 2021-12-07 10:00 (t=0) until 2221-12-07 10:00 (t=200), leapfrog")
plt.xlabel("t (yr)")
plt.ylabel("z (AU)")
plt.legend()
plt.savefig("NUR4Q1solsystzorbits.png")
plt.close()

figure = plt.figure(figsize=(12,12))
for i in range(1,len(solsyspos)):
	plt.plot(xposE[:,i-1], yposE[:,i-1], label=solsysname[i])
	
plt.title("Solar System orbits from 2021-12-07 10:00 until 2221-12-07 10:00, Euler Method")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.legend()
plt.savefig("NUR4Q1solsysxyEuler.png")
plt.close()

figure = plt.figure(figsize=(15,10))
for i in range(1,len(solsyspos)):
	plt.plot(timerange/year, zposE[:,i-1], label=solsysname[i], zorder=(10-i))
	
plt.title("Solar System z position evolution from 2021-12-07 10:00 (t=0) until 2221-12-07 10:00 (t=200), Euler Method")
plt.xlabel("t (yr)")
plt.ylabel("z (AU)")
plt.legend()
plt.savefig("NUR4Q1solsystzEuler.png")
plt.close()

figure = plt.figure(figsize=(15,10))
for i in range(1,len(solsyspos)):
	plt.plot(timerange/year, xposs[:,i-1]-xposE[:,i-1], label=solsysname[i])
	
plt.title("Solar System orbits from 2021-12-07 10:00 until 2221-12-07 10:00 comparison")
plt.xlabel("t (yr)")
plt.ylabel("x leapfrog - x Euler (AU)")
plt.legend()
plt.savefig("NUR4Q1solsysorbitdiff.png")
plt.close()
