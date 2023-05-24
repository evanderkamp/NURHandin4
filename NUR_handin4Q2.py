import numpy as np
import matplotlib.pyplot as plt
import timeit

#exercise 2

#a

#load in the given code
np.random.seed(121)

n_mesh = 16
n_part = 1024
positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))

grid = np.arange(n_mesh) + 0.5
densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
cellvol = 1.

for p in range(n_part):
    cellind = np.zeros(shape=(3, 2))
    dist = np.zeros(shape=(3, 2))

    for i in range(3):
        cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
                              (abs(positions[i, p] - grid - 16) < 1) | 
                              (abs(positions[i, p] - grid + 16) < 1))[0]
        dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])

    cellind = cellind.astype(int)

    for (x, dx) in zip(cellind[0], dist[0]):    
        for (y, dy) in zip(cellind[1], dist[1]):
            for (z, dz) in zip(cellind[2], dist[2]):
                if dx > 15: dx = abs(dx - 16)
                if dy > 15: dy = abs(dy - 16)
                if dz > 15: dz = abs(dz - 16)

                densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol

#calculate mean density and density contrast
meandens = n_part/(n_mesh**3)

denscontr = (densities - meandens)/meandens

print(len(grid))
#the grid shows the coordinates so the index is the coordinate - 0.5
plt.imshow(densities[:, :, 4])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 4.5 slice")
plt.colorbar(label="density contrast")
plt.savefig("NUR4Q2denscontr4.png")
plt.close()

#z=9.5
plt.imshow(densities[:, :, 9])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 9.5 slice")
plt.colorbar(label="density contrast")
plt.savefig("NUR4Q2denscontr9.png")
plt.close()

#z=11.5
plt.imshow(densities[:, :, 11])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 11.5 slice")
plt.colorbar(label="density contrast")
plt.savefig("NUR4Q2denscontr11.png")
plt.close()

#z=14.5
plt.imshow(densities[:, :, 14])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 14.5 slice")
plt.colorbar(label="density contrast")
plt.savefig("NUR4Q2denscontr14.png")
plt.close()


#b
def DFT(x, Nj):
    """Takes an array x and recursively calls on itself with the even and odd elements of x with Nj/2 and transforms it."""
    even = x[::2].copy()
    odd = x[1::2].copy()
    if Nj > 2:
        #even elements
        even = DFT(even, Nj*0.5)
        #odd elements
        odd = DFT(odd, Nj*0.5)
    
    for k in range(0,int(Nj*0.5)):

        t = even[k].copy()
        xknj = odd[k].copy()
        exp = np.exp(2j*np.pi*k/Nj)
        x[k] = t + exp*xknj
        x[int(k+Nj*0.5)] = t - exp*xknj
        
        
    return x

def FFT(x):
    """FFT routine calling on the DFT routine"""
    #make a copy of x to change and change it to a complex type
    arr = x.copy().astype(complex)
    N = len(arr)
    
    arr = DFT(arr, N)
    #print("end arr", arr)
    
    return arr


def IDFT(x, Nj):
    """Inverse DFT routine that takes an array x and recursively calls on itself with the even and odd elements of x with Nj/2 and transforms it back by doing exp(-2j) instead of exp(2j)."""
    even = x[::2].copy()
    odd = x[1::2].copy()
    if Nj > 2:
        #even elements
        even = IDFT(even, Nj*0.5)
        #odd elements
        odd = IDFT(odd, Nj*0.5)
    
    for k in range(0,int(Nj*0.5)):
        t = even[k].copy()
        xknj = odd[k].copy()
#now exp(-2j) instead of exp(+2j) like in the DFT
        exp = np.exp(-2j*np.pi*k/Nj)
        x[k] = t + exp*xknj
        x[int(k+Nj*0.5)] = t - exp*xknj
        

    return x

def IFFT(x):
    """Inverse FFT that calls on the inverse DFT routine."""
    #make a copy of x to change and change it to a complex type
    arr = x.copy().astype(complex)
    N = len(arr)

    arr = IDFT(arr, N)
#divide by N at the end to properly inverse
    arr = arr/N
    #print("end arr", arr)
    
    return arr

#make an array to store the FFT in
contrfft = np.array(denscontr, dtype=complex)

#do the FFT of all 3 dimensions separately
for i in range(len(denscontr[0,:,0])):
    contrfft[:,i,:] = FFT(contrfft[:,i,:])
for j in range(len(denscontr[:,0,0])):
    contrfft[j,:,:] = FFT(contrfft[j,:,:])
for p in range(len(denscontr[0,0,:])):
    contrfft[:,:,p] = FFT(contrfft[:,:,p])

#now we have delta tilde so we have to rewrite it (divide by k^2 which are the gridpoints)
potentfft = contrfft / grid**2 

pot = np.array(potentfft, dtype=complex)

#do the IFFT of all 3 dimensions separately
for i in range(len(potentfft[0,:,0])):
    pot[:,i,:] = IFFT(pot[:,i,:])
for j in range(len(potentfft[:,0,0])):
    pot[j,:,:] = IFFT(pot[j,:,:])
for p in range(len(potentfft[0,0,:])):
    pot[:,:,p] = IFFT(pot[:,:,p])

#transform it back to float instead of complex so we can plot it
pot = np.array(pot, dtype=float)
plt.imshow(pot[:, :, 4])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 4.5 slice, potential")
plt.colorbar(label="gravitational potential")
plt.savefig("NUR4Q2gravpot4.png")
plt.close()

#z=9.5
plt.imshow(pot[:, :, 9])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 9.5 slice, potential")
plt.colorbar(label="gravitational potential")
plt.savefig("NUR4Q2gravpot9.png")
plt.close()


#z=11.5
plt.imshow(pot[:, :, 11])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 11.5 slice, potential")
plt.colorbar(label="gravitational potential")
plt.savefig("NUR4Q2gravpot11.png")
plt.close()


#z=14.5
plt.imshow(pot[:, :, 14])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z = 14.5 slice, potential")
plt.colorbar(label="gravitational potential")
plt.savefig("NUR4Q2gravpot14.png")
plt.close()

#now the log10 of the absolute value of the FFT potential
potfftabs = np.log10(np.abs(potentfft))
plt.imshow(potfftabs[:, :, 4])
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"z = 4.5 slice, $\log_{10}(|\~\Phi|)$")
plt.colorbar(label=r"$\log_{10}(|\~\Phi|)$")
plt.savefig("NUR4Q2abs4.png")
plt.close()

#z=9.5
plt.imshow(potfftabs[:, :, 9])
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"z = 9.5 slice, $\log_{10}(|\~\Phi|)$")
plt.colorbar(label=r"$\log_{10}(|\~\Phi|)$")
plt.savefig("NUR4Q2abs9.png")
plt.close()


#z=11.5
plt.imshow(potfftabs[:, :, 11])
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"z = 11.5 slice, $\log_{10}(|\~\Phi|)$")
plt.colorbar(label=r"$\log_{10}(|\~\Phi|)$")
plt.savefig("NUR4Q2abs11.png")
plt.close()


#z=14.5
plt.imshow(potfftabs[:, :, 14])
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"z = 14.5 slice, $\log_{10}(|\~\Phi|)$")
plt.colorbar(label=r"$\log_{10}(|\~\Phi|)$")
plt.savefig("NUR4Q2abs14.png")
plt.close()

