import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================================
class PathWay():
    def __init__(self, id, z_depths, times):
        self.id = id
        self.z_depths = np.array([float(i) for i in z_depths])
        self.times = np.array([float(i) for i in times])
        self.velocity = 0

    def set_velocity(self, v):
    	self.velocity = v

    def return_vel(self):
    	return self.velocity

class Paths():
    def __init__(self, path):
        self.paths = [path]

    def add_path(self, path):
    	self.paths.append(path)

    def return_paths(self):
    	return self.paths


class time_steps():
    def __init__(self, num_frames):
        self.steps = list(range(num_frames))
        self.mean_vels = []
        self.densities = []

    def add_mean_vel(self, mean_vel):
    	self.mean_vels.append(mean_vel)

    def add_density(self, density):
    	self.densities.append(density)

# ============================================================================

#Number of frames calculation
os.system("ls ../results/images_mog2_2(0.01)/  | wc -l > num_frames.txt")
file = open("../results/num_frames.txt", 'r')
contents = file.read()
file.close()
num_frames = contents.split('\n')
num_frames = int(int(num_frames[0])/3)
print(num_frames)

# ============================================================================

address = '../results/path_ways.txt'
file = open(address, 'r')
contents = file.read()
file.close()
list_of_lines = contents.split('\n')


list_of_depth_lists = [lst for lst in list_of_lines 
					 if list_of_lines.index(lst)%2 == 0 ]
list_of_time_lists = [lst for lst in list_of_lines 
					 if list_of_lines.index(lst)%2 == 1 ]


if len(list_of_depth_lists) > len(list_of_time_lists):
	list_of_depth_lists.pop()
if len(list_of_depth_lists) < len(list_of_time_lists):
	list_of_time_lists.pop()

i = 0
for dp_lst, tm_lst in zip(list_of_depth_lists, list_of_time_lists):
	dp_lst = dp_lst.split(',')
	tm_lst = tm_lst.split(',')
	new_path = PathWay(i, dp_lst, tm_lst)
	if i == 0: path_admin = Paths(new_path)
	else: path_admin.add_path(new_path)
	i += 1


all_paths = path_admin.return_paths()




# ============================================================================
#plt.figure()
for path in all_paths: 
	t =  path.times
	z = path.z_depths
	t_lsp = np.linspace(0.1, 24, 300)

	
	def fit_func(t, v, z0):
	    return v*t + z0

	params = curve_fit(fit_func, t, z)
	[v, z0] = params[0]
	path.set_velocity(-v)

	plt.scatter(t, z)
	plt.plot(t_lsp, fit_func(t_lsp, v, z0), 
		label='Vel: {v} km/h'.format(v = int(v*-3.6)))

plt.title('Velocity Fittings')
#plt.xlim(0, 24)
plt.ylim(0, 45)
plt.ylabel('depth (m) position')
plt.xlabel('Time (s)')
plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
plt.savefig('../results/velocity_Fittings.png')
#plt.show()

# ============================================================================


step_admin = time_steps(num_frames)

for frame in range(0, num_frames):
	fr = frame* 1.0/23.9760239760625
	present_cars = 0
	mean_vel = 0
	for path in all_paths:
		if fr > path.times[0] and fr < path.times[-1]:
			present_cars += 1
			mean_vel += path.velocity
	if present_cars == 0: mean_vel = 0
	else: mean_vel = mean_vel/present_cars

	step_admin.add_density(present_cars)
	step_admin.add_mean_vel(mean_vel)


N = 8
chunk_number = int(len(step_admin.steps)/N)
residue = len(step_admin.steps)%N

chunks_mean_velocities = []
chunks_mean_densities = []
for chunk in range(chunk_number):
	chk_mean_vel = 0
	chk_density = 0
	for n in range(N):
		chk_mean_vel += step_admin.mean_vels[chunk + n]
		chk_density += step_admin.densities[chunk + n]
	chk_mean_vel = float(chk_mean_vel)/N
	chk_density = float(chk_density)/N
	chunks_mean_velocities.append(chk_mean_vel)
	chunks_mean_densities.append(chk_density)


if residue != 0:
	chk_mean_vel = 0
	chk_density = 0
	for n in range(chunk_number*N, len(step_admin.steps)):
		chk_mean_vel += step_admin.mean_vels[n]
		chk_density += step_admin.densities[n]
	chk_mean_vel = float(chk_mean_vel)/residue
	chk_density = float(chk_density)/residue
	chunks_mean_velocities.append(chk_mean_vel)
	chunks_mean_densities.append(chk_density)	


plt.figure()
x = step_admin.densities
y = np.array(step_admin.densities)*np.array(step_admin.mean_vels)
z = np.array(step_admin.mean_vels)
plt.scatter(x, y)
plt.ylabel('Pedestrian FLow [Per/s)]')
plt.xlabel('Density [arbitrary unit]')
plt.savefig('../results/Fundamental_diagram_1.png')
#plt.xlim(0, 4)
#plt.ylim(0, 40)
plt.legend()
plt.show()

#print(chunks_mean_densities)
plt.figure()
x = np.array(chunks_mean_densities)
y = np.array(chunks_mean_densities)*np.array(chunks_mean_velocities)
z = np.array(chunks_mean_velocities)

plt.scatter(x, y, s=10)
plt.title('Fundamental Diagram')
#plt.xlim(0, 40)
#plt.ylim(0, 40)
plt.ylabel('Pedestrian FLow (Per/s)')
plt.xlabel('Density [arbitrary unit]')
plt.legend()
plt.savefig('../results/Fundamental_diagram.png')
plt.show()

plt.scatter(x, z)
plt.title('Mean Velocity vs Density')
#plt.xlim(0, 10)
#plt.ylim(0, 40)
plt.ylabel('Mean Velocity (km/h)')
plt.xlabel('Density [arbitrary unit]')
plt.legend()
plt.savefig('../results/Mean_vel_vs_density.png')
plt.show()

