# option: mpi gpu serial basic_serial
# --scenario water_drop dam_break wave river
mod=serial
scenario=water_drop
nx=250
ny=250
niter=1000
make $mod
./build/${mod} --nx ${nx} --nx ${ny} --num_iter ${niter} --scenario ${scenario} --output ${mod}.out
python utils/visualizer.py ${mod}.out ${mod}.gif