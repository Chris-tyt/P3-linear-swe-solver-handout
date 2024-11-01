# option: mpi gpu serial serial_omp basic_serial 
# --scenario water_drop dam_break wave river
mod=serial_omp
scenario=water_drop
nx=1000
ny=1000
niter=10000
make $mod
./build/${mod} --nx ${nx} --nx ${ny} --num_iter ${niter} --scenario ${scenario} --output ${mod}.out
python utils/visualizer.py ${mod}.out ${mod}.gif