# ./build/basic_serial --nx 250 --nx 250 --num_iter 10000 --scenario wave --output serial.out
# ./build/gpu --nx 250 --nx 250 --num_iter 10000 --scenario wave --output serial.out
# ./build/mpi --nx 250 --nx 250 --num_iter 10000 --scenario wave --output serial.out

# make option: mpi gpu serial basic_serial
make serial
./build/serial --nx 250 --nx 250 --num_iter 1000 --output serial.out
python utils/visualizer.py serial.out serial.gif