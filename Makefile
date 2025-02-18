.PHONY: clean all dep tests test_cpu test_gpu test_formal python
# .PHONY: test_model_cpu test_model_gpu test_model_formal
# .PHONY: test_op_cpu test_op_gpu test_op_formal

BUILD := build
INCLUDE := include
TESTS := tests

all: lib python tests html
	echo ${TEST_CPUS}

# Mac OS should install libomp with brew
dep:
	python3 install/deps.py

lib: dep
	@cd ${BUILD} && cmake ../ && $(MAKE)

# Make sure install the python dependency package before
# 	make python target.
python: lib
	@cd python && python3 setup.py install

html:
	@make -C docs html

TEST_SRCS := $(wildcard ${TESTS}/*.cc)
TEST_EXES := $(patsubst ${TESTS}/%.cc,%,${TEST_SRCS})

TEST_CPUS := $(patsubst %,%_cpu,${TEST_EXES})
TEST_GPUS := $(patsubst %,%_gpu,${TEST_EXES})
TEST_FORMALS := $(patsubst %,%_formal,${TEST_EXES})
TEST_OPENCL := $(patsubst %,%_opencl,${TEST_EXES})

tests: lib test_cpu test_gpu test_formal

test_cpu: ${TEST_CPUS}
test_gpu: ${TEST_GPUS}
test_formal: ${TEST_FORMALS}
test_opencl: ${TEST_OPENCL}

%_cpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=0 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_gpu: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=1 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_formal: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=2 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm -fopenmp -fsigned-char -pthread -Wl,-rpath=${BUILD}

%_opencl: ${TESTS}/%.cc lib
	g++ -o ${BUILD}/${TESTS}/$@ $< -DDEVICE=3 -std=c++11 -I${INCLUDE} -L${BUILD} -lcvm_runtime -fopenmp -L/usr/local/cuda/lib64/ -lOpenCL -fsigned-char -pthread -Wl,-rpath=${BUILD}

clean:
	rm -rf ./build/*
