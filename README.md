# parallel-computing-samples
MPI/OpenMP/TBB usage samples

## MPI Gauss-Jordan elimination
Sample usage of pair and collective operations. Test example is taken from:
http://linearalgebra4fun.blogspot.com/2015/06/

## MPI Cannon's algorithm
Sample usage of a cartesian topology. Sample processes 6x6 matrices using 3x3 grid (9 ranks). Method description can be found at:
http://parallelcomp.uw.hu/ch08lev1sec2.html

## Global search algorithm
Parallel global optimization algorithm. Benefitable when function evaluation is time-expensive. Target function plot via WolframAlpha:
https://www.wolframalpha.com/input/?i=min%28sin%2818.+*+x+-+3.%29+*+cos%2810.+*+x+-+7.%29+%2B+1.5%29+x+from+-3+to+3

### MPI / OpenMP version
Each rank parallely processes N >= 1 best search intervals in an iteration.

### TBB version
Parallely initializes search invervals and processes N >= 1 best intervals in an iteration using **tbb::parallel_for**. Visual Studio project expects TBB to be installed in: $(USERPROFILE)\Opt\tbb-2020.3-win. You should add DLLs location $(USERPROFILE)\Opt\tbb-2020.3-win\tbb\bin\ia32\vc14 to PATH for debugging purposes. Code tested with this TBB release:
https://github.com/oneapi-src/oneTBB/releases/tag/v2020.3
