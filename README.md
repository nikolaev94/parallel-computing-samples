# parallel-computing-samples
MPI/OpenMP/TBB usage samples

## MPI Gauss-Jordan elimination
Sample usage of pair and collective operations. Test example is taken from:
http://linearalgebra4fun.blogspot.com/2015/06/

## MPI Cannon's algorithm
Sample usage of a cartesian topology. Sample processes 6x6 matrices using 3x3 grid (9 ranks). Method description can be found at:
http://parallelcomp.uw.hu/ch08lev1sec2.html

## MPI / OpenMP Global search algorithm
Parallel global optimization algorithm.
Each rank parallely processes N >= 1 search intervals in an iteration. Benefitable when function evaluation is time-expensive.
Target function plot via WolframAlpha:
https://www.wolframalpha.com/input/?i=min%28sin%2818.+*+x+-+3.%29+*+cos%2810.+*+x+-+7.%29+%2B+1.5%29+x+from+-3+to+3
