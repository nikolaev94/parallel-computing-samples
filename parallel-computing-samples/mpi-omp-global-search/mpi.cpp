
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <Windows.h>

typedef double(*pfunction)(double);

typedef struct node {
    double x, z;
} Node;

typedef struct segment {
    int index;
    double lipConst;
} Segment;

typedef struct result
{
    int count;
    double x, z;
} Result;

double targetFunction(double x) {
    Sleep(10);
    return sin(18. * x - 3.) * cos(10. * x - 7.) + 1.5;
}

int compareNode(const void* p1, const void* p2) {
    Node* l = (Node*) p1;
    Node* r = (Node*) p2;
    double diff = l->x - r->x;
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}

int compareSegment(const void* p1, const void* p2) {
    Segment* l = (Segment*) p1;
    Segment* r = (Segment*) p2;
    double diff = l->lipConst - r->lipConst;
    if (diff < 0) {
        return 1;
    } else if (diff > 0) {
        return -1;
    } else {
        return 0;
    }
}

void globalSearch(pfunction function,
    const double leftBound,
    const double rightBound,
    const double methodParameter,
    const double methodEpsilon,
    Result* outResult,
    int procSize,
    int procRank,
    int threadsNum) {

    const int MAX_ITERS = 1000;
    Node* nodes = (Node*)malloc(sizeof(Node) * MAX_ITERS  * procSize * threadsNum + 2);

    Node left = {leftBound, (*function)(leftBound)};
    Node right = {rightBound, (*function)(rightBound)};
    nodes[0] = left;
    nodes[1] = right;
    int count = 0;
    int nodesNum = 2;
    bool isDone = false;
    bool isSingle = false;
    bool noPassing = false;
    omp_set_num_threads(threadsNum);
    while((count <= MAX_ITERS) && (!isDone))
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double lipMax = 0;
        double* segments = (double*)malloc(sizeof(double) * 4 * threadsNum);
        double* test     = (double*)malloc(sizeof(double) * 5 * threadsNum);
        if (procRank == 0)
        {
            if (nodesNum > 2)
            {
                qsort(nodes, nodesNum, sizeof(Node), compareNode);
            }
            lipMax = DBL_MIN;
            for (int i = 1; i < nodesNum; i++)
            {
                double lip = fabs((nodes[i].z - nodes[i - 1].z)
                    / (nodes[i].x - nodes[i - 1].x));
                if (lip > lipMax)
                {
                    lipMax = lip;
                }
            }
            if (lipMax > 0)
            {
                lipMax *= methodParameter;
            } else
            {
                lipMax = 1;
            }

            Segment* allSegments = (Segment*)malloc(sizeof(Segment) * (nodesNum - 1));
            for (int i = 0; i < nodesNum - 1; i++)
            {
                allSegments[i].index = i + 1;
                allSegments[i].lipConst = lipMax * (nodes[i + 1].x - nodes[i].x)
                    + (pow(nodes[i + 1].z - nodes[i].z, 2)
                        / (lipMax * (nodes[i + 1].x - nodes[i].x)))
                    - 2 * (nodes[i + 1].z + nodes[i].z);
            }
            qsort(allSegments, nodesNum - 1, sizeof(Segment), compareSegment);

            if (nodesNum - 1 >= threadsNum)
            {
                isSingle = false;
            }
            else
            {
                isSingle = true;
            }
            MPI_Bcast(&nodesNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (procSize == 1)
            {
                noPassing = true;
            }
            else
            {
                MPI_Status stat;
                MPI_Recv(&noPassing, 1, MPI_C_BOOL,
                         MPI_ANY_SOURCE,
                         MPI_ANY_TAG,
                         MPI_COMM_WORLD,
                         &stat);
            }
            if (!noPassing)
            {
                MPI_Bcast(&lipMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                for (int j = 1; j < procSize; j++)
                {
                    for (int k = 0; k < threadsNum; k++)
                    {
                        segments[4 * k] = nodes[allSegments[j * threadsNum + k].index].x; // right x
                        segments[4 * k + 1] = nodes[allSegments[j * threadsNum + k].index - 1].x; // left x
                        segments[4 * k + 2] = nodes[allSegments[j * threadsNum + k].index].z; // right z
                        segments[4 * k + 3] = nodes[allSegments[j * threadsNum + k].index - 1].z; // left z
                    }
                    MPI_Send(segments, 4 * threadsNum, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
                }
            }
            for (int k = 0; k < threadsNum; k++)
            {
                segments[4 * k] = nodes[allSegments[k].index].x;
                segments[4 * k + 1] = nodes[allSegments[k].index - 1].x;
                segments[4 * k + 2] = nodes[allSegments[k].index].z;
                segments[4 * k + 3] = nodes[allSegments[k].index - 1].z;
                if (isSingle)
                {
                    break;
                }
            }
            free(allSegments);
        }
        else
        {
            MPI_Bcast(&nodesNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (nodesNum - 1 >= procSize * threadsNum) {
                noPassing = false;
                if (procRank == 1)
                {
                    MPI_Send(&noPassing, 1, MPI_C_BOOL, 0, procRank, MPI_COMM_WORLD);
                }
                MPI_Bcast(&lipMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Status stat;
                MPI_Recv(segments, 4 * threadsNum, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
            }
            else
            {
                noPassing = true;
                if (procRank == 1)
                {
                    MPI_Send(&noPassing, 1, MPI_C_BOOL, 0, procRank, MPI_COMM_WORLD);
                }
                goto end;
            }
        }
        // ================================================================================
        #pragma omp parallel if(!isSingle)
        {
            int threadId = omp_get_thread_num();
            double xnew = ((segments[threadId * 4]
                            + segments[threadId * 4 + 1]) / 2)
                            - ((segments[threadId * 4 + 2]
                            - segments[threadId * 4 + 3]) / (2 * lipMax));
            double znew = (*function)(xnew);
            test[threadId * 5]     = xnew;
            test[threadId * 5 + 1] = znew;
            test[threadId * 5 + 2] = segments[threadId * 4] - segments[threadId * 4 + 1];
            test[threadId * 5 + 3] = segments[threadId * 4];
            test[threadId * 5 + 4] = segments[threadId * 4 + 2];
        }
        // ================================================================================
        if (procRank != 0)
        {
            MPI_Send(test, 5 * threadsNum, MPI_DOUBLE, 0, procRank, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < procSize; j++)
            {
                // when j = 0 point is already initialized
                if (j != 0)
                {
                    MPI_Status stat;
                    MPI_Recv(test, 5 * threadsNum, MPI_DOUBLE, j, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
                }
                for (int k = 0; k < threadsNum; k++)
                {
                    Node point = {test[5 * k], test[5 * k + 1]};
                    nodes[nodesNum] = point;
                    nodesNum++;
                    if ((test[5 * k + 2] < methodEpsilon) || (count == MAX_ITERS))
                    {
                        outResult->x = test[5 * k + 3];
                        outResult->z = test[5 * k + 4];
                        outResult->count = count;
                        isDone = true;
                    }
                    if (isSingle)
                    {
                        break;
                    }
                }
                if (noPassing)
                {
                    break;
                }
            }
        }
        end:
        MPI_Bcast(&isDone, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        count++;

        free(test);
        free(segments);
    }
    free(nodes);
}

int main(int argc, char** argv) {
    int nthreads = 1;
    if (argc > 1) {
        nthreads = atoi(argv[1]);
        if (nthreads <= 0)
            nthreads = 1;
    }
    Result result = {-1, 0, 0};
    const double LEFT_BOUND       = -10.0;
    const double RIGHT_BOUND      = 10.0;
    const double METHOD_PARAMETER = 2.25;
    const double METHOD_EPSILON   = 0.0001;

    int procNum, procRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    double t = MPI_Wtime();
    globalSearch(targetFunction,
                 LEFT_BOUND,
                 RIGHT_BOUND,
                 METHOD_PARAMETER,
                 METHOD_EPSILON,
                 &result,
                 procNum,
                 procRank,
                 nthreads);
    t = MPI_Wtime() - t;
    if (procRank == 0)
    {
        printf("Elapsed time: %f \n", t);
        printf("X* = %f Z* = %f Count = %d\n", result.x, result.z, result.count);
    }
    MPI_Finalize();
    return 0;
}
