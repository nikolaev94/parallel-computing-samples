
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>

typedef double(*pfunction)(double);

struct Node {
    double x, z;
};

struct Segment {
    int index;
    double lipConst;
};

struct Result
{
    int count;
    double x, z;
};

double targetFunction(double x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

Result outResult = {-1, 0, 0};
int nodesNum = 0;
bool isDone = true;
Node* nodes = NULL;

// Class to evaluate function in a new node
class NodeInitializer {
private:
    static tbb::mutex mutex;
    const Segment* segment;
    const pfunction function;
    const double lipMax, eps;
    const int count;

public:
    NodeInitializer(Segment* _segment,
                    pfunction _pfunction,
                    double _lipMax,
                    double _eps,
                    int _count):
                    segment(_segment),
                    function(_pfunction),
                    lipMax(_lipMax),
                    eps(_eps),
                    count(_count) {}

    void operator() (const tbb::blocked_range<int>& r) const {
        int begin = r.begin();
        int end = r.end();
        int segsNum = nodesNum - 1;
        for (int i = begin; i != end; i++) {
            if (i >= segsNum)
                continue;
            int j = segment[i].index;
            double xnew = ((nodes[j].x + nodes[j - 1].x) / 2)
                        + ((nodes[j].z - nodes[j - 1].z)
                        / (2 * lipMax));
            double znew = (*function)(xnew);
            Node point = {xnew, znew};
            tbb::mutex::scoped_lock lock;
            lock.acquire(mutex);
            nodes[nodesNum] = point;
            nodesNum++;
            outResult.x = nodes[j].x;
            outResult.z = nodes[j].z;
            outResult.count = count;
            if ((nodes[j].x - nodes[j - 1].x) < eps)
            {
                isDone = true;
            }
            lock.release();
        }
    }
};

tbb::mutex NodeInitializer::mutex;

// Class to reinitialize search segments
class SegmentInitializer {
private:
    Segment* allSegments;
    const Node* nodes;
    const double lipMax;

public:
    SegmentInitializer(Segment* _allSegments,
                    Node* _nodes,
                    double _lipMax):
                    allSegments(_allSegments),
                    nodes(_nodes),
                    lipMax(_lipMax) {}

    void operator() (const tbb::blocked_range<int>& r) const
    {
        int begin = r.begin();
        int end = r.end();
        for (int i = begin; i != end; i++)
        {
            allSegments[i].index = i + 1;
            allSegments[i].lipConst = lipMax * (nodes[i + 1].x - nodes[i].x)
                + (pow(nodes[i + 1].z - nodes[i].z, 2)
                / (lipMax * (nodes[i + 1].x - nodes[i].x)))
                - 2 * (nodes[i + 1].z + nodes[i].z);
        }
    }
};

void globalSearch(pfunction function,
    const double leftBound,
    const double rightBound,
    const double methodParameter,
    const double methodEpsilon,
    Result& outResult,
    int threadsNum) {

    const int MAX_ITERS = 1000;
    nodes = new Node [MAX_ITERS * 2 * threadsNum + 2];
    Node left = { leftBound, (*function)(leftBound) };
    Node right = { rightBound, (*function)(rightBound) };
    nodes[0] = left;
    nodes[1] = right;
    int count = 1;
    nodesNum = 2;
    isDone = false;
    while((count <= MAX_ITERS) && (!isDone))
    {
        if (nodesNum > 2)
        {
            qsort(nodes, nodesNum, sizeof(Node), compareNode);
        }

        double lipMax = DBL_MIN;
        for (int i = 1; i < nodesNum; i++)
        {
            double lip = std::abs((nodes[i].z - nodes[i - 1].z)
                        / (nodes[i].x - nodes[i -1].x));
            if (lip > lipMax)
            {
                lipMax = lip;
            }
        }

        if (lipMax > 0)
        {
            lipMax *= methodParameter;
        }
        else
        {
            lipMax = 1;
        }

        Segment* allSegments = new Segment[nodesNum - 1];
        tbb::parallel_for(tbb::blocked_range<int>(0, nodesNum - 1),
                          SegmentInitializer(allSegments, nodes, lipMax));
        qsort(allSegments, nodesNum - 1, sizeof(Segment), compareSegment);
        tbb::parallel_for(tbb::blocked_range<int>(0, 2 * threadsNum, 2),
                          NodeInitializer(allSegments,
                                          function,
                                          lipMax,
                                          methodEpsilon,
                                          count));
        count++;
        delete[] allSegments;
    }
    delete[] nodes;
}

int main(int argc, char** argv) {
    int nthreads = 1;
    if (argc > 1) {
        nthreads = atoi(argv[1]);
        if (nthreads <= 0)
        {
            nthreads = 1;
        }
    }
    const double LEFT_BOUND = -10.0;
    const double RIGHT_BOUND = 10.0;
    const double METHOD_PARAMETER = 2.25;
    const double METHOD_EPSILON = 0.0001;
    tbb::task_scheduler_init init(nthreads);
    tbb::tick_count t0 = tbb::tick_count::now();
    globalSearch(targetFunction,
                 LEFT_BOUND,
                 RIGHT_BOUND,
                 METHOD_PARAMETER,
                 METHOD_EPSILON,
                 outResult,
                 nthreads);
    tbb::tick_count t1 = tbb::tick_count::now();
    printf("Elapsed time: %f \n", (t1 - t0).seconds());
    printf("X* = %f Z = %f Count = %d\n", outResult.x, outResult.z, outResult.count);
    return 0;
}
