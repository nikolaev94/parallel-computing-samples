#include <ctime>
#include <iostream>
#include <chrono>
#include <mutex>
#include <thread>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>

std::mutex g_mutex;

double scalarProductSequential(const double* vect1, const double* vect2, int size)
{
	double result = 0.0;
	for (size_t i = 0; i < size; ++i)
		result += vect1[i] * vect2[i];
	return result;
}

void productThreadRoutine(
    const double* vect1,
    const double* vect2,
    int startIdx,
    int endIdx,
    double& result
)
{
	double localResult = 0.0;
	for (int i = startIdx; i < endIdx; ++i) {
		localResult += vect1[i] * vect2[i];
	}
	g_mutex.lock();
	result += localResult;
	g_mutex.unlock();
}

double scalarProductParallel(const double* vect1, const double* vect2, int size)
{
    double result = 0.0;
    std::thread thr1(
        productThreadRoutine, vect1, vect2, 0, size / 4, std::ref(result)
    );
    std::thread thr2(
        productThreadRoutine, vect1, vect2, size / 4, size / 2, std::ref(result)
    );
    std::thread thr3(
        productThreadRoutine, vect1, vect2, size / 2, 3 * size / 4, std::ref(result)
    );
    std::thread thr4(
        productThreadRoutine, vect1, vect2, 3 * size / 4, size, std::ref(result)
    );
    thr1.join();
    thr2.join();
    thr3.join();
    thr4.join();
    return result;
}

double scalarProductTbb(const double* vect1, const double* vect2, int size)
{
    tbb::task_scheduler_init init;
    double result = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size),
        0.0, [&vect1, &vect2](const tbb::blocked_range<size_t>& r, double init) -> double {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            init += vect1[i] * vect2[i];
        }
        return init;
    }, [](double x, double y) -> double {
        return x + y;
    });
    return result;
}

void initializeVector(double* vect, int count) {
	for (size_t i = 0; i < count; ++i) {
		if (rand() % 2) {
			vect[i] = -1.0 * (double)rand() / (double)RAND_MAX;
		} else {
			vect[i] = (double)rand() / (double)RAND_MAX;
		}
	}
}

int main() {
	srand((unsigned int)time(nullptr));

	const int SIZE = 10'000'000;
	double* vect1 = new double[SIZE], *vect2 = new double[SIZE];
	initializeVector(vect1, SIZE);
	initializeVector(vect2, SIZE);

	auto startSequential = std::chrono::high_resolution_clock::now();
	double productSequential = scalarProductSequential(vect1, vect2, SIZE);
	auto endSequential = std::chrono::high_resolution_clock::now();

    auto startStdThreads = std::chrono::high_resolution_clock::now();
    double productParallel = scalarProductParallel(vect1, vect2, SIZE);
	auto endStdThread = std::chrono::high_resolution_clock::now();

    auto startTbb = std::chrono::high_resolution_clock::now();
    double productTbb = scalarProductTbb(vect1, vect2, SIZE);
    auto endTbb = std::chrono::high_resolution_clock::now();

    std::cout << "Sequential: "
        << std::chrono::duration<double>(endSequential - startSequential).count()
        << " Product: " << productSequential << std::endl
        << "Parallel (std::thread): "
        << std::chrono::duration<double>(endStdThread - startStdThreads).count()
        << " Product: " << productParallel << std::endl
        << "Parallel (TBB): "
        << std::chrono::duration<double>(endTbb - startTbb).count()
        << " Product: " << productTbb << std::endl;

	delete[] vect1;
	delete[] vect2;
	return 0;
}
