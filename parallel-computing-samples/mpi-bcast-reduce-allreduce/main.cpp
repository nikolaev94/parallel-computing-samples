
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <mpi.h>

void myMPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if (rank == root) {
		for (int dst = 0; dst < size; dst++) {
			if (dst == root) {
				continue;
			}
			MPI_Send(buffer, count, datatype, dst, root, comm);
		}
	} else {
		MPI_Status status;
		MPI_Recv(buffer, count, datatype, root, MPI_ANY_TAG, comm, &status);
	}
}

template<class T>
int myMPI_Reduce(
    T* sendbuf,
    T* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm comm
) {
    const int subvectorSize = 1000;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if (rank == root) {
		MPI_Status status;
		MPI_Sendrecv(
            sendbuf,
            count,
            datatype,
            0,
            root,
            recvbuf,
            count,
            datatype,
            0,
            root,
            MPI_COMM_SELF,
            &status
        );
	}
    for (int partIndex = 0; partIndex < count / subvectorSize + 1; partIndex++)
    {
        int amountToProcess = subvectorSize;
        int startIndex = subvectorSize * partIndex;
        if (partIndex == count / subvectorSize)
        {
            amountToProcess = count % subvectorSize;
        }
        if (rank == root)
        {
            for (int processToRecvFrom = 0; processToRecvFrom < size; processToRecvFrom++) {
                if (processToRecvFrom == root) {
                    continue;
                }
                T* received = new T[amountToProcess];
                MPI_Status status;
                MPI_Recv(
                    received,
                    amountToProcess,
                    datatype,
                    processToRecvFrom,
                    partIndex,
                    comm,
                    &status
                );
                for (int offset = 0; offset < amountToProcess; offset++)
                {
                    switch (op)
                    {
                        case MPI_MIN: {
                            if (received[offset] < recvbuf[startIndex + offset])
                            {
                                recvbuf[startIndex + offset] = received[offset];
                            }
                            break;
                        }
                        case MPI_MAX: {
                            if (received[offset] > recvbuf[startIndex + offset])
                            {
                                recvbuf[startIndex + offset] = received[offset];
                            }
                            break;
                        }
                        case MPI_SUM: {
                            recvbuf[startIndex + offset] += received[offset];
                            break;
                        }
                    }
                }
                delete[] received;
                received = nullptr;
            }
        }
        else
        {
            MPI_Send(sendbuf + startIndex, amountToProcess, datatype, root, partIndex, comm);
        }
    }
    return 0;
}

template<class T>
int myMPI_Allreduce(
    T* sendbuf,
    T* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
) {
    myMPI_Reduce<T>(sendbuf, recvbuf, count, datatype, op, 0, comm);
    myMPI_Bcast(recvbuf, count, datatype, 0, comm);
    return 0;
}

template<class T>
void setVectorsWithRandom(size_t vectorSize, T* vectorRef, T* vectorTest) {
    for (size_t i = 0; i < vectorSize; ++i) {
        T randomNumber = (T)rand() / (T)RAND_MAX;
        vectorTest[i] = vectorRef[i] = (rand() % 2) ? randomNumber : (T)(-1.0) * randomNumber;
    }
}

template<class T>
void setVectorsWithZeroes(size_t vectorSize, T* vectorRef, T* vectorTest) {
    for (size_t i = 0; i < vectorSize; ++i) {
        vectorRef[i] = vectorTest[i] = (T)0.0;
    }
}

template<class T>
T calculateVectorError(size_t vectorSize, T* vectorRef, T* vectorTest) {
    T maxError = std::numeric_limits<T>::epsilon();
    for (size_t i = 0; i < vectorSize; ++i) {
        T error = std::abs(vectorRef[i] - vectorTest[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    return maxError;
}

void bcastTest(size_t vectorSize) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* vectorToBcastRef = new double[vectorSize];
    double* vectorToBcastTest = new double[vectorSize];

    if (rank == 0) {
        setVectorsWithRandom(vectorSize, vectorToBcastRef, vectorToBcastTest);
    }
    else {
        setVectorsWithZeroes(vectorSize, vectorToBcastRef, vectorToBcastTest);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startBcastRef = MPI_Wtime();
    MPI_Bcast(vectorToBcastRef, vectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double endBcastRef = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double startBcastTest = MPI_Wtime();
    myMPI_Bcast(vectorToBcastTest, vectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double endBcastTest = MPI_Wtime();

    if (rank == 0) {
        std::cout << std::endl << " MPI_Bcast " << std::endl;
        std::cout << "Max error between MPI_Bcast and myMPI_Bcast vectors[rank #"
            << rank << "]: "
            << calculateVectorError(vectorSize, vectorToBcastRef, vectorToBcastTest)
            << std::endl;
        std::cout << "MPI_Bcast time: " << (endBcastRef - startBcastRef)
            << " sec." << std::endl;
        std::cout << "myMPI_Bcast time: " << (endBcastTest - startBcastTest)
            << " sec." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == size - 1) {
        std::cout << "Max error between MPI_Bcast and myMPI_Bcast vectors[rank #"
            << rank << "]: "
            << calculateVectorError(vectorSize, vectorToBcastRef, vectorToBcastTest)
            << std::endl;
    }

    delete[] vectorToBcastTest;
    delete[] vectorToBcastRef;
}

void reduceTest(size_t vectorSize, MPI_Op op) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float* sendVectorForReduceRef = new float[vectorSize];
    float* recvVectorForReduceRef = new float[vectorSize];
    float* sendVectorForReduceTest = new float[vectorSize];
    float* recvVectorForReduceTest = new float[vectorSize];

    setVectorsWithRandom(vectorSize, sendVectorForReduceRef, sendVectorForReduceTest);
    setVectorsWithZeroes(vectorSize, recvVectorForReduceRef, recvVectorForReduceTest);

    MPI_Barrier(MPI_COMM_WORLD);
    double startReduceRef = MPI_Wtime();
    MPI_Reduce(
        sendVectorForReduceRef,
        recvVectorForReduceRef,
        vectorSize,
        MPI_FLOAT,
        op,
        0,
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    double endReduceRef = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double startReduceTest = MPI_Wtime();
    myMPI_Reduce<float>(
        sendVectorForReduceTest,
        recvVectorForReduceTest,
        vectorSize,
        MPI_FLOAT,
        op,
        0,
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    double endReduceTest = MPI_Wtime();


    if (rank == 0) {
        std::cout << std::endl << " MPI_Reduce " << std::endl;
        std::cout << '[' << op << ']'
            << " - Max error between MPI_Reduce and myMPI_Reduce vectors[rank #"
            << rank << "]: "
            << calculateVectorError(vectorSize, recvVectorForReduceRef, recvVectorForReduceTest)
            << std::endl;
        std::cout << '[' << op << ']'
            << " - MPI_Reduce time: " << (endReduceRef - startReduceRef)
            << " sec." << std::endl;
        std::cout << '[' << op << ']'
            << " - myMPI_Reduce time: " << (endReduceTest - startReduceTest)
            << " sec." << std::endl;
    }

    delete[] sendVectorForReduceTest;
    delete[] recvVectorForReduceTest;
    delete[] sendVectorForReduceRef;
    delete[] recvVectorForReduceRef;
}

void allreduceTest(size_t vectorSize, MPI_Op op)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float* sendVectorForAllreduceRef = new float[vectorSize];
    float* recvVectorForAllreduceRef = new float[vectorSize];
    float* sendVectorForAllreduceTest = new float[vectorSize];
    float* recvVectorForAllreduceTest = new float[vectorSize];

    setVectorsWithRandom(vectorSize, sendVectorForAllreduceRef, sendVectorForAllreduceTest);
    setVectorsWithZeroes(vectorSize, recvVectorForAllreduceRef, recvVectorForAllreduceTest);

    MPI_Barrier(MPI_COMM_WORLD);
    double startAllreduceRef = MPI_Wtime();
    MPI_Allreduce(
        sendVectorForAllreduceRef,
        recvVectorForAllreduceRef,
        vectorSize,
        MPI_FLOAT,
        op,
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    double endAllreduceRef = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double startAllreduceTest = MPI_Wtime();
    myMPI_Allreduce<float>(
        sendVectorForAllreduceTest,
        recvVectorForAllreduceTest,
        vectorSize,
        MPI_FLOAT,
        op,
        MPI_COMM_WORLD
        );
    MPI_Barrier(MPI_COMM_WORLD);
    double endAllreduceTest = MPI_Wtime();

    if (rank == 0) {
        std::cout << std::endl << " MPI_Allreduce " << std::endl;
        std::cout << '[' << op << ']'
            << " - Max error between MPI_Allreduce and myMPI_Allreduce vectors[rank #"
            << rank << "]: "
            << calculateVectorError(
                vectorSize,
                recvVectorForAllreduceRef,
                recvVectorForAllreduceTest
            ) << std::endl;
        std::cout << '[' << op << ']' << " - MPI_Allreduce time: "
            << (endAllreduceRef - startAllreduceRef)
            << " sec." << std::endl;
        std::cout << '[' << op << ']' << " - myMPI_Allreduce time: "
            << (endAllreduceTest - startAllreduceTest)
            << " sec." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == size - 1) {
        std::cout << '[' << op << ']'
            << " - Max error between MPI_Allreduce and myMPI_Allreduce vectors[rank #"
            << rank << "]: "
            << calculateVectorError(
                vectorSize,
                recvVectorForAllreduceRef,
                recvVectorForAllreduceTest
            ) << std::endl;
    }
}

int main(int argc, char* argv[]) {
	const int SIZE = 30000;
    int vectorBcastSize(SIZE), vectorReduceSize(SIZE), vectorAllreduceSize(SIZE);

    if (argc > 3) {
        vectorBcastSize = atoi(argv[1]);
        vectorReduceSize = atoi(argv[2]);
        vectorAllreduceSize = atoi(argv[3]);
    }
    else if (argc > 1)
    {
        vectorBcastSize = vectorReduceSize = vectorAllreduceSize = atoi(argv[1]);
    }

	int rank, size;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	srand((unsigned int)time(nullptr) + rank);
    bcastTest(vectorBcastSize);

    MPI_Op ops[] = { MPI_MIN, MPI_MAX, MPI_SUM };
    for (MPI_Op op : ops) {
        reduceTest(vectorReduceSize, op);
    }

    for (MPI_Op op : ops) {
        allreduceTest(vectorAllreduceSize, op);
    }
	MPI_Finalize();
	return 0;
}
