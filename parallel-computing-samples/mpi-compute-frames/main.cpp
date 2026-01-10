
#include <iostream>
#include <mpi.h>
#include <Windows.h>

void pretendSomeComputing(unsigned int millisecondsToSleep) {
    Sleep(millisecondsToSleep);
}

void setRandomSeed(const int rank) {
    srand((unsigned int)rank);
}

int generateFrame() {
    return rand() % 1000 + 1;
}

int initialSendFrames(const int totalFrames, const int processes) {
    int framesLeftToSend = totalFrames;
    for (int processToSendTo = 1; processToSendTo < processes; processToSendTo++) {
        if (framesLeftToSend) {
            int frameStub = generateFrame();
            MPI_Send(&frameStub, 1, MPI_INT, processToSendTo, 0, MPI_COMM_WORLD);
            framesLeftToSend--;
            std::cout << "Sending a frame to the process #"
                << processToSendTo << " [" << frameStub << "ms]. "
                << "Frames to send: " << framesLeftToSend << std::endl;
        } else {
            int stopStub = 0;
            MPI_Send(&stopStub, 1, MPI_INT, processToSendTo, 0, MPI_COMM_WORLD);
            std::cout << "Sending a stop stub to the process #"
                << processToSendTo << std::endl;
        }
    }
    return framesLeftToSend;
}

void waitForFrames(
    const int totalFrames, const int framesLeftToSend, const int processes
) {
    int framesComputed = 0, framesToSend = framesLeftToSend;
    while (framesComputed < totalFrames) {
        for (int process = 1; process < processes; process++) {
            int frameIsReady = 0;
            MPI_Status probeStatus;
            MPI_Iprobe(
                process, MPI_ANY_TAG, MPI_COMM_WORLD, &frameIsReady, &probeStatus
            );
            if (frameIsReady) {
                int computedFrame = 0;
                MPI_Status recvStatus;
                MPI_Recv(
                    &computedFrame,
                    1,
                    MPI_INT,
                    process,
                    MPI_ANY_TAG,
                    MPI_COMM_WORLD,
                    &recvStatus
                );
                framesComputed++;
                std::cout << "Receiving a frame from the process #"
                    << process << " [" << computedFrame << "ms]. "
                    << "Computed frames: " << framesComputed
                    << ". Frames to compute: " << (totalFrames - framesComputed)
                    << ". Frames to send: " << framesToSend << std::endl;

                if (framesToSend) {
                    int frameStub = generateFrame();
                    MPI_Send(&frameStub, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                    framesToSend--;
                    std::cout << "Sending a frame to the process #"
                        << process << " [" << frameStub << "ms]. "
                        << "Frames to send: " << framesToSend << std::endl;
                }
                else {
                    int stopStub = 0;
                    MPI_Send(&stopStub, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                    std::cout << "Sending a stop stub to the process #"
                        << process << std::endl;
                }
            }
        }
    }
}

void workerRoutine(const int rank) {
    int frameStub = 0;
    do {
        MPI_Status recvStatus;
        MPI_Recv(&frameStub, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);
        if (frameStub) {
            if (frameStub > 0) {
                pretendSomeComputing(frameStub);
            }
            MPI_Send(&frameStub, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
        }
    } while (frameStub);
}

int main(int argc, char* argv[]) {
	int procRank, procNum;
    const int FRAMES_TO_COMPUTE = 10;
    int framesToCompute = FRAMES_TO_COMPUTE;
    if (argc > 1) {
        framesToCompute = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    if (procRank == 0) {
        int framesLeftToSend = initialSendFrames(framesToCompute, procNum);
        waitForFrames(framesToCompute, framesLeftToSend, procNum);
    } else {
        workerRoutine(procRank);
    }
    MPI_Finalize();
    return 0;
}
