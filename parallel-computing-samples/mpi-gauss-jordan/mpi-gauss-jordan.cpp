
#include <fstream>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Missing augmented matrix input file" << std::endl;
		exit(1);
	}

	const char* inputFile = argv[1];
	const unsigned int dim = 4;
	double matrAB[dim + 1][dim + 1] = { 0.0 };
	double inputAB[dim + 1][dim + 1] = { 0.0 };
	int leadRowNo = -1;
	int procNum, procRank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

	if (procNum == 1)
	{
		MPI_Finalize();
		exit(1);
	}

	if (procRank == 0)
	{
		std::ifstream ifs;
		ifs.open(inputFile);
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim + 1; j++)
			{
				ifs >> matrAB[i][j];
				inputAB[i][j] = matrAB[i][j];
			}
		}
		ifs.close();

		for (int i = 0; i < dim; i++)
		{
			if (i != 0)
			{
				std::cout << std::endl;
			}
			for (int j = 0; j < dim + 1; j++)
			{
				if (j == dim)
				{
					std::cout << "| " << matrAB[i][j];
				}
				else
				{
					std::cout << matrAB[i][j] << ' ';
				}
			}
		}
		std::cout << std::endl;
	}

	MPI_Bcast(inputAB, (dim + 1) * (dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (procRank == 0)
	{
		for (int rowIdx = 0; rowIdx < dim; rowIdx++)
		{
			if (rowIdx > 0)
			{
				for (int updatedRow = rowIdx; updatedRow < dim; updatedRow++)
				{
					MPI_Status recvStat;
					MPI_Recv(matrAB[updatedRow], dim + 1, MPI_DOUBLE,
						MPI_ANY_SOURCE, updatedRow, MPI_COMM_WORLD, &recvStat);
				}
			}
			leadRowNo = rowIdx;
			MPI_Bcast(&leadRowNo, 1, MPI_INT, 0, MPI_COMM_WORLD);
			double alpha = matrAB[leadRowNo][leadRowNo];
			for (int colIdx = 0; colIdx < dim + 1; colIdx++)
			{
				matrAB[rowIdx][colIdx] /= alpha;
			}
			MPI_Bcast(matrAB, (dim + 1)*(dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	else
	{
		for (int cnt = 0; cnt < dim; cnt++)
		{
			MPI_Bcast(&leadRowNo, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(matrAB, (dim + 1)*(dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			int rankRowNo = leadRowNo + procRank;
			if (rankRowNo >= dim)
			{
				MPI_Barrier(MPI_COMM_WORLD);
				continue;
			}
			bool done = false;
			while (!done)
			{
				for (int colIdx = dim; colIdx >= 0; colIdx--)
				{
					matrAB[rankRowNo][colIdx] -= matrAB[leadRowNo][colIdx] * matrAB[rankRowNo][leadRowNo];
				}
				MPI_Send(matrAB[rankRowNo], dim + 1, MPI_DOUBLE, 0, rankRowNo, MPI_COMM_WORLD);
				rankRowNo += procNum - 1;
				if (rankRowNo >= dim)
				{
					done = true;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	if (procRank == 0)
	{
		for (int rowIdx = dim - 1; rowIdx >= 0; rowIdx--)
		{
			if (rowIdx < dim - 1)
			{
				for (int updRowIdx = rowIdx; updRowIdx >= 0; updRowIdx--)
				{
					MPI_Status recvStat;
					MPI_Recv(matrAB[updRowIdx], dim + 1, MPI_DOUBLE,
						MPI_ANY_SOURCE, updRowIdx, MPI_COMM_WORLD, &recvStat);
				}
			}
			MPI_Bcast(matrAB, (dim + 1)*(dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	else
	{
		for (int rowIdx = dim - 1; rowIdx >= 0; rowIdx--)
		{
			MPI_Bcast(matrAB, (dim + 1)*(dim + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			int rankRowNo = rowIdx - procRank;
			if (rankRowNo < 0)
			{
				MPI_Barrier(MPI_COMM_WORLD);
				continue;
			}
			bool done = false;
			while (!done)
			{
				for (int colIdx = dim; colIdx >= rankRowNo + 1; colIdx--)
				{
					matrAB[rankRowNo][colIdx] -= matrAB[rowIdx][colIdx] * matrAB[rankRowNo][rowIdx];
				}
				MPI_Send(matrAB[rankRowNo], dim + 1, MPI_DOUBLE, 0, rankRowNo, MPI_COMM_WORLD);
				rankRowNo -= procNum - 1;
				if (rankRowNo < 0)
				{
					done = true;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	if (procRank == 0)
	{
		for (int i = 0; i < dim; i++)
		{
			std::cout << std::endl;
			for (int j = 0; j < dim + 1; j++)
			{
				std::cout << matrAB[i][j] << ' ';
			}
		}
		std::cout << std::endl << " x = (";
		for (int i = 0; i < dim; i++)
		{
			if (i == dim - 1)
			{
				std::cout << " " << matrAB[i][dim] << " )";
			}
			else
			{
				std::cout << " " << matrAB[i][dim] << " ;";
			}
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (procRank == 0)
	{
		double result[dim] = { 0.0 };
		std::cout << std::endl << " b = (";
		for (int i = 0; i < dim; i++)
		{
			MPI_Status recvStat;
			MPI_Recv(&result[i], 1, MPI_DOUBLE, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &recvStat);
		}
		for (int i = 0; i < dim; i++)
		{
			if (i == dim - 1)
			{
				std::cout << " " << result[i] << " )";
			}
			else
			{
				std::cout << " " << result[i] << " ;";
			}
		}
		std::cout << std::endl;
	}
	else
	{
		int rankRowNo = procRank - 1;
		if (rankRowNo > dim)
		{
			std::cout << "Rank #" << procRank << " has finished" << std::endl;
			MPI_Finalize();
			return 0;
		}
		bool done = false;
		while (!done)
		{
			double component = 0;
			for (int i = 0; i < dim; i++)
			{
				component += inputAB[rankRowNo][i] * matrAB[i][dim];
			}
			MPI_Send(&component, 1, MPI_DOUBLE, 0, rankRowNo, MPI_COMM_WORLD);
			rankRowNo += procNum - 1;
			if (rankRowNo >= dim)
			{
				done = true;
			}
		}
	}
	std::cout << "Rank #" << procRank << " has finished" << std::endl;
	MPI_Finalize();
}
