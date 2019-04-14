
#include <fstream>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[])
{
	const int dim = 6;

	if (argc < 3)
	{
		std::cout << "Missing matrices input files" << std::endl;
		exit(1);
	}
	const char* inputMatrA = argv[1];
	const char* inputMatrB = argv[2];

	double matrA[dim][dim] = { 0.0 };
	double matrB[dim][dim] = { 0.0 };
	double matrC[dim][dim] = { 0.0 };

	int procNum, procRank;
	int dims[2], periods[2];
	int coords[2];

	double subMatrA[2][2] = { 0.0 };
	double subMatrB[2][2] = { 0.0 };
	double subMatrC[2][2] = { 0.0 };

	for (int i = 0; i < 2; i++)
	{
		dims[i] = dim / 2;
		periods[i] = true;
	}
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

	if (procNum != (dim / 2)*(dim / 2))
	{
		if (procRank == 0)
		{
			std::cout << std::endl << "Test can be run on "\
				<< (dim / 2)*(dim / 2) << " ranks only";
		}
		MPI_Finalize();
		return 1;
	}

	if (procRank == 0)
	{
		// Input matrix A
		std::ifstream ifs;
		ifs.open(inputMatrA);
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				ifs >> matrA[i][j];
			}
		}
		ifs.close();
		MPI_Bcast(matrA, dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrB, dim*dim, MPI_DOUBLE, 1, MPI_COMM_WORLD);
	}
	else
	{
		if (procRank == 1)
		{
			// Input matrix B
			std::ifstream ifs;
			ifs.open(inputMatrB);
			for (int i = 0; i < dim; i++)
			{
				for (int j = 0; j < dim; j++)
				{
					ifs >> matrB[i][j];
				}
			}
			ifs.close();
			MPI_Bcast(matrB, dim*dim, MPI_DOUBLE, 1, MPI_COMM_WORLD);
			MPI_Bcast(matrA, dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Bcast(matrA, dim*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(matrB, dim*dim, MPI_DOUBLE, 1, MPI_COMM_WORLD);
		}
	}

	// Print A|B
	if (procRank == 0)
	{
		for (int i = 0; i < dim; i++)
		{
			if (i != 0)
			{
				std::cout << std::endl;
			}
			for (int j = 0; j < 2 * dim + 1; j++)
			{
				if (j < dim)
				{
					std::cout << matrA[i][j] << ' ';
				}
				if (j == dim)
				{
					std::cout << '|';
				}
				if (j > dim)
				{
					std::cout << ' ' << matrB[i][j - (dim + 1)];
				}
			}
		}
	}

	double startTime = MPI_Wtime();
	// Create processes grid
	MPI_Comm comm_cart;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &comm_cart);

	// Passing submatrices to the grid ranks
	MPI_Cart_coords(comm_cart, procRank, 2, coords);
	int coordx = coords[0], coordy = coords[1];
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			subMatrA[i][j] = matrA[2 * coordx + i][2 * coordy + j];
			subMatrB[i][j] = matrB[2 * coordx + i][2 * coordy + j];
			subMatrC[i][j] = 0;
		}
	}

	// Initial matrices alignment
	if (coordx > 0)
	{
		// Shift A submatrices by i-1 positions left
		int left = -1, right = -1;
		int disp = (coordx + 1) - 1;
		MPI_Cart_shift(comm_cart, 1, -disp, &right, &left);
		MPI_Status recvStat;
		MPI_Sendrecv_replace(subMatrA, 4, MPI_DOUBLE, left, 1, right, 1, comm_cart, &recvStat);
	}

	if (coordy > 0)
	{
		// Shift B submatrices by j-1 positions up
		int up = -1, down = -1;
		int disp = (coordy + 1) - 1;
		MPI_Cart_shift(comm_cart, 0, disp, &down, &up);
		MPI_Status recvStat;
		MPI_Sendrecv_replace(subMatrB, 4, MPI_DOUBLE, down, 0, up, 0, comm_cart, &recvStat); 
	}

	// Main loop
	for (int cnt = 0; cnt < dim / 2; cnt++)
	{
		// Multiplication of the A and B submatrices
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				double s = 0;
				for (int k = 0; k < 2; k++)
				{
					s += subMatrA[i][k] * subMatrB[k][j];
				}
				subMatrC[i][j] += s;
			}
		}

		MPI_Status recvStat;
		int left = -1, right = -1;
		int up = -1, down = -1;

		// Shift A submatrices by 1 position left
		MPI_Cart_shift(comm_cart, 1, -1, &right, &left);
		MPI_Sendrecv_replace(subMatrA, 4, MPI_DOUBLE, left, 1, right, 1, comm_cart, &recvStat);

		// Shift B submatrices by 1 position up
		MPI_Cart_shift(comm_cart, 0, 1, &down, &up);
		MPI_Sendrecv_replace(subMatrB, 4, MPI_DOUBLE, down, 0, up, 0, comm_cart, &recvStat);
	}

	// Gather C=A*B
	if (procRank == 0)
	{
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				matrC[i][j] = subMatrC[i][j];
			}
		}

		for (int rank = 1; rank < procNum; rank++)
		{
			MPI_Status recvStatSub, recvStatCoords;
			MPI_Recv(subMatrC, 4, MPI_DOUBLE, rank, 0, comm_cart, &recvStatSub);
			MPI_Recv(coords, 2, MPI_INT, rank, 1, comm_cart, &recvStatCoords);
			coordx = coords[0], coordy = coords[1];
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					matrC[2 * coordx + i][2 * coordy + j] = subMatrC[i][j];
				}
			}
		}
		std::cout << std::endl;
		for (int i = 0; i < dim; i++)
		{
			std::cout << std::endl;
			for (int j = 0; j < dim; j++)
			{
				std::cout << matrC[i][j] << ' ';
			}
		}
		std::cout << std::endl << "Elapsed time: " << MPI_Wtime() - startTime;
	}
	else
	{
		MPI_Send(subMatrC, 4, MPI_DOUBLE, 0, 0, comm_cart);
		MPI_Send(coords, 2, MPI_INT, 0, 1, comm_cart);
	}
	MPI_Finalize();
}
